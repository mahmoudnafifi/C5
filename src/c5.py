"""
##### Copyright 2021 Google LLC. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import torch.nn as nn
import torch
from src import ops
import math

# uncomment this part if torch 1.8 or higher is used.
#import torch.fft as fft


class network(nn.Module):
  def __init__(self, input_size=64, learn_g=False, data_num=7, device='cuda'):
    """ C5 Network constructor.

    Args:
      input_size: histogram input size (number of bins).
      learn_g: boolean flat to learn the gain multiplier G; default is false.
      data_num: number of input data including the original histogram (m in
        the paper). If data_m = 3, for example, the network will have three
        encoders, one for the main histogram and the remaining encodres for the
        additional histograms. Default is 7.
      device: network allocation ('cuda' or 'cpu'); default is 'cuda'.

    Returns:
      C5 network object with the selected settings.
    """

    super(network, self).__init__()

    assert (input_size - 2 ** math.ceil(math.log2(input_size)) == 0 and
            input_size >= 16)
    assert (data_num >= 1)

    self.input_size = input_size
    self.device = device
    self.data_num = data_num
    self.learn_g = learn_g
    self.u_coord, self.v_coord = ops.get_uv_coord(self.input_size,
                                                  tensor=True,
                                                  device=self.device)

    initial_conv_depth = 8  # output channel of the first encoder layer
    max_conv_depth = 32  # maximum output channels can be produced by any cov
    network_depth = 4  # number of encoder/decoder layers
    assert (network_depth == 4)
    intermediate_sz = pow(2, math.ceil(math.log2(input_size)) - network_depth)
    assert (max_conv_depth > initial_conv_depth * 2)
    network_depth = int(math.ceil(math.log2(input_size)) -
                        math.log2(intermediate_sz))

    # Encoder-decoder C5 net
    # encoder
    self.encoder = Encoder(in_channel=4, first_conv_depth=initial_conv_depth,
                           max_conv_depth=max_conv_depth,
                           data_num=data_num,
                           depth=network_depth, normalization=True,
                           norm_type='BN')
    self.encoder.to(device=device)

    # B bias decoder
    self.decoder_B = Decoder(output_channels=1,
                             encoder_first_conv_depth=initial_conv_depth,
                             encoder_max_conv_depth=max_conv_depth,
                             normalization=True,
                             norm_type='IN',
                             depth=network_depth)
    self.decoder_B.to(device=device)

    # F decoder
    self.decoder_F = Decoder(output_channels=2,
                             encoder_first_conv_depth=initial_conv_depth,
                             encoder_max_conv_depth=max_conv_depth,
                             normalization=True,
                             norm_type='IN',
                             depth=network_depth)
    self.decoder_F.to(device=device)

    if self.learn_g:
      # G decoder
      self.decoder_G = Decoder(output_channels=1,
                               encoder_first_conv_depth=initial_conv_depth,
                               encoder_max_conv_depth=max_conv_depth,
                               normalization=True,
                               norm_type='IN',
                               depth=network_depth)
      self.decoder_G.to(device=device)
    else:
      self.decoder_G = None

    # bottleneck
    self.bottleneck = DoubleConvBlock(
      in_depth=min(
        initial_conv_depth * 2 ** (network_depth - 1), max_conv_depth),
      mid_depth=min(initial_conv_depth * 2 ** network_depth, max_conv_depth),
      out_depth=min(
        initial_conv_depth * 2 ** (network_depth - 1), max_conv_depth),
      pooling=False, normalization_block='Second', normalization=False,
      norm_type='IN')
    self.bottleneck.to(device=device)

    self.softmax = nn.Softmax(dim=-1)

  def forward(self, N, model_in_N):
    """ Forward function of C5 network

    Args:
      N: input histogram(s)
      model_in_N: input histogram(s) concatenated with the additional
        histogram(s) over the second axis (i.e., dim = 1).

    Returns:
      rgb: predicted illuminant rgb colors in the format b x 3, where b is
        the batch-size.
      P: illuminant heat map as described in Eq. 4 (or Eq. 10 if G is used)
        in the paper.
      F: conv filter of the CCC model emitted by the network.
      B: bias of the CCC model emitted by the network.
      G: gain multiplier of the CCC model emitted by the network (if learn_g
        is false, then G is None).
    """

    assert (N.shape[-1] == model_in_N.shape[-1] and
            N.shape[-2] == model_in_N.shape[-2])
    assert (N.shape[-1] == self.input_size and
            N.shape[-2] == self.input_size)

    latent, encoder_output = self.encoder(model_in_N)
    latent = self.bottleneck(latent)
    B = self.decoder_B(latent, encoder_output)
    B = torch.squeeze(B)

    F = self.decoder_F(latent, encoder_output)

    if self.learn_g:
      G = self.decoder_G(latent, encoder_output)
      G = torch.squeeze(G)
    else:
      G = None

    # convolving over the histogram in the frequency domain
    # comment out this part if torch 1.8 or higher is used.
    N_fft = torch.rfft(N[:, :2, :, :], 2, onesided=False)
    F_fft = torch.rfft(F, 2, onesided=False)
    N_after_conv = torch.irfft(ops.complex_multiplication(N_fft, F_fft), 2,
                               onesided=False)

    # uncomment this part if torch 1.8 or higher is used.
    # N_fft = fft.rfft2(N[:, :2, :, :])
    # F_fft = fft.rfft2(F)
    # N_after_conv = fft.irfft2(N_fft * F_fft)



    # adding the bias
    N_after_conv = torch.sum(N_after_conv, dim=1)
    if G is not None:
      N_after_bias = (G * N_after_conv) + B
    else:
      N_after_bias = N_after_conv + B

    # generating the heat map
    N_after_bias = torch.clamp(N_after_bias, -100, 100)
    P = self.softmax(torch.reshape(N_after_bias, (N_after_bias.shape[0], -1)))
    P = torch.reshape(P, N_after_bias.shape)

    # producing the RGB illuminant estimate
    u = torch.sum(P * self.u_coord, dim=[-1, -2])
    v = torch.sum(P * self.v_coord, dim=[-1, -2])
    u, v = ops.from_coord_to_uv(self.input_size, u, v)
    rgb = ops.uv_to_rgb(torch.stack([u, v], dim=1), tensor=True)
    return rgb, P, F, B, G


class ConvBlock(nn.Module):
  """ Conv layer block """

  def __init__(self, kernel, in_depth, conv_depth, stride=1, padding=1,
               normalization=False, norm_type='BN', pooling=False,
               bias_initialization='zeros', activation=True, dilation=1,
               return_before_pooling=False):
    """ ConvBlock constructor

    Args:
      kernel: kernel size (int)
      in_depth: depth of input tensor
      conv_depth: number of out channels produced by the convolution
      stride: stide of the convolution; default is 1.
      padding: zero-padding added to both sides before the convolution
        operation; default is 1.
      normalization: boolean flag to apply normalization after the conv;
        default is false.
      norm_type: normalization operation: 'BN' for batch-norm (default),
        'IN' for instance normalization.
      pooling: boolean flag to apply a 2 x 2 max-pooling with stride of 2
        before returning the final result; default is false.
      bias_initialization: bias initialization: 'zeros' (default) or 'ones'.
      activation: boolean flag to apply a leaky ReLU activation; default is
        true.
      dilation: spacing between conv kernel elements; default is 1.
      return_before_pooling: boolean flag to return the tensor before
        applying max-pooling (if 'pooling' is true); default is false.

    Returns:
      ConvBlock object with the selected settings.
    """

    super(ConvBlock, self).__init__()

    conv = torch.nn.Conv2d(in_depth, conv_depth, kernel, stride=stride,
                           dilation=dilation, padding=padding,
                           padding_mode='replicate')
    torch.nn.init.kaiming_normal_(conv.weight)  # He initialization
    if bias_initialization == 'ones':
      torch.nn.init.ones_(conv.bias)
    elif bias_initialization == 'zeros':
      torch.nn.init.zeros_(conv.bias)
    else:
      raise NotImplementedError

    if activation:
      self.activation = torch.nn.LeakyReLU(inplace=False)
    else:
      self.activation = None
    if normalization:
      if norm_type == 'BN':
        self.normalization = torch.nn.BatchNorm2d(conv_depth, affine=True)
      elif norm_type == 'IN':
        self.normalization = torch.nn.InstanceNorm2d(conv_depth,
                                                     affine=False)
      else:
        raise NotImplementedError
    else:
      self.normalization = None
    self.conv = conv
    if pooling:
      self.pooling = torch.nn.MaxPool2d(2, stride=2)
    else:
      self.pooling = None
    self.return_before_pooling = return_before_pooling

  def forward(self, x):
    """ Forward function of ConvBlock module

    Args:
      x: input tensor.

    Returns:
      y: processed tensor.
    """
    x = self.conv(x)
    if self.normalization is not None:
      x = self.normalization(x)
    if self.activation is not None:
      x = self.activation(x)
    if self.pooling is not None:
      y = self.pooling(x)
    else:
      y = x
    if self.return_before_pooling:
      return y, x
    else:
      return y


class DoubleConvBlock(nn.Module):
  """ Double conv layers block """

  def __init__(self, in_depth, out_depth, mid_depth=None, kernel=3, stride=1,
               padding=None, dilation=None, normalization=False, norm_type='BN',
               pooling=True, return_before_pooling=False,
               normalization_block='Both'):
    """ DoubleConvBlock constructor

    Args:
      in_depth: depth of input tensor
      out_depth: number of out channels produced by the second convolution
      mid_depth: number of out channels produced by the first convolution;
        default is mid_depth = out_depth.
      kernel: kernel size (int); default is 3.
      stride: stide of the convolution; default is 1.
      padding: zero-padding added to both sides before the convolution
        operations; default is [1, 1].
      dilation: spacing between elements of each conv kernel; default is [1, 1].
      normalization: boolean flag to apply normalization after the conv;
        default is false.
      norm_type: normalization operation: 'BN' for batch-norm (default),
        'IN' for instance normalization.
      pooling: boolean flag to apply a 2 x 2 max-pooling with stride of 2
        before returning the final result; default is false.
      return_before_pooling: boolean flag to return the tensor before
        applying max-pooling (if 'pooling' is true); default is false.
      normalization_block: if normalization flag is set to true; this
        variable controls when to apply the normalization process. It can be:
        'Both' (apply normalization after both conv layers), 'First', or
        'Second'.

    Returns:
      DoubleConvBlock object with the selected settings.
    """

    super().__init__()
    if padding is None:
      padding = [1, 1]
    if dilation is None:
      dilation = [1, 1]
    if mid_depth is None:
      mid_depth = out_depth
    if normalization:
      if normalization_block == 'First':
        norm = [True, False]
      elif normalization_block == 'Second':
        norm = [False, True]
      elif normalization_block == 'Both':
        norm = [True, True]
      else:
        raise NotImplementedError
    else:
      norm = [False, False]
    self.double_conv_1 = ConvBlock(kernel=kernel, in_depth=in_depth,
                                   conv_depth=mid_depth, stride=stride,
                                   padding=padding[0], pooling=False,
                                   dilation=dilation[0],
                                   norm_type=norm_type,
                                   normalization=norm[0])
    self.double_conv_2 = ConvBlock(kernel=kernel, in_depth=mid_depth,
                                   conv_depth=out_depth, stride=stride,
                                   padding=padding[1], pooling=pooling,
                                   dilation=dilation[1],
                                   norm_type=norm_type,
                                   normalization=norm[1],
                                   return_before_pooling=
                                   return_before_pooling)

  def forward(self, x):
    """ Forward function of DoubleConvBlock module

    Args:
      x: input tensor

    Returns:
      y: processed tensor
    """

    x = self.double_conv_1(x)
    return self.double_conv_2(x)


class Flatten(nn.Module):
  """ Flattening """

  def forward(self, x):
    """ Forward function of Flatten module

    Args:
      x: input tensor with a total number of values = n

    Returns:
      an batch x n vector

    """

    x = x.view(x.size()[0], -1)
    return x


class CrossPooling(nn.Module):
  """ Cross pooling """

  def forward(self, x):
    """ Forward function of CrossPooling module.

    Args:
      x: a stack of (batch x channel x height x width) tensors on the last axis.

    Returns:
      A (batch x channel x height x width) tensor after applying max-pooling
        over the last axis.
    """

    x, _ = torch.max(x, dim=-1)
    return x


class Encoder(nn.Module):
  """ Encoder """

  def __init__(self, in_channel, first_conv_depth=48, max_conv_depth=32,
               data_num=7, normalization=False, norm_type='BN', depth=4):
    """ Encoder constructor

    Args:
      in_channel: number of channels of the input.
      first_conv_depth: output channels produced by the first encoder layer.
      max_conv_depth: maximum output channels can be produced by any cov in
        the encoder; default is 32.
      data_num: number of additional histograms + the input histogram (the
        value of m in the paper); default is 7.
      normalization: boolean flag to apply normalization in the encoder;
        default is false.
      norm_type: when 'normalization' is set to true, the value of this variable
        (i.e., norm_type) specifies which normalization process is applied.
        'BN' (default) refers to batch normalization and 'IN' refers to instance
        normalization.
      depth: number of encoder layers; default is 4.

    Returns:
      Encoder object with the selected settings.
    """

    super().__init__()
    self.encoders = nn.ModuleList([])
    self.data_num = data_num
    self.encoder_depth = depth
    # encodor merging blocks
    if self.data_num > 1:
      self.merge_layers = nn.ModuleList([])
      self.cross_pooling = CrossPooling()
    else:
      self.merge_layers = None
      self.cross_pooling = None

    for data_i in range(self.data_num):
      encoder_i = nn.ModuleList([])
      if self.data_num > 1:
        merge_layers_i = nn.ModuleList([])
      if data_i == 0:
        skip_connections = True
      else:
        skip_connections = False

      for block_j in range(self.encoder_depth):
        if block_j % 2 == 0 and normalization:
          norm = normalization
        else:
          norm = False

        if block_j == 0:
          in_depth = in_channel
        else:
          in_depth = first_conv_depth * (2 ** (block_j - 1))
        out_depth = min(first_conv_depth * (2 ** block_j), max_conv_depth)

        double_conv_block = DoubleConvBlock(
          in_depth=in_depth,
          out_depth=out_depth,
          normalization=norm, norm_type=norm_type,
          normalization_block='Second', return_before_pooling=skip_connections)

        encoder_i.append(double_conv_block)

        # if self.data_num > 1:
        if self.data_num > 1 and block_j < self.encoder_depth - 1:
          # add merging 1x1 conv layer
          merge_layer = ConvBlock(kernel=1,
                                  in_depth=2 * out_depth,
                                  conv_depth=out_depth,
                                  stride=1, padding=0,
                                  normalization=False,
                                  norm_type='BN', pooling=False)
          merge_layers_i.append(merge_layer)

      self.encoders.append(encoder_i)
      if self.data_num > 1:
        self.merge_layers.append(merge_layers_i)

  def forward(self, x):
    """ Forward function of Encoder module

    Args:
      x: input tensor in the format (batch x h x channel x height x
      width); where h refer to the index of each histogram (i.e., h = 0 is
      the input histogram, h = 1 is the first additional histogram, etc.).

    Returns:
      y: processed data by the encoder, which is the input to the bottleneck.
      skip_connection_data: a list of processed data by each encoder for
        u-net skipp connections; this will be used by the decoder.
    """
    assert (self.data_num == x.shape[1])
    skip_connection_data = []
    latent_x = []
    for encoder_block_i in range(self.encoder_depth):
      for data_j in range(self.data_num):
        if encoder_block_i == 0:
          if data_j == 0:
            curr_latent, latent_before_pooling = self.encoders[data_j][
              encoder_block_i](x[:, data_j, :, :, :])
            skip_connection_data.append(latent_before_pooling)
          else:
            curr_latent = self.encoders[data_j][
              encoder_block_i](x[:, data_j, :, :, :])
          latent_x.append(curr_latent)
        else:
          if data_j == 0:
            curr_latent, latent_before_pooling = self.encoders[data_j][
              encoder_block_i](latent_x[data_j])
            skip_connection_data.append(latent_before_pooling)
          else:
            curr_latent = self.encoders[data_j][encoder_block_i](
              latent_x[data_j])
          latent_x[data_j] = curr_latent
        if self.merge_layers is not None:
          if data_j == 0:
            stacked_latent = torch.unsqueeze(latent_x[data_j], dim=-1)
          else:
            stacked_latent = torch.cat([stacked_latent,
                                        torch.unsqueeze(latent_x[data_j],
                                                        dim=-1)], dim=-1)
      if self.merge_layers is not None:
        pooled_data = self.cross_pooling(stacked_latent)
        if encoder_block_i < (self.encoder_depth - 1):
          for data_j in range(self.data_num):
            latent_x[data_j] = self.merge_layers[data_j][
              encoder_block_i](torch.cat([latent_x[data_j], pooled_data],
                                         dim=1))

    if self.merge_layers is None:
      y = latent_x[0]
    else:
      y = pooled_data

    skip_connection_data.reverse()
    return y, skip_connection_data


class Decoder(nn.Module):
  """ Decoder """

  def __init__(self, output_channels, encoder_first_conv_depth=8,
               normalization=False, encoder_max_conv_depth=32,
               norm_type='IN', depth=4):
    """ Decoder constructor

    Args:
      output_channels: output channels of the last layer in the decoder.
      encoder_first_conv_depth: output channels produced by the first encoder
        layer; default is 8. This and 'encoder_max_conv_depth' variables are
        used to dynamically compute the output of each corresponding decoder
        layer.
      normalization: boolean flag to apply normalization in the decoder;
        default is false.
      encoder_max_conv_depth: maximum output channels can be produced by any cov
        in the encoder; default is 32. This and 'encoder_first_conv_depth'
        variables are used to dynamically compute the output of each
        corresponding decoder layer. This variable also is used to know the
        output of the bottleneck unite.
      norm_type: when 'normalization' is set to true, the value of this variable
        (i.e., norm_type) specifies which normalization process is applied.
        'BN' refers to batch normalization and 'IN' (default) refers to instance
        normalization.
      depth: number of encoder layers; default is 4.

    Returns:
      Decoder object with the selected settings.
    """
    super().__init__()
    self.decoder = nn.ModuleList([])
    self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear',
                                  align_corners=True)
    self.final_block = ConvBlock(kernel=3, activation=False, in_depth=int(
      encoder_first_conv_depth / 2), conv_depth=output_channels,
                                 stride=1, padding=1)
    for i in range(depth):
      mid_depth = int(min(encoder_first_conv_depth * 2 ** (depth - 1 - i),
                          encoder_max_conv_depth))
      out_depth = int(min(encoder_first_conv_depth * 2 ** (depth - 2 - i),
                          encoder_max_conv_depth))
      in_depth = int(min(encoder_first_conv_depth * 2 ** (depth - i),
                         encoder_max_conv_depth * 2))
      double_conv_block = DoubleConvBlock(
        in_depth=in_depth, out_depth=out_depth, mid_depth=mid_depth,
        normalization_block='Second',
        normalization=normalization, norm_type=norm_type, pooling=False)
      self.decoder.append(double_conv_block)

  def forward(self, x, encoder_output):
    """ Forward function of Encoder module

    Args:
      x: processed data by the bottleneck
      encoder_output: skipped data from the encoder layers

    Returns:
      tensor of one the CCC model components (i.e., F, B, or G) emitted by the
        network.

    """
    for decoder_block, e_i, i in zip(self.decoder, encoder_output,
                                     range(len(encoder_output))):
      x = self.upsampling(x)
      x = torch.cat([e_i, x], dim=1)
      x = decoder_block(x)
    return self.final_block(x)
