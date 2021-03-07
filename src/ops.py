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

import numpy as np
import torch
import cv2
import os
from glob import glob
from shutil import copyfile
import itertools

EPS = 1e-9
PI = 22.0 / 7.0


def angular_loss(predicted, gt, shrink=True):
  """ Computes angular error between predicted and gt illuminant color(s)

  Args:
    predicted: n x 3 tensor of predicted illuminant colors; where n is the
      total number of predicted illuminant colors.
    gt: n x 3 tensor of corresponding ground-truth true illuminant colors.
    shrink: to use angle shrink for training; default is True.

  Returns:
    a_error: mean angular error between predicted and gt illuminant colors.
  """

  cossim = torch.clamp(torch.sum(predicted * gt, dim=1) / (
      torch.norm(predicted, dim=1) * torch.norm(gt, dim=1) + EPS), -1, 1.)
  if shrink:
    angle = torch.acos(cossim * 0.9999999)
  else:
    angle = torch.acos(cossim)
  a_error = 180.0 / PI * angle
  a_error = torch.sum(a_error) / a_error.shape[0]
  return a_error


def get_hist_boundary():
  """ Returns histogram boundary values.

  Returns:
    bounardy_values: a list of boundary values.
  """

  boundary_values = [-2.85, 2.85]
  assert (boundary_values[0] == -boundary_values[1])
  return boundary_values


def read_image(im_file):
  """ Reads an rgb image file.

  Args:
    im_file: full path of image file.

  Returns:
    results the rgb image in floating-point (height x width x channel) format.
  """

  in_img = cv2.imread(im_file, -1)
  assert len(in_img.shape) == 3, 'Grayscale images are not allowed'
  in_img = from_bgr2rgb(in_img)  # convert from BGR to RGB
  in_img = im2double(in_img)  # convert to double
  return in_img


def resize_image(im, target_size):
  """ Resizes a given image to a target size.

  Args:
    im: input ndarray image (height x width x channel).
    target_size: target size (list) in the format [target_height, target_width].

  Returns:
    results the resized image (target_height x target_width x channel).
  """

  h, w, c = im.shape
  if h != target_size[1] or w != target_size[0]:
    im = cv2.resize(im, (target_size[0], target_size[1]))
  if c == 1:
    im = np.expand_dims(im, axis=-1)
  return im


def get_uv_coord(hist_size, tensor=True, normalize=False, device='cuda'):
  """ Gets uv-coordinate extra channels to augment each histogram as
    mentioned in the paper.

  Args:
    hist_size: histogram dimension (scalar).
    tensor: boolean flag for input torch tensor; default is true.
    normalize: boolean flag to normalize each coordinate channel; default
      is false.
    device: output tensor allocation ('cuda' or 'cpu'); default is 'cuda'.

  Returns:
    u_coord: extra channel of the u coordinate values; if tensor arg is True,
      the returned tensor will be in (1 x height x width) format; otherwise,
      it will be in (height x width) format.
    v_coord: extra channel of the v coordinate values. The format is the same
      as for u_coord.
  """

  u_coord, v_coord = np.meshgrid(
    np.arange(-(hist_size - 1) / 2, ((hist_size - 1) / 2) + 1),
    np.arange((hist_size - 1) / 2, (-(hist_size - 1) / 2) - 1, -1))
  if normalize:
    u_coord = (u_coord + ((hist_size - 1) / 2)) / (hist_size - 1)
    v_coord = (v_coord + ((hist_size - 1) / 2)) / (hist_size - 1)
  if tensor:
    u_coord = torch.from_numpy(u_coord).to(device=device, dtype=torch.float32)
    u_coord = torch.unsqueeze(u_coord, dim=0)
    u_coord.requires_grad = False
    v_coord = torch.from_numpy(v_coord).to(device=device, dtype=torch.float32)
    v_coord = torch.unsqueeze(v_coord, dim=0)
    v_coord.requires_grad = False
  return u_coord, v_coord


def from_coord_to_uv(hist_size, u, v):
  """ Calculates the corresponding log-chroma values of given (u,v) coordinates.

  Args:
    hist_size: histogram dimension (scalar).
    u, v: input u,v coordinates.

  Returns:
    corresponding log-chroma values.

  """

  coord_range = get_hist_boundary()
  space_range = -coord_range[0] + coord_range[1]
  scale = space_range / hist_size
  U = u * scale
  V = v * scale
  return U, V


def to_tensor(im, dims=3):
  """ Converts a given ndarray image to torch tensor image.

  Args:
    im: ndarray image (height x width x channel x [sample]).
    dims: dimension number of the given image. If dims = 3, the image should
      be in (height x width x channel) format; while if dims = 4, the image
      should be in (height x width x channel x sample) format; default is 3.

  Returns:
    torch tensor in the format (channel x height x width)  or (sample x
      channel x height x width).
  """

  assert (dims == 3 or dims == 4)
  if dims == 3:
    im = im.transpose((2, 0, 1))
  elif dims == 4:
    im = im.transpose((3, 2, 0, 1))
  else:
    raise NotImplementedError
  return torch.from_numpy(im)


def complex_multiplication(a, b):
  """ Computes element-wise complex-tensor multiplication.

  Args:
    a and b: multiplication operands (tensors of complex numbers). Each
      tensor is in (batch x channel x height x width x complex_channel) format,
      where complex_channel contains the real and imaginary parts of each
      complex number in the tensor.

  Returns:
    results of a x b = (c+di)(j+hi) = (cj - dh) + (jd + ch)i.
  """

  assert (len(a.shape) >= 4 and len(a.shape) >= 4)
  assert (a.shape[-3:] == b.shape[-3:])
  assert (a.shape[-1] == 2)
  if len(a.shape) == 4:
    a = torch.unsqueeze(a, dim=1)
  if len(b.shape) == 4:
    b = torch.unsqueeze(b, dim=1)
  real_a = a[:, :, :, :, 0]
  imag_a = a[:, :, :, :, 1]
  real_b = b[:, :, :, :, 0]
  imag_b = b[:, :, :, :, 1]
  result = torch.stack([real_a * real_b - imag_a * imag_b,
                        real_a * imag_b + imag_a * real_b], dim=-1)
  return result


def from_tensor_to_image(tensor):
  """ Converts torch tensor image to numpy tensor image.

  Args:
    tensor: torch image tensor in one of the following formats:
      - 1 x channel x height x width
      - channel x height x width

  Returns:
    return a cpu numpy tensor image in one of the following formats:
      - 1 x height x width x channel
      - height x width x channel
  """

  image = tensor.cpu().numpy()
  if len(image.shape) == 4:
    image = image.transpose(0, 2, 3, 1)
  if len(image.shape) == 3:
    image = image.transpose(1, 2, 0)
  return image


def compute_histogram(chroma_input, hist_boundary, nbins, rgb_input=None):
  """ Computes log-chroma histogram of a given log-chroma values.

  Args:
    chroma_input: k x 2 array of log-chroma values; k is the total number of
      pixels and 2 is for the U and V values.
    hist_boundary: histogram boundaries obtained from the 'get_hist_boundary'
      function.
    nbins: number of histogram bins.
    rgb_input: k x 3 array of rgb colors; k is the totanl number of pixels and
      3 is for the rgb vectors. This is an optional argument, if it is
      omitted, the computed histogram will not consider the overall
      brightness value in Eq. 3 in the paper.

  Returns:
    N: nbins x nbins log-chroma histogram.
  """

  eps = np.sum(np.abs(hist_boundary)) / (nbins - 1)
  hist_boundary = np.sort(hist_boundary)
  A_u = np.arange(hist_boundary[0], hist_boundary[1] + eps / 2, eps)
  A_v = np.flip(A_u)
  if rgb_input is None:
    Iy = np.ones(chroma_input.shape[0])
  else:
    Iy = np.sqrt(np.sum(rgb_input ** 2, axis=1))
  # differences in log_U space
  diff_u = np.abs(np.tile(chroma_input[:, 0], (len(A_u), 1)).transpose() -
                  np.tile(A_u, (len(chroma_input[:, 0]), 1)))

  # differences in log_V space
  diff_v = np.abs(np.tile(chroma_input[:, 1], (len(A_v), 1)).transpose() -
                  np.tile(A_v, (len(chroma_input[:, 1]), 1)))

  # counts only U values that is higher than the threshold value
  diff_u[diff_u > eps] = 0
  diff_u[diff_u != 0] = 1

  # counts only V values that is higher than the threshold value
  diff_v[diff_v > eps] = 0
  diff_v[diff_v != 0] = 1

  Iy_diff_v = np.tile(Iy, (len(A_v), 1)) * diff_v.transpose()
  N = np.matmul(Iy_diff_v, diff_u)
  norm_factor = np.sum(N) + EPS
  N = np.sqrt(N / norm_factor)  # normalization
  return N


def get_hist_colors(img, from_rgb):
  """ Gets valid chroma and color values for histogram computation.

  Args:
    img: input image as an ndarray in the format (height x width x channel).
    from_rgb: a function to convert from rgb to chroma.

  Returns:
    valid_chroma: valid chroma values.
    valid_colors: valid rgb color values.
  """

  img_r = np.reshape(img, (-1, 3))
  img_chroma = from_rgb(img_r)
  valid_pixels = np.sum(img_r, axis=1) > EPS  # exclude any zero pixels
  valid_chroma = img_chroma[valid_pixels, :]
  valid_colors = img_r[valid_pixels, :]
  return valid_chroma, valid_colors


def vis_tensor(tensor, norm=True, dim=None, scale=1):
  """ Returns a processed tensor for visualization purposes.

  Args:
    tensor: image tensor; if it is in the format (batch x channel x height x
      width), use dim=None (default); otherwise, use dim to determine which
      axis is used for dimension extension to be in the (batch x channel x
      height x width) format.
    norm: boolean to apply min-max normalization; default is True.
    dim: a dimension of size one inserted at the specified position; this
      dimension extension is an optional; default is without extension.
    scale: gain scale for visualization purposes; default is 1.

  Returns:
    enhanced tensor for visualization purposes.
  """

  if dim is not None:
    tensor = torch.unsqueeze(tensor, dim=dim)
  if norm:
    tensor = (tensor - torch.min(tensor)) / (
        torch.max(tensor) - torch.min(tensor))
  tensor = tensor * scale
  return tensor


def rgb_to_uv(rgb, tensor=False):
  """ Converts RGB to log-chroma space.

  Args:
    rgb: input color(s) in rgb space.
    tensor: boolean flag for input torch tensor; default is false.

  Returns:
    color(s) in chroma log-chroma space.
  """

  if tensor:
    log_rgb = torch.log(rgb + EPS)
    u = log_rgb[:, 1] - log_rgb[:, 0]
    v = log_rgb[:, 1] - log_rgb[:, 2]
    return torch.stack([u, v], dim=-1)
  else:
    log_rgb = np.log(rgb + EPS)
    u = log_rgb[:, 1] - log_rgb[:, 0]
    v = log_rgb[:, 1] - log_rgb[:, 2]
    return np.stack([u, v], axis=-1)


def uv_to_rgb(uv, tensor=False):
  """ Converts log-chroma space to RGB.

  Args:
    uv: input color(s) in chroma log-chroma space.
    tensor: boolean flag for input torch tensor; default is false.

  Returns:
    color(s) in rgb space.
  """

  if tensor:
    rb = torch.exp(-uv)
    rgb = torch.stack([rb[:, 0], torch.ones(
      rb.shape[0], dtype=uv.dtype, device=uv.device), rb[:, 1]],
                      dim=-1)
    rgb = rgb / torch.unsqueeze(vect_norm(rgb, tensor), dim=-1)
    return rgb
  else:
    rb = np.exp(-uv)
    rgb = np.stack([rb[:, 0], np.ones(rb.shape[0]), rb[:, 1]], axis=-1)
    return rgb / np.transpose(np.tile(vect_norm(rgb), (3, 1)))


def vect_norm(vect, tensor=False, axis=1):
  """ Computes vector norm.

  Args:
    vect: input vector(s) (float).
    tensor: boolean flag for input torch tensor; default is false.
    axis: sum axis; default is 1.

  Returns:
    vector norm.
  """

  if tensor:
    return torch.sqrt(torch.sum(vect ** 2, dim=axis))
  else:
    return np.sqrt(np.sum(vect ** 2, axis=axis))


def from_bgr2rgb(im):
  """ Converts bgr image to rgb image.

  Args:
    im: bgr image (ndarray).

  Returns:
    input image in rgb format.
  """
  return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def from_rgb2bgr(im):
  """ Converts rgb image to bgr image.

  Args:
    im: rgb image (ndarray).

  Returns:
    input image in bgr format.
  """
  return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)


def im2double(im):
  """ Converts an uint image to floating-point format [0-1].

  Args:
    im: image (uint ndarray); supported input formats are: uint8 or uint16.

  Returns:
    input image in floating-point format [0-1].
  """

  if im[0].dtype == 'uint8':
    max_value = 255
  elif im[0].dtype == 'uint16':
    max_value = 65535
  return im.astype('float') / max_value


def compute_edges(im):
  """ Computes gradient intensities of a given image; this is used to
    generate the edge histogram N_1, as described in the paper.

  Args:
    im: image as an ndarray (float).

  Returns:
    gradient intensities as ndarray with the same dimensions of im (float).
  """

  assert (len(im.shape) == 3)  # should be a 3D tensor
  assert (im.shape[-1] == 3)  # should be 3-channel color image
  edge_img = np.zeros(im.shape)
  img_pad = cv2.copyMakeBorder(im, 1, 1, 1, 1, cv2.BORDER_REFLECT)
  offsets = [-1, 0, 1]
  for filter_index, (dx, dy) in enumerate(
      itertools.product(offsets, repeat=2)):
    if dx == 0 and dy == 0:
      continue
    edge_img[:, :, :] = edge_img[:, :, :] + (
      np.abs(im[:, :, :] - img_pad[1 + dx:im.shape[0] + 1 + dx,
                           1 + dy:im.shape[1] + 1 + dy, :]))
  edge_img = edge_img / 8
  return edge_img


def add_camera_name(dataset_dir):
  """ Adds camera model name to each image/metadata filename.

  Args:
    dataset_dir: dataset directory that should include sub-directories of
    camera models. We assume the following structure:
      - dataset_dir:
          - camera_1:
              - image1.png
              - image1_metadata.json
              - image2.png
              - image2_metadata.json
              - ....
          - camera_2:
             - image1.png
             - image1_metadata.json
             - ...
          - ...
    The new dataset will be located in dataset_dir_files
  """
  dataset_dir_new = os.path.dirname(dataset_dir) + '_files'
  if not os.path.exists(dataset_dir_new):
    os.mkdir(dataset_dir_new)
  cameras = glob(f'{dataset_dir}/*')
  img_extensions = ['.png', '.PNG']
  for camera_model in cameras:
    postfix = os.path.split(camera_model)[-1]
    print(f'processing {postfix}...')
    filenames = glob(f'{camera_model}/*')
    for filename in filenames:
      base, ext = os.path.splitext(filename)
      base = os.path.split(base)[-1]
      if ext.lower() in img_extensions:
        print(os.path.join(dataset_dir_new,
                           f'{base}_sensorname_{postfix}.png'))
        copyfile(filename, os.path.join(dataset_dir_new,
                                        f'{base}_sensorname_{postfix}.png'))
        metadata_file = os.path.join(camera_model, base + '_metadata.json')
        if not os.path.exists(metadata_file):
          metadata_file = os.path.join(camera_model, base + '_metadata.JSON')
          if not os.path.exists(metadata_file):
            raise FileNotFoundError

        copyfile(metadata_file, os.path.join(
          dataset_dir_new, f'{base}_sensorname_{postfix}_metadata.json'))
