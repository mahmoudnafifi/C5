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

import argparse
import logging
import os
import sys
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from src import c5
import random
import math
from src import ops

try:
  from torch.utils.tensorboard import SummaryWriter

  use_tb = True
except ImportError:
  use_tb = False

from src import dataset
from torch.utils.data import DataLoader

torch.manual_seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(0)


def train_net(net, device, dir_img, val_dir_img=None, val_ratio=0.1,
              epochs=1000, batch_size=64, lr=0.001, l2reg=0.00001,
              grad_clip_value=0, increasing_batch_size=False, load_hist=True,
              data_num=7, chkpoint_period=10, smoothness_factor_F=0.15,
              smoothness_factor_B=0.02, smoothness_factor_G=0.02,
              optimizer_algo='Adam', learn_g=True, cross_validation=False,
              aug_dir=None, input_size=64, validation_frequency=10,
              model_name='c5_model', save_cp=True):
  """ Trains C5 network and saves the trained model in harddisk.

  Args:
    net: network object (c5.network).
    device: use 'cpu' or 'cuda' (string).
    dir_img: full path of training set directory (string).
    val_dir_img: full path of validation set directory; if it is None (
      default), some images in training set will be used for validation.
    val_ratio: if val_dir_img is None, this variable set the ratio of the
      total number of training images to be used for validation.
    batch_size: mini-batch size; default value is 64.
    lr: learning rate; default value is 0.001.
    l2reg: L2 regularization factor; default value is 0.00001.
    grad_clip_value: threshold value for clipping gradients. If it is set to
      0 (default) clipping gradient is not applied.
    increasing_batch_size: boolean flag to use increasing batch size during
      training; default value is False.
    load_hist: boolean flag to load histograms from beginning (if exists in the
      image directory); default value is True.
    data_num: number of input histograms to C5 network (m in the paper);
      default value is 7.
    chkpoint_period: save a checkpoint every chkpoint_period epochs; default
      value is 10.
    smoothness_factor_F: smoothness regularization factor of convolutional
      filter F; default value is 0.15.
    smoothness_factor_B: smoothness regularization factor of bias B; default
      value is 0.02.
    smoothness_factor_G: smoothness regularization factor of gain multiplier
      map G (applied if learn_g is True); default value is 0.02.
    optimizer_algo: Optimization algorithm: 'SGD' or 'Adam'; default is 'Adam'.
    learn_g: boolean flag to learn the gain multiplier map G; default value
      is True.
    cross_validation: boolean flag to use three-fold cross-validation on the
      training data; default value is False.
    aug_dir: full path of additional images (for augmentation). If it is None,
      only the images in the 'dir_img' will be used for training; default
      value is None.
    input_size: Number of bins in histogram; default is 64.
    validation_frequency: Number of epochs to validate the model; default
      value is 10.
    model_name: Name of the final trained model; default is 'c5_model'.
    save_cp: boolean flag to save checkpoints during training; default is True.
  """

  dir_checkpoint = 'checkpoints_model/'  # check points directory

  # check if there is additional images to use
  if aug_dir is not None:
    aug_files = []
    for aug_set in aug_dir:
      aug_files = aug_files + dataset.Data.load_files(aug_set)
    random.shuffle(aug_files)
    augmentation = True
  else:
    augmentation = False

  # if cv is applied, load 3-fold validation indices (if exist) or create new
  #  indices.
  if cross_validation:
    if not os.path.exists('folds'):
      os.mkdir('folds')
      logging.info('Created cross validation folds directory')

    dataset_name = os.path.basename(dir_img)
    if (os.path.exists(f'folds/{dataset_name}_fold_1.npy') and
        os.path.exists(f'folds/{dataset_name}_fold_2.npy') and
        os.path.exists(f'folds/{dataset_name}_fold_3.npy')):
      logging.info('Loading CV folds...')
      testing_fold_1_filenames = np.load(f'folds/{dataset_name}_fold_1.npy')
      testing_fold_2_filenames = np.load(f'folds/{dataset_name}_fold_2.npy')
      testing_fold_3_filenames = np.load(f'folds/{dataset_name}_fold_3.npy')

      testing_fold_1 = [os.path.join(dir_img, os.path.basename(file))
                        for file in testing_fold_1_filenames]
      testing_fold_2 = [os.path.join(dir_img, os.path.basename(file))
                        for file in testing_fold_2_filenames]
      testing_fold_3 = [os.path.join(dir_img, os.path.basename(file))
                        for file in testing_fold_3_filenames]

    # if cv files are not exist, create new cv indices; save them in 'folds'
    # directory.
    else:
      input_files = dataset.Data.load_files(dir_img)
      random.shuffle(input_files)
      testing_fold_1 = input_files[:math.ceil(len(input_files) * 1 / 3)]
      testing_fold_2 = input_files[math.ceil(len(input_files) * 1 / 3):
                                   math.ceil(len(input_files) * 2 / 3)]
      testing_fold_3 = input_files[math.ceil(len(input_files) * 2 / 3):]
      np.save(f'folds/{dataset_name}_fold_1.npy', testing_fold_1)
      np.save(f'folds/{dataset_name}_fold_2.npy', testing_fold_2)
      np.save(f'folds/{dataset_name}_fold_3.npy', testing_fold_3)

    data = [testing_fold_1, testing_fold_2, testing_fold_3]
    folds = 3

  else:  # if cv is not applied, use regular training/validation settings.
    input_files = dataset.Data.load_files(dir_img)
    random.shuffle(input_files)
    # if validation directory is not given, use a part of training data for
    # validation.
    if val_dir_img is not None:
      tr_files = input_files
      val_files = dataset.Data.load_files(val_dir_img)
    else:
      assert (0 < val_ratio < 1)

      val_files = input_files[:math.ceil(len(input_files) * val_ratio)]
      tr_files = input_files[math.ceil(len(input_files) * val_ratio):]
    if aug_dir is not None:
      tr_files = tr_files + aug_files
    folds = 1  # set folds to 1 as there is no cv applied

  # smoothness Sobel filters
  u_variation = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
  u_variation = torch.tensor(
    u_variation, dtype=torch.float32).unsqueeze(0).expand(
    1, 1, 3, 3).to(device=device)
  u_variation.requires_grad = False

  v_variation = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
  v_variation = torch.tensor(
    v_variation, dtype=torch.float32).unsqueeze(0).expand(
    1, 1, 3, 3).to(device=device)
  v_variation.requires_grad = False

  for fold in range(folds):
    if folds > 1:  # cv is used
      train_folds = list({0, 1, 2} - {fold})
      tr_files = []
      for train_fold_i in train_folds:
        fold_files = [os.path.join(dir_img, os.path.basename(file)) for
                      file in data[train_fold_i]]
        tr_files = tr_files + fold_files

      if aug_dir is not None:
        tr_files = tr_files + aug_files
      val_files = data[fold]

    val_batch_sz = min(len(val_files), batch_size)

    train = dataset.Data(tr_files, input_size=input_size, data_num=data_num,
                         mode='training', load_hist=load_hist)
    val = dataset.Data(val_files, input_size=input_size, data_num=data_num,
                       mode='testing', load_hist=load_hist)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=val_batch_sz, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=True)

    if use_tb:  # if TensorBoard is used
      if folds > 1:
        writer = SummaryWriter(log_dir='runs/' + model_name + f'_fold_{fold}',
                               comment=f'LR_{lr}_BS_{batch_size}')
      else:
        writer = SummaryWriter(log_dir='runs/' + model_name,
                               comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    if folds > 1:
      logging.info(f'Fold number {fold}.')

    logging.info(f'''Starting training:
          Model Name:            {model_name}
          Epochs:                {epochs}
          Batch size:            {batch_size}
          Input size:            {input_size} x {input_size}
          Number of input:       {data_num}
          Learning rate:         {lr}
          L2 reg. weight:        {l2reg}
          Training data:         {len(train)}
          Augmentation:          {augmentation}
          Increasing batch size: {increasing_batch_size}
          Smoothness factor F:   {smoothness_factor_F}
          Smoothness factor B:   {smoothness_factor_B}
          Smoothness factor G:   {smoothness_factor_G}
          Learn G multiplier:    {learn_g}
          Grad. clipping:        {grad_clip_value}
          Optimizer:             {optimizer_algo}
          Validation size:       {len(val)}
          Validation Frq.:       {validation_frequency}
          Checkpoints:           {save_cp}
          Device:                {device.type}
          Cross-validation:      {cross_validation}
          TensorBoard:           {use_tb}
      ''')

    if optimizer_algo == 'Adam':
      optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999),
                             weight_decay=l2reg)
    elif optimizer_algo == 'SGD':
      optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=l2reg)
    else:
      raise NotImplementedError

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    curr_batch_size = batch_size

    if increasing_batch_size:
      max_batch_size = 128  # maximum number of mini-batch
      milestones = [20, 50]  # milestones (epochs) to duplicate curr_batch_size

    for epoch in range(epochs):
      net.train()

      epoch_angular_loss = 0
      epoch_smoothness_loss = 0

      if increasing_batch_size and (epoch + 1) in milestones:
        if curr_batch_size < max_batch_size:
          curr_batch_size = min(curr_batch_size * 2, max_batch_size)

          # training data loader
          train = dataset.Data(tr_files, input_size=input_size,
                               data_num=data_num,
                               mode='training', load_hist=True)
          train_loader = DataLoader(train, batch_size=curr_batch_size,
                                    shuffle=True, num_workers=4,
                                    pin_memory=True)

      with tqdm(total=len(train), desc=f'Epoch {epoch + 1}/{epochs}',
                unit='img') as pbar:
        first_iteration = True
        for batch in train_loader:
          if use_tb:  # if TensorBoard is used
            # input images for visualization
            img_rgb = batch['image_rgb']
            img_rgb = img_rgb.to(device=device, dtype=torch.float32)

          # input histogram batch
          histogram = batch['histogram']
          histogram = histogram.to(device=device, dtype=torch.float32)

          # model histogram(s)
          model_histogram = batch['model_input_histograms']
          model_histogram = model_histogram.to(device=device,
                                               dtype=torch.float32)

          # gt illuminant color batch
          gt = batch['gt_ill']
          gt = gt.to(device=device, dtype=torch.float32)

          predicted_ill, P, F, B, G = net(histogram, model_in_N=model_histogram)

          if len(B.shape) == 2:
            B = torch.unsqueeze(B, dim=0)

          if len(F.shape) == 3:
            F = torch.unsqueeze(F, dim=0)

          if G is not None:
            if len(G.shape) == 2:
              G = torch.unsqueeze(G, dim=0)

          loss = ops.angular_loss(predicted_ill, gt)
          py_loss = loss.item()

          # convert shrink angular error back to true angular error for printing
          try:
            py_loss = np.rad2deg(np.math.acos(np.math.cos(np.deg2rad(
              py_loss)) / 0.9999999))
          except:
            pass

          # decouple F into chroma and edge filters for visualization
          F_chroma = F[:, 0, :, :]
          F_edges = F[:, 1, :, :]

          # smoothing regularization for B
          s_loss_B = smoothness_factor_B * (torch.mean(
            torch.nn.functional.conv2d(
              torch.unsqueeze(B, dim=1), u_variation, stride=1) ** 2) +
                                            torch.mean(
                                              torch.nn.functional.conv2d(
                                                torch.unsqueeze(B, dim=1),
                                                v_variation,
                                                stride=1) ** 2))

          # smoothing regularization for G (if applied)
          if G is not None:
            s_loss_G = smoothness_factor_G * (torch.mean(
              torch.nn.functional.conv2d(torch.unsqueeze(
                G, dim=1), u_variation, stride=1) ** 2) + torch.mean(
              torch.nn.functional.conv2d(torch.unsqueeze(G, dim=1),
                                         v_variation, stride=1) ** 2))
          else:
            s_loss_G = 0

          # smoothing regularization for F
          s_loss_F_chroma = (torch.mean(torch.nn.functional.conv2d(
            torch.unsqueeze(F_chroma, dim=1), u_variation, stride=1) ** 2) +
                             torch.mean(torch.nn.functional.conv2d(
                               torch.unsqueeze(F_chroma, dim=1), v_variation,
                               stride=1) ** 2))

          s_loss_F_edges = (torch.mean(torch.nn.functional.conv2d(
            torch.unsqueeze(F_edges, dim=1), u_variation, stride=1) ** 2) +
                            torch.mean(torch.nn.functional.conv2d(
                              torch.unsqueeze(F_edges, dim=1), v_variation,
                              stride=1) ** 2))

          s_loss_F = smoothness_factor_F * (s_loss_F_chroma +
                                            s_loss_F_edges)

          # final smoothing regularization
          smoothness_loss = s_loss_F + s_loss_G + s_loss_B

          loss = loss + smoothness_loss

          epoch_smoothness_loss += smoothness_loss.item()

          epoch_angular_loss += py_loss

          optimizer.zero_grad()
          loss.backward()

          if grad_clip_value > 0:
            torch.nn.utils.clip_grad_value_(net.parameters(), grad_clip_value)

          optimizer.step()

          if use_tb:
            writer.add_scalar('Loss/train', py_loss, global_step)
            if first_iteration and epoch % 2 == 0:
              first_iteration = False
              bt_sz = predicted_ill.shape[0]
              gt_ill_img = torch.ones(bt_sz, 3, 200, 200).to(device=gt.device)
              predicted_ill_img = torch.ones(
                bt_sz, 3, 200, 200).to(device=gt.device)
              gt_ill_img = gt_ill_img * gt.view(bt_sz, 3, 1, 1)
              predicted_ill_img = (
                  predicted_ill_img * (predicted_ill / torch.unsqueeze(
                torch.norm(predicted_ill, dim=1), dim=1)).view(bt_sz, 3, 1, 1))
              writer.add_images('predicted ills', predicted_ill_img,
                                global_step)
              writer.add_images('gt ills', gt_ill_img, global_step)
              writer.add_images('images', torch.pow(img_rgb, 1.0/2.19921875),
                                global_step)
              writer.add_images('P', ops.vis_tensor(
                P, dim=1, scale=20 / torch.mean(P)), global_step)
              writer.add_images('histogram-image', ops.vis_tensor(
                histogram[:, 0, :, :], norm=True, dim=1), global_step)
              writer.add_images('histogram-edge', ops.vis_tensor(
                histogram[:, 1, :, :], norm=True, dim=1), global_step)
              writer.add_images('F-chroma (time domain)', ops.vis_tensor(
                F_chroma, dim=1, norm=True), global_step)
              writer.add_images('F-edges (time domain)', ops.vis_tensor(
                F_edges, dim=1, norm=True), global_step)
              writer.add_images('B', ops.vis_tensor(B, dim=1, norm=True),
                                global_step)
              if G is not None:
                writer.add_images('G', ops.vis_tensor(G, dim=1, norm=True),
                                  global_step)

          pbar.update(np.ceil(histogram.shape[0]))

          pbar.set_postfix(**{'angular loss (batch)': py_loss},
                           **{'smoothness loss (batch)'
                              : smoothness_loss.item()})

          global_step += 1

      epoch_smoothness_loss = epoch_smoothness_loss / (len(train) /
                                                       curr_batch_size)
      epoch_angular_loss = epoch_angular_loss / (len(train) / curr_batch_size)

      logging.info(f'Epoch loss: angular = {epoch_angular_loss}, '
                   f'smoothness = {epoch_smoothness_loss}')

      scheduler.step()

      # if load_hist is false and this is the first epoch, recreate the
      # dataloader with load_hist = True to save training time.
      if epoch == 0 and load_hist is False:
        train = dataset.Data(tr_files, input_size=input_size,
                             data_num=data_num,
                             mode='training', load_hist=True)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)

      # model validation
      if (epoch + 1) % validation_frequency == 0:

        val_score = vald_net(net=net, loader=val_loader, device=device)

        # if load_hist is false and this is the first epoch, recreate the
        # dataloader with load_hist = True to save training time.
        if epoch == 0 and load_hist is False:
          val = dataset.Data(val_files, input_size=input_size,
                             data_num=data_num,
                             mode='testing', load_hist=True)

          val_loader = DataLoader(val, batch_size=val_batch_sz, shuffle=False,
                                  num_workers=4, pin_memory=True,
                                  drop_last=True)

        logging.info('Validation loss: {}'.format(val_score))
        if use_tb:
          writer.add_scalar('learning_rate',
                            optimizer.param_groups[0]['lr'], global_step)
          writer.add_scalar('Loss/test', val_score, global_step)

      # save a checkpoint
      if save_cp and (epoch + 1) % chkpoint_period == 0:
        if not os.path.exists(dir_checkpoint):
          os.mkdir(dir_checkpoint)
          logging.info('Created checkpoint directory')

        torch.save(net.state_dict(), dir_checkpoint +
                   f'{model_name}_{epoch + 1}.pth')
        logging.info(f'Checkpoint {epoch + 1} saved!')

    # save final trained model
    if not os.path.exists('models'):
      os.mkdir('models')
      logging.info('Created trained models directory')

    if folds > 1:  # if cv is applied
      torch.save(net.state_dict(), 'models/' + f'{model_name}_fold_'
                                               f'{fold + 1}.pth')
      logging.info('Saved trained model!')
    else:
      torch.save(net.state_dict(), 'models/' + f'{model_name}.pth')
      logging.info('Saved trained model!')
    if use_tb:
      writer.close()
    logging.info('End of training')

    # reset the network; this is if cv is applied to train for the next fold
    # from scratch.
    net = c5.network(input_size=input_size, learn_g=learn_g,
                     data_num=data_num, device=device)


def vald_net(net, loader, device='cuda'):
  """ Evaluates using the validation set.

  Args:
    net: network object
    loader: dataloader of validation data
    device: 'cpu' or 'cuda'; default is 'cuda'

  Returns:
    val_loss: validation angular error
  """

  net.eval()
  n_val = 0
  val_loss = 0

  with tqdm(total=len(loader), desc='Validation round', unit='batch',
            leave=False) as pbar:
    for batch in loader:

      histogram = batch['histogram']
      histogram = histogram.to(device=device,
                               dtype=torch.float32)

      model_histogram = batch['model_input_histograms']
      model_histogram = model_histogram.to(device=device,
                                           dtype=torch.float32)

      gt = batch['gt_ill']
      gt = gt.to(device=device, dtype=torch.float32)

      with torch.no_grad():

        predicted_ill, _, _, _, _ = net(histogram, model_in_N=model_histogram)

        loss = ops.angular_loss(predicted_ill, gt)

        try:
          py_loss = np.rad2deg(np.math.acos(np.math.cos(np.deg2rad(
            loss.item())) / 0.9999999))
        except:
          py_loss = loss.item()

        py_loss = py_loss * predicted_ill.shape[0]
        n_val = n_val + predicted_ill.shape[0]
        val_loss = val_loss + py_loss

      pbar.update(np.ceil(histogram.shape[0]))

  net.train()
  val_loss = val_loss / n_val
  return val_loss


def get_args():
  """ Gets command-line arguments.

  Returns:
    Return command-line arguments as a set of attributes.
  """

  parser = argparse.ArgumentParser(description='Train C5.')
  parser.add_argument('-e', '--epochs', metavar='E', type=int, default=60,
                      help='Number of epochs', dest='epochs')

  parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?',
                      default=16, help='Batch size', dest='batch_size')

  parser.add_argument('-dn', '--data-num', dest='data_num', type=int,
                      default=1, help='Number of input data to create a model')

  parser.add_argument('-opt', '--optimizer', dest='optimizer', type=str,
                      default='Adam', help='Adam or SGD')

  parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float,
                      nargs='?', default=5e-4, help='Learning rate', dest='lr')

  parser.add_argument('-l2r', '--l2reg', metavar='L2Reg', type=float,
                      nargs='?', default=5e-4, help='L2 regularization '
                                                    'factor', dest='l2r')

  parser.add_argument('-l', '--load', dest='load', type=bool, default=False,
                      help='Load model from a .pth file')

  parser.add_argument('-ml', '--model-location', dest='model_location',
                      default=None)

  parser.add_argument('-vr', '--validation-ratio', dest='val_ratio',
                      type=float, default=0.25, help='Validation set ratio.')

  parser.add_argument('-vf', '--validation-frequency', dest='val_frq',
                      type=int, default=1, help='Validation frequency.')

  parser.add_argument('-s', '--input-size', dest='input_size', type=int,
                      default=64, help='Size of input histogram')

  parser.add_argument('-lh', '--load-hist', dest='load_hist',
                      type=bool, default=True, help='Load histogram if exists')

  parser.add_argument('-ibs', '--increasing-batch-size',
                      dest='increasing_batch_size', type=bool, default=True,
                      help='Increasing batch size.')

  parser.add_argument('-gc', '--grad-clip-value', dest='grad_clip_value',
                      type=float, default=0, help='Gradient clipping value; '
                                                  'if = 0, no clipping applied')

  parser.add_argument('-slf', '--smoothness-factor-F',
                      dest='smoothness_factor_F', type=float, default=0.15,
                      help='Smoothness regularization factor of conv filter')

  parser.add_argument('-slb', '--smoothness-factor-B',
                      dest='smoothness_factor_B', type=float, default=0.02,
                      help='Smoothness regularization factor of bias')

  parser.add_argument('-slg', '--smoothness-factor-G',
                      dest='smoothness_factor_G', type=float, default=0.02,
                      help='Smoothness regularization factor of gain')

  parser.add_argument('-cv', '--cross-validation', dest='cross_validation',
                      type=bool, default=False,
                      help='Use three cross validation. If true, it will '
                           'ignore both validation-dir-in and '
                           'validation-ratio and do a 3-fold cross-validation '
                           'on the data provided in the --training-dir-in. '
                           'The final models will be saved with a postfix of '
                           'the testing fold. The testing fold filenames will '
                           'be saved as will as .npy files for further '
                           'evaluation')

  parser.add_argument('-lg', '--learn-G', type=bool, default=False,
                      help='Learn G multiplier', dest='learn_g')

  parser.add_argument('-ntrd', '--training-dir-in', dest='in_trdir',
                      default='/training_dir/',
                      help='Input training image directory')

  parser.add_argument('-nvld', '--validation-dir-in', dest='in_vldir',
                      default=None,
                      help='Input validation image directory; if is None, the '
                           'validation will be taken from the training data '
                           'based on the validation-ratio argument')

  parser.add_argument('-augd', '--augmentation-dir', dest='aug_dir',
                      default=None, nargs='+',
                      help='Directory include augmentation data.')

  parser.add_argument('-n', '--model-name', dest='model_name',
                      default='c5_model')

  parser.add_argument('-g', '--gpu', dest='gpu', default=0, type=int)

  return parser.parse_args()


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  logging.info('Training C5')
  args = get_args()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if device.type != 'cpu':
    torch.cuda.set_device(args.gpu)

  logging.info(f'Using device {device}')

  net = c5.network(input_size=args.input_size, learn_g=args.learn_g,
                   data_num=args.data_num, device=device)
  if args.load:
    net.load_state_dict(
      torch.load(args.model_location, map_location=device)
    )
    logging.info(f'Model loaded from {args.model_location}')

  net.to(device=device)

  try:
    train_net(net=net, device=device, dir_img=args.in_trdir,
              val_dir_img=args.in_vldir, epochs=args.epochs,
              batch_size=args.batch_size, lr=args.lr, data_num=args.data_num,
              smoothness_factor_F=args.smoothness_factor_F,
              smoothness_factor_B=args.smoothness_factor_B,
              smoothness_factor_G=args.smoothness_factor_G,
              l2reg=args.l2r, load_hist=args.load_hist, learn_g=args.learn_g,
              optimizer_algo=args.optimizer, aug_dir=args.aug_dir,
              increasing_batch_size=args.increasing_batch_size,
              grad_clip_value=args.grad_clip_value,
              chkpoint_period=args.val_frq,
              cross_validation=args.cross_validation,
              validation_frequency=args.val_frq, input_size=args.input_size,
              val_ratio=args.val_ratio, model_name=args.model_name)

  except KeyboardInterrupt:
    torch.save(net.state_dict(), 'c5_intrrupted_check_point.pth')
    logging.info('Saved interrupt checkpoint backup')
    try:
      sys.exit(0)
    except SystemExit:
      os._exit(0)
