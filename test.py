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
import numpy as np
import torch
from src import c5
from scipy.io import savemat
from src import dataset
from torch.utils.data import DataLoader
from src import ops
from torchvision.utils import save_image
from torchvision.utils import make_grid


def test_net(net, device, dir_img, batch_size=64, input_size=64, data_num=7,
             g=False, model_name='c5_model', load_hist=False,
             white_balance=False, multiple_test=False, files=None,
             cross_validation=False, save_output=True):
  """ Tests C5 network.

  Args:
    net: network object (c5.network).
    device: use 'cpu' or 'cuda' (string).
    dir_img: full path of testing set directory (string).
    batch_size: mini-batch size; default value is 64.
    input_size: Number of bins in histogram; default is 64.
    data_num: number of input histograms to C5 network (m in the paper);
      default value is 7.
    g: boolean flag to learn the gain multiplier map G; default value
      is True.
    model_name: Name of the trained model; default is 'c5_model'.
    load_hist: boolean flag to load histograms from beginning (if exists in the
      image directory); default value is True.
    white_balance: boolean to perform a diagonal correction using the estimated
      illuminant color and save output images in harddisk. The saved images
      will be located in white_balanced_images/model_name/.; default is False.
    multiple_test: boolean flag to perform ten tests as described in the
      paper; default is False.
    files: a list to override loading files located in dir_img; default is
      None.
    cross_validation: boolean flag to use three-fold cross-validation on
      files located in the 'dir_img' directory; default value is False.
    save_output: boolean flag to save the results in results/model_name/.;
      default is True.
  """

  if files is None:
    files = dataset.Data.load_files(dir_img)

  batch_size = min(batch_size, len(files))

  test = dataset.Data(files, input_size=input_size, mode='testing',
                      data_num=data_num,
                      load_hist=load_hist)
  test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                           num_workers=8, pin_memory=True)

  logging.info(f'''Starting testing:
        Model Name:            {model_name}
        Batch size:            {batch_size}
        Number of input:       {data_num}
        Learn G multiplier:    {g}
        Input size:            {input_size} x {input_size}
        Testing data:          {len(files)}
        White balance:         {white_balance}
        Multiple tests:        {multiple_test}
        Cross validation:      {cross_validation}
        Save output:           {save_output}
        Device:                {device.type}
    ''')

  if multiple_test:
    number_of_tests = 10
  else:
    number_of_tests = 1

  if white_balance:
    save_filter_dir_wb = os.path.join('white_balanced_images', model_name)
    if not os.path.exists(save_filter_dir_wb):
      if not os.path.exists('white_balanced_images'):
        os.mkdir('white_balanced_images')
      os.mkdir(save_filter_dir_wb)
      logging.info(f'Created visualization directory {save_filter_dir_wb}')

  with torch.no_grad():
    for test_i in range(number_of_tests):
      results = np.zeros((len(test), 3))  # to store estimated illuminant values
      gt = np.zeros((len(test), 3))  # to store ground-truth illuminant colors
      filenames = []  # to store filenames
      index = 0

      for batch in test_loader:
        model_histogram = batch['model_input_histograms']
        model_histogram = model_histogram.to(device=device,
                                             dtype=torch.float32)
        if white_balance:
          image = batch['image_rgb']
          image = image.to(device=device, dtype=torch.float32)
          file_names = batch['file_name']

        histogram = batch['histogram']
        histogram = histogram.to(device=device, dtype=torch.float32)

        gt_ill = batch['gt_ill']
        gt_ill = gt_ill.to(device=device, dtype=torch.float32)

        predicted_ill, _, _, _, _ = net(histogram, model_in_N=model_histogram)

        if white_balance and test_i == 0:
          bs = image.shape[0]
          for c in range(3):
            correction_ratio = predicted_ill[:, 1] / predicted_ill[:, c]
            correction_ratio = correction_ratio.view(bs, 1, 1)
            image[:, c, :, :] = image[:, c, :, :] * correction_ratio

          image = 1 * torch.pow(image, 1.0 / 2.19921875)
          for b in range(bs):
            save_image(make_grid(image[b, :, :, :], nrow=1), os.path.join(
              save_filter_dir_wb, file_names[b]))

        L = len(predicted_ill)
        results[index:index + L, :] = predicted_ill.cpu().numpy()
        gt[index:index + L, :] = gt_ill.cpu().numpy()
        for f in file_names:
            filenames.append(f)
        index = index + L

      if save_output:
        save_dir = os.path.join('results', model_name)
        if not os.path.exists(save_dir):
          if not os.path.exists('results'):
            os.mkdir('results')
          os.mkdir(save_dir)
          logging.info(f'Created results directory {save_dir}')
        if multiple_test:
          savemat(os.path.join(save_dir, f'gt_{test_i + 1}.mat'), {'gt': gt})
          savemat(os.path.join(save_dir, f'results_{test_i + 1}.mat'),
                  {'predicted': results})
          savemat(os.path.join(save_dir, f'filenames_{test_i + 1}.mat'),
                  {'filenames': filenames})
        else:
          savemat(os.path.join(save_dir, 'gt.mat'), {'gt': gt})
          savemat(os.path.join(save_dir, 'results.mat'), {'predicted': results})
          savemat(os.path.join(save_dir, 'filenames.mat'), {'filenames': filenames})

  logging.info('End of testing')


def get_args():
  parser = argparse.ArgumentParser(description='Test C5.')

  parser.add_argument('-b', '--batch-size', metavar='B', type=int,
                      nargs='?', default=64,
                      help='Batch size', dest='batchsize')

  parser.add_argument('-s', '--input-size', dest='input_size', type=int,
                      default=64, help='Size of input (hist and image)')

  parser.add_argument('-ntrd', '--testing-dir-in', dest='in_tedir',
                      default='/testing_set/',
                      help='Input testing image directory')

  parser.add_argument('-lh', '--load-hist', dest='load_hist',
                      type=bool, default=True,
                      help='Load histogram if exists')

  parser.add_argument('-dn', '--data-num', dest='data_num', type=int, default=7,
                      help='Number of input data for calibration')

  parser.add_argument('-lg', '--g-multiplier', type=bool, default=False,
                      help='Have a G multiplier', dest='g_multiplier')

  parser.add_argument('-mt', '--multiple_test', type=bool, default=False,
                      help='do 10 tests and save the results',
                      dest='multiple_test')

  parser.add_argument('-wb', '--white-balance', type=bool,
                      default=False, help='save white-balanced image',
                      dest='white_balance')

  parser.add_argument('-cv', '--cross-validation', dest='cross_validation',
                      type=bool, default=False,
                      help='Use three cross validation. If true, we assume '
                           'that there are three pre-trained models saved '
                           'with a postfix of the fold number. The testing '
                           'image filenames should be listed in .npy files '
                           'located in "folds" directory with the same name of '
                           'the dataset, which should be the same as the '
                           'folder name in --testing-dir-in')

  parser.add_argument('-n', '--model-name', dest='model_name',
                      default='c5_model')

  parser.add_argument('-g', '--gpu', dest='gpu', default=0, type=int)

  return parser.parse_args()


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  logging.info('Testing C5')
  args = get_args()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if device.type != 'cpu':
    torch.cuda.set_device(args.gpu)
  logging.info(f'Using device {device}')

  net = c5.network(input_size=args.input_size, learn_g=args.g_multiplier,
                   data_num=args.data_num, device=device)

  if args.cross_validation:
    dataset_name = os.path.basename(args.in_tedir)
    for fold in range(3):
      model_path = os.path.join('models', args.model_name +
                                f'_fold_{fold + 1}.pth')
      net.load_state_dict(torch.load(model_path, map_location=device))
      logging.info(f'Model loaded from {model_path}')
      net.to(device=device)
      net.eval()
      testing_files = np.load(f'folds/{dataset_name}_fold_{fold + 1}.npy')
      files = [os.path.join(args.in_tedir, os.path.basename(file)) for file in
               testing_files]
      test_net(net=net, device=device, dir_img=args.in_tedir,
               cross_validation=args.cross_validation,
               g=args.g_multiplier,
               multiple_test=args.multiple_test,
               white_balance=args.white_balance,
               files=files, data_num=args.data_num,
               batch_size=args.batchsize,
               model_name=f'{args.model_name}_fold_{fold + 1}',
               input_size=args.input_size,
               load_hist=args.load_hist)
  else:
    model_path = os.path.join('models', args.model_name + '.pth')
    net.load_state_dict(torch.load(model_path, map_location=device))
    logging.info(f'Model loaded from {model_path}')
    net.to(device=device)
    net.eval()
    test_net(net=net, device=device,
             data_num=args.data_num, dir_img=args.in_tedir,
             cross_validation=args.cross_validation,
             g=args.g_multiplier,
             multiple_test=args.multiple_test,
             white_balance=args.white_balance,
             batch_size=args.batchsize, model_name=args.model_name,
             input_size=args.input_size,
             load_hist=args.load_hist)
