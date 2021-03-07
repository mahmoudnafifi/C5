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

from os.path import join
from os import listdir
from os import path
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import json
from src import ops
import random


class Data(Dataset):
  def __init__(self, imgfiles, data_num=1, mode='training', input_size=64,
               load_hist=True):
    """ Data constructor

    Args:
      imgfiles: a list of full filenames to be used by the dataloader. If the
         mode is set to 'training', each filename in the list should have a
         metadata json file with a postfix '_metadata'. For example, if the
         filename is 'data/image1_sensorname_canon.png', the metadata file
         should be 'data/image1_sensorname_canon_metadata.json'. Each
         metadata file should contain a key named 'illuminant_color_raw' or
         'gt_ill' that contains the true rgb illuminant color.
      data_num: number of input histograms to C5 network (m in the paper);
         default is 1.
      mode: 'training' or 'testing'. In the training mode, ground-truth
         illuminant information should be loaded; while for the testing mode it
         is an optional. Default is 'training'.
      input_size: histogram dimensions (number of bins).
      load_hist: boolean flat to load histogram file if it exists; default is
        true.

    Returns:
      Dataset loader object with the selected settings.
    """

    assert (data_num >= 1)
    assert (mode == 'training' or mode == 'testing')
    assert (input_size % 2 == 0)
    self.imgfiles = imgfiles
    self.input_size = input_size
    self.additional_data_num = data_num - 1
    self.image_size = [384, 256]  # width, height
    self.load_hist = load_hist  # load histogram if exists
    self.mode = mode
    self.from_rgb = ops.rgb_to_uv  # rgb to chroma conversion function
    self.to_rgb = ops.uv_to_rgb  # chroma to rgb conversion function
    self.hist_boundary = ops.get_hist_boundary()

    logging.info(f'Creating dataset with {len(self.imgfiles)} examples')

  def __len__(self):
    """ Gets length of image files in the dataloader. """

    return len(self.imgfiles)

  def __getitem__(self, i):
    """ Gets next data in the dataloader.

    Args:
      i: index of file in the dataloader.

    Returns:
      A dictionary of the following keys:
      - image_rgb:
      - file_name: filename (without the full path).
      - histogram: input histogram.
      - model_input_histograms: input histogram and the additional histograms
          to be fed to C5 network.
      - gt_ill: ground-truth illuminant color. If the dataloader's 'mode'
         variable was  set to 'testing' and the ground-truth illuminant
         information does not exist, it will contain an empty tensor.
    """

    img_file = self.imgfiles[i]

    in_img = ops.read_image(img_file)
    in_img = ops.resize_image(in_img, self.image_size)

    rgb_img = ops.to_tensor(in_img)  # for visualization

    # gets the ground-truth illuminant color
    with open(path.splitext(img_file)[
                0] + '_metadata.json', 'r') as metadata_file:
      metadata = json.load(metadata_file)

    if self.mode == 'training':
      assert ['illuminant_color_raw' in metadata.keys() or 'gt_ill' in
              metadata.keys()]
    if 'illuminant_color_raw' in metadata.keys():
      gt_ill = np.array(metadata['illuminant_color_raw'])
      gt_ill = torch.from_numpy(gt_ill)
    elif 'gt_ill' in metadata.keys():
      gt_ill = np.array(metadata['gt_ill'])
      gt_ill = torch.from_numpy(gt_ill)
    else:
      gt_ill = torch.tensor([])

    # computes histogram feature of rgb and edge images
    if self.input_size is 64:
      post_fix = ''
    else:
      post_fix = f'_{self.input_size}'

    if path.exists(path.splitext(img_file)[0] +
                   f'_histogram{post_fix}.npy') and self.load_hist:
      histogram = np.load(path.splitext(img_file)[0] +
                          f'_histogram{post_fix}.npy', allow_pickle=False)
    else:
      histogram = np.zeros((self.input_size, self.input_size, 2))
      valid_chroma_rgb, valid_colors_rgb = ops.get_hist_colors(
        in_img, self.from_rgb)
      histogram[:, :, 0] = ops.compute_histogram(
        valid_chroma_rgb, self.hist_boundary, self.input_size,
        rgb_input=valid_colors_rgb)

      edge_img = ops.compute_edges(in_img)
      valid_chroma_edges, valid_colors_edges = ops.get_hist_colors(
        edge_img, self.from_rgb)

      histogram[:, :, 1] = ops.compute_histogram(
        valid_chroma_edges, self.hist_boundary, self.input_size,
        rgb_input=valid_colors_edges)

      np.save(path.splitext(img_file)[0] + f'_histogram{post_fix}.npy',
              histogram)

    in_histogram = ops.to_tensor(histogram)

    # gets additional input data
    if self.additional_data_num > 0:
      additiona_files = Data.get_rand_examples_from_sensor(
        current_file=img_file, files=self.imgfiles,
        target_number=self.additional_data_num)
    else:
      additiona_files = None

    additional_histogram = histogram

    u_coord, v_coord = ops.get_uv_coord(self.input_size,
                                        tensor=False, normalize=True)
    u_coord = np.expand_dims(u_coord, axis=-1)
    v_coord = np.expand_dims(v_coord, axis=-1)

    additional_histogram = np.concatenate([additional_histogram, u_coord],
                                          axis=-1)
    additional_histogram = np.concatenate([additional_histogram, v_coord],
                                          axis=-1)
    additional_histogram = np.expand_dims(additional_histogram, axis=-1)

    # if multiple input is used, load them
    if additiona_files is not None:
      for file, i in zip(additiona_files, range(len(additiona_files))):
        # computes histogram feature of rgb and edge images
        if path.exists(path.splitext(file)[0] +
                       f'_histogram{post_fix}.npy') and self.load_hist:
          histogram = np.load(path.splitext(file)[0] +
                              f'_histogram{post_fix}.npy', allow_pickle=False)

        else:
          img = ops.read_image(file)
          h, w, _ = img.shape
          if h != self.image_size[1] or w != self.image_size[0]:
            img = ops.resize_image(img, self.image_size)
          histogram = np.zeros((self.input_size, self.input_size, 2))
          valid_chroma_rgb, valid_colors_rgb = ops.get_hist_colors(
            img, self.from_rgb)
          histogram[:, :, 0] = ops.compute_histogram(
            valid_chroma_rgb, self.hist_boundary, self.input_size,
            rgb_input=valid_colors_rgb)
          edge_img = ops.compute_edges(img)
          valid_chroma_edges, valid_colors_edges = ops.get_hist_colors(
            edge_img, self.from_rgb)

          histogram[:, :, 1] = ops.compute_histogram(
            valid_chroma_edges, self.hist_boundary, self.input_size,
            rgb_input=valid_colors_edges)

          np.save(path.splitext(file)[0] + f'_histogram{post_fix}.npy',
                  histogram)

        histogram = np.concatenate([histogram, u_coord], axis=-1)
        histogram = np.concatenate([histogram, v_coord], axis=-1)
        histogram = np.expand_dims(histogram, axis=-1)

        additional_histogram = np.concatenate([additional_histogram, histogram],
                                              axis=-1)

    additional_histogram = ops.to_tensor(additional_histogram, dims=4)

    return {'image_rgb': rgb_img,
            'file_name': path.basename(img_file),
            'histogram': in_histogram,
            'model_input_histograms': additional_histogram,
            'gt_ill': gt_ill}

  @staticmethod
  def load_files(img_dir):
    """ Loads filenames in a given image directory.

    Args:
      img_dir: image directory. Note that if the dataloader's 'mode' variable
        was set to 'training', each filename in the list should have a
        metadata json file with a postfix '_metadata'. For example, if the
        filename is 'data/image1_sensorname_canon.png', the metadata file
        should be 'data/image1_sensorname_canon_metadata.json'. Each
        metadata file should contain a key named 'illuminant_color_raw' or
        'gt_ill' that contains the true rgb illuminant color.

    Returns:
      imgfiles: a list of full filenames.
    """

    logging.info(f'Loading images information from {img_dir}...')
    imgfiles = [join(img_dir, file) for file in listdir(img_dir)
                if file.endswith('.png') or file.endswith('.PNG')]
    return imgfiles

  @staticmethod
  def get_rand_examples_from_sensor(current_file, files, target_number):
    """ Randomly selects additional filenames of images taken by the same
       sensor.

    Args:
      current_file: filename of the current image; this filename should be in
         the following format: 'a_sensorname_b.png', where a is image id (can
         contain any string) and b is camera model name. The function will
         randomly select additional images that have the same camera model
         name (i.e., b).
      files: filenames of images in the dataloader.
      target_number: number of the additional images.

    Returns:
      sensor_files: additional image filenames taken by the same camera model
         used to capture the image in current_file.
    """
    assert ('sensorname' in current_file)
    sensor_name = path.splitext(current_file)[0].split('sensorname_')[-1]
    sensor_files = [file for file in files if sensor_name in file]
    sensor_files.remove(current_file)
    random.shuffle(sensor_files)
    if len(sensor_files) < target_number:
      raise Exception('Cannot find enough training data from sensor:'
                      f'{sensor_name}')
    return sensor_files[:target_number]
