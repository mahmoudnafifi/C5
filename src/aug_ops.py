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
import os
import json
import logging
import random
import cv2
import ops
import copy

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
rand = np.random.rand
cameras = ['canon eos 550d', 'canon eos 5d', 'canon eos-1ds',
           'canon eos-1ds mark iii', 'fujifilm x-m1', 'nikon d40',
           'nikon d5200', 'olympus e-pl6', 'panasonic dmc-gx1',
           'samsung nx2000', 'sony slt-a57']


def set_sampling_params(im_per_scene_per_camera=1, intensity_transfer=False,
                        target_aug_im_num=5000, excluded_camera_models=None,
                        excluded_datasets=None, save_as_16_bits=True,
                        remove_saturated_pixels=False, saturation_level=0.97,
                        output_img_size=None, cropping=True,
                        color_temp_balance=True, lambda_r=0.7, lambda_g=1.2,
                        k=15):
  """ Sets sampling parameters.

  Args:
    im_per_scene_per_camera: number of sampled images per scene per camera
      model; the default is 1.
    intensity_transfer: transfer the intensity of target image into the
      source image. This is useful for methods that are not relying on the
      log-chroma space as an input; the default is False.
    target_aug_im_num: target number of images for augmentation; the default
      is 5,000 images.
    excluded_camera_models: a list of excluded camera models from the source
      set (i.e., camera models listed here will not have any image content in
      the augmented set). The default value is None.
    excluded_datasets: Similar to excluded_camera_models, but here you can
      use dataset names to determine the excluded images. Options are: Cube+,
      Gehler-Shi, NUS, Cube+_challenge, and Cube+_challenge_2. The default
      value is None.
    save_as_16_bits: boolean flag to save augmented images in 16-bit format.
    remove_saturated_pixels: mask out saturated pixels of the augmented
      images; the default is False.
    saturation_level: if remove_saturated_pixels is True, the saturation_level
      determines the threshold of saturation; default is 0.97.
    output_img_size: size of output images; the default is [256, 384].
    cropping: boolean flag to apply a random cropping in the augmented
      images; default is True.
    color_temp_balance: boolean flag to apply color temperature balancing as
      described in the paper. The default value is True.
    lambda_r: scale factor for the random shift applied to the red chroma
      channel in sampling (see the paper for more info.); default is 0.7.
    lambda_g: scale factor for the random shift applied to the green chroma
      channel in sampling (see the paper for more info.); default is 1.2.
    k: number of nearest neighbors; default is 15.
  Returns:
    params: a dict of sampling parameters.
  """

  if excluded_camera_models is None:
    excluded_camera_models = []
  if excluded_datasets is None:
    excluded_datasets = []
  if output_img_size is None:
    output_img_size = [256, 384]

  params = {'images_per_scene_per_camera': im_per_scene_per_camera,
            'intensity_transfer': intensity_transfer,
            'total_number': target_aug_im_num,
            'excluded_camera_models': excluded_camera_models,
            'excluded_datasets': excluded_datasets,
            'save_as_16_bits': save_as_16_bits,
            'remove_saturated_pixels': remove_saturated_pixels,
            'saturation_level': saturation_level,
            'cropping': cropping,
            'output_image_size': output_img_size,
            'color_temp_balancing': color_temp_balance,
            'lambda_r': lambda_r,
            'lambda_g': lambda_g,
            'k': k}
  return params


def map_raw_images(xyz_img_dir, target_cameras, output_dir, params):
  """ Maps raw images to target camera models.

  Args:
    xyz_img_dir: directory of XYZ images.
    target_cameras: target camera model name is a list of one or more
      of the following models:
        'Canon EOS 550D', 'Canon EOS 5D', 'Canon EOS-1DS',
        'Canon EOS-1Ds Mark III', 'Fujifilm X-M1', 'Nikon D40', 'Nikon D5200',
        'Olympus E-PL6', 'Panasonic DMC-GX1', 'Samsung NX2000', 'Sony SLT-A57',
        or 'All'.
    output_dir: output directory to save the augmented images.
    params: sampling parameters set by the 'set_sampling_params' function.
  """

  assert_target_camera(target_cameras)
  if ('all' in target_cameras or 'All' in target_cameras or 'ALL'
      in target_cameras):
    target_cameras = cameras

  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  images_per_scene = params['images_per_scene_per_camera']
  intensity_transfer = params['intensity_transfer']
  excluded_camera_models = params['excluded_camera_models']
  excluded_datasets = params['excluded_datasets']
  total_number = params['total_number']
  save_as_16_bits = params['save_as_16_bits']
  remove_saturated_pixels = params['remove_saturated_pixels']
  saturation_level = params['saturation_level']
  output_image_size = params['output_image_size']
  cropping = params['cropping']
  temp_balancing = params['color_temp_balancing']
  t_cam_num = len(target_cameras)
  assert (total_number >= (images_per_scene * t_cam_num))

  with open('aug_data.json') as json_file:
    metadata = json.load(json_file)

    cam_models = []
    datasets = []
    c_temps = []
    apertures = []
    exposure_times = []
    ISOs = []

    for sample in metadata:
      cam_models.append(sample['model'].lower())
      datasets.append(sample['dataset'].lower())
      c_temps.append(sample['color_temp'])
      apertures.append(sample['aperture'])
      exposure_times.append(sample['exposure_time'])
      ISOs.append(sample['ISO'])

    c_temps_quantized = (np.array(c_temps, dtype=np.float32) / 250).astype(
      np.uint16) * 250

    # exclude source scenes
    scene_ids = range(len(metadata))
    scene_ids = [*scene_ids]
    excluded_inds = []

    for camera_i in excluded_camera_models:
      excluded_inds = excluded_inds + np.where(
        np.isin(cam_models, camera_i.lower()))[0].tolist()

    for dataset_i in excluded_datasets:
      excluded_inds = excluded_inds + np.where(
        np.isin(datasets, dataset_i.lower()))[0].tolist()
    excluded_inds = np.unique(np.array(excluded_inds)).tolist()

    scene_ids = list(set(scene_ids) - set(excluded_inds))

    assert_xyz_img_dir(xyz_img_dir, scene_ids)

    random.shuffle(scene_ids)
    c_temps_quantized = c_temps_quantized[scene_ids]

    scene_iterations = round(total_number / (
        images_per_scene * t_cam_num))

    if scene_iterations - (total_number / (images_per_scene *
                                           t_cam_num)) != 0:
      logging.warn('Based on the current settings, the total number of '
                   'generated images may not be exactly as requested.')

    logging.info('Generating augmentation set ...')

    if temp_balancing:
      # divides source image's indices into groups based on their color
      # temperatures for a fair sampling.
      groups = []
      for t in np.unique(c_temps_quantized):
        t_list = []
        indices = np.where(c_temps_quantized == t)[0]
        for index in indices:
          t_list.append(scene_ids[index])
        groups.append(t_list)

      for g in range(len(groups)):
        group_ids = groups[g]
        random.shuffle(group_ids)

      random.shuffle(groups)
      pointers = np.zeros((len(groups), 1))

    counter = 1

    target_cameras_data = []
    for camera_i in target_cameras:
      indices = np.where(np.isin(cam_models, camera_i.lower()))[0]
      t_camera_data = []
      for index in indices:
        t_camera_data.append(metadata[index])
      target_cameras_data.append(t_camera_data)

    for scene_i in range(scene_iterations):
      logging.info('Processing secene number (%d/%d) ...' % (scene_i + 1,
                                                             scene_iterations))
      if temp_balancing:
        j = (scene_i % len(groups))
        pointers[j] = pointers[j] + 1
        ids = groups[j]
        scene_id = ids[int(pointers[j] % len(ids))]
      else:
        scene_id = scene_ids[scene_i % len(scene_ids)]

      image = ops.read_image(os.path.join(xyz_img_dir, '%04d.png' %
                                          (scene_id + 1)))

      mean_intensity = metadata[scene_id]['mean_intensity']
      c_temp = c_temps[scene_id]
      ISO = ISOs[scene_id]
      baseline_exposure = metadata[scene_id]['baseline_exposure']
      baseline_noise = metadata[scene_id]['baseline_noise']
      aperture_norm = metadata[scene_id]['aperture_norm']
      rotated = metadata[scene_id]['rotated']
      aperture = apertures[scene_id]
      exposure_time = exposure_times[scene_id]

      for camera_i in range(len(target_cameras)):
        logging.info('    Sampling from camera number (%d/%d)...', camera_i,
                     len(target_cameras))

        for image_i in range(images_per_scene):
          filename = 'image_%07d_sensorname_%s.png' % (
            counter, target_cameras[camera_i].lower().replace(' ', '_'))
          status = sampling(copy.deepcopy(image), target_cameras_data[camera_i],
                            output_dir, filename, c_temp, baseline_exposure,
                            baseline_noise, ISO, aperture, aperture_norm,
                            exposure_time, intensity_transfer,
                            remove_saturated_pixels, save_as_16_bits,
                            output_image_size, saturation_level, cropping,
                            rotated, mean_intensity,
                            lambda_r=params['lambda_r'],
                            lambda_g=params['lambda_g'], k=params['k'])

          if not status:
            logging.warning('Cannot generate a sample')
            continue

          counter = counter + 1


def sampling(im, t_camera_data, output_dir, filename,
             c_temp, baseline_exposure, baseline_noise, ISO, aperture,
             aperture_norm, exposure_time, transfer_intensity,
             remove_saturated_pixels, save_as_16_bits,
             output_image_size, saturation_level, cropping, rotated,
             mean_intensity, lambda_r, lambda_g, k):
  """ Samples from target camera set's settings and maps input image (im) to
  the sampled setting.

  Args:
    im: source image.
    t_camera_data: metadata of the target camera model.
    output_dir: output directory to save mapped images and metadata.
    filename: filename of output image.
    c_temp: color temperature of source image.
    baseline_exposure: baseline exposure of source image.
    baseline_noise: baseline noise of source image.
    ISO: ISO value of source image.
    aperture: aperture value of source image.
    aperture_norm: normalized aperture value of source image.
    exposure_time: exposure time of source image.
    transfer_intensity: boolean flag to transfer intensity from the target
      sample to mapped image.
    remove_saturated_pixels: boolean flag to remove saturated pixels.
    save_as_16_bits: boolean flag to save output image in 16-bit format.
    output_image_size: mapped image dimensions.
    saturation_level: threshold value for masking out saturated pixels. This
      is only be used if 'remove_saturated_pixels' is True.
    cropping: boolean flag to apply random cropping before saving the output
      mapped image.
    rotated: boolean flag if the image is rotated.
    mean_intensity: mean intensity of source image (used if
      transfer_intensity is True).
    lambda_r: scale factor for the random shift applied to the red chroma
      channel in sampling (see the paper for more info.).
    lambda_g: scale factor for the random shift applied to the green chroma
      channel in sampling (see the paper for more info.).
    k: number of nearest neighbors.
  """

  output_metadata = {'filename': filename,
                     'rotated': rotated,
                     'ISO': None,
                     'aperture': None,
                     'aperture_norm': None,
                     'exposure_time': None,
                     'baseline_noise': None,
                     'baseline_exposure': None}

  min_feature_vec = np.array([[2500, 100, 0, 0]], dtype=np.float32)
  max_feature_vec = np.array([[7500, 9600, 1.128964, 1]], dtype=np.float32)

  h, w, c = im.shape

  chromaticity_ML = t_camera_data[0]['chromaticity_ML']
  t_cam_temps = []
  t_cam_ISOs = []
  t_cam_apertures = []
  t_cam_exposure_times = []
  t_cam_baseline_exposures = []
  t_cam_baseline_noises = []
  for sample in t_camera_data:
    t_cam_temps.append(sample['color_temp'])
    t_cam_ISOs.append(sample['ISO'])
    t_cam_apertures.append(sample['aperture_norm'])
    t_cam_exposure_times.append(sample['exposure_time'])
    t_cam_baseline_exposures.append(sample['baseline_exposure'])
    t_cam_baseline_noises.append(sample['baseline_noise'])

  t_cam_baseline_exposures = np.array(t_cam_baseline_exposures)
  t_cam_baseline_noises = np.array(t_cam_baseline_noises)

  feature_vecs = np.array([t_cam_temps, t_cam_baseline_noises * t_cam_ISOs,
                           np.sqrt(2 ** t_cam_baseline_exposures) *
                           t_cam_exposure_times, t_cam_apertures]).transpose()

  feature_vecs = ((feature_vecs - min_feature_vec) / (
      max_feature_vec - min_feature_vec))

  in_feature_vec = np.array([[c_temp, baseline_noise * ISO,
                              np.sqrt(2 ** baseline_exposure) * exposure_time,
                              aperture_norm]])

  in_feature_vec = ((in_feature_vec - min_feature_vec) /
                    (max_feature_vec - min_feature_vec))

  nearest_inds, dists = knnsearch(feature_vecs, in_feature_vec, k)
  dists = dists / (sum(dists) + ops.EPS)
  probs = softmax(1 - dists)

  chroma = np.zeros((len(dists), 2))
  raw2xyz_matrices = np.zeros((len(dists), 3, 3))
  for i, index in enumerate(nearest_inds):
    raw2xyz_matrices[i, :, :] = t_camera_data[index]['raw2xyz']
    chroma[i, :] = t_camera_data[index]['illuminant_chromaticity_raw']

  raw2xyz = np.sum(np.reshape(probs, (-1, 1, 1)) * raw2xyz_matrices, axis=0)
  xyz2raw = np.linalg.inv(raw2xyz)

  chroma_r = sum(
    probs * chroma[:, 0]) + lambda_r * np.random.normal(0, np.std(chroma[:, 0]))
  chroma_g = predict(chroma_r, chromaticity_ML) + lambda_g * np.random.normal(
    0, np.std(chroma[:, 1]))
  chroma_b = 1 - chroma_r - chroma_g
  gt_ill = [chroma_r, chroma_g, chroma_b]
  gt_ill = np.array(gt_ill)
  gt_ill = gt_ill / ops.vect_norm(gt_ill, tensor=False, axis=0)

  im = apply_cst(im, xyz2raw)

  for c in range(3):
    im[:, :, c] = im[:, :, c] * gt_ill[c] / gt_ill[1]

  if transfer_intensity:
    iso = aper = aper_n = expo_t = bline_expo = bline_noise = t_intesity = 0

    for i, index_i in enumerate(nearest_inds):
      iso = iso + probs[i] * t_camera_data[index_i]['ISO']
      aper = aper + probs[i] * t_camera_data[index_i]['aperture']
      aper_n = aper_n + probs[i] * t_camera_data[index_i]['aperture_norm']
      expo_t = expo_t + probs[i] * t_camera_data[index_i]['exposure_time']
      bline_expo = bline_expo + probs[i] * t_camera_data[index_i][
        'baseline_exposure']
      bline_noise = bline_noise + probs[i] * t_camera_data[index_i][
        'baseline_noise']
      t_intesity = t_intesity + probs[i] * t_camera_data[index_i][
        'mean_intensity']

    output_metadata['ISO'] = iso
    output_metadata['aperture'] = aper
    output_metadata['aperture_norm'] = aper_n
    output_metadata['exposure_time'] = expo_t
    output_metadata['baseline_exposure'] = bline_expo
    output_metadata['baseline_noise'] = bline_noise
    im = im / mean_intensity * t_intesity
  else:
    output_metadata['ISO'] = ISO
    output_metadata['aperture'] = aperture
    output_metadata['aperture_norm'] = aperture_norm
    output_metadata['exposure_time'] = exposure_time
    output_metadata['baseline_exposure'] = baseline_exposure
    output_metadata['baseline_noise'] = baseline_noise

  if remove_saturated_pixels:
    im = np.reshape(im, (-1, 3))
    saturation_level = np.max(im) * saturation_level
    im[np.argwhere((im[:, 0] >= saturation_level) + (
        im[:, 1] >= saturation_level) + (im[:, 2] >= saturation_level))] = 0
    im = np.reshape(im, (h, w, 3))

  if cropping:
    im = im[round(rand() * 0.1) * h: round(h * (rand() * 0.1 + 0.9)),
         round(rand() * 0.1) * w: round(w * (rand() * 0.1 + 0.9)), :]
    h, w, _ = im.shape

  if not np.sum([h, w] == output_image_size):
    im = ops.resize_image(im, [output_image_size[1], output_image_size[0]])

  if not check_sampled_data(im, gt_ill, xyz2raw):
    logging.warning('Failed to sample')
    return False

  for c in range(3):
    im[im[:, :, c] > 1] = 1
    im[im[:, :, c] < 0] = 0

  output_metadata['gt_ill'] = gt_ill.tolist()

  if save_as_16_bits:
    im = np.array(im * (2 ** 16 - 1), dtype=np.uint16)
  else:
    im = np.array(im * (2 ** 8 - 1), dtype=np.uint8)
  im = ops.from_rgb2bgr(im)
  cv2.imwrite(os.path.join(output_dir, filename), im)

  with open(os.path.join(
      output_dir, filename.lower().replace('.png', '') + '_metadata.json'),
      'w') as outfile:
    json.dump(output_metadata, outfile)

  return True


def check_sampled_data(im, ill, cst):
  """ Checks the mapped image, illuminant value, and the inverse of CST matrix.
  """
  h, w, c = im.shape
  num_pixels = h * w
  if (np.sum(np.isnan(cst)) >= 1 or np.sum(np.isnan(im)) >= 1 or
      np.sum(np.isinf(im)) >= 1 or np.mean(im) < 0.009 or
      np.sum(im[:, :, 0] > 1) > 0.4 * num_pixels or
      np.sum(im[:, :, 1] > 1) > 0.4 * num_pixels or
      np.sum(im[:, :, 2] > 1) > 0.4 * num_pixels or
      np.sum(im[:, :, 0] < 0) > 0.4 * num_pixels or
      np.sum(im[:, :, 1] < 0) > 0.4 * num_pixels or
      np.sum(im[:, :, 2] < 0) > 0.4 * num_pixels or
      np.sum(np.isnan(ill)) >= 1):
    return False
  else:
    return True


def softmax(x):
  """ Applies softmax function: softmax(x) = np.exp(x)/sum(np.exp(x)).
  """
  return np.exp(x) / sum(np.exp(x))


def knnsearch(query, data, k):
  """ Finds the nearest K data points.

    Args:
      query: input vector (1 x d).
      data: training vectors (n x d).
      k: number of nearest data points.

    Returns:
      indices: indices of the k nearest data points.
      d: distances between the query data point and the k nearest data points.
  """

  d = np.sqrt(np.sum((query - data) ** 2, axis=1))
  indices = np.argsort(d)
  d.sort()
  indices = indices[0: k]
  d = d[0: k]
  return indices, d


def predict(x, w):
  """ Predicts a response y given a value x in a linear regression model with
    polynomial basis function.

  Args:
    x: input data point
    w: (d x 1) vector contains the weights of the basis functions.

  Returns:
    y: predicted value.
  """

  d = len(w)
  phi = np.zeros((1, d))
  for i in range(d):
    phi[0, i] = x ** (i)
  return np.matmul(phi, w)[0]


def apply_cst(im, cst):
  """ Applies CST matrix to image.

  Args:
    im: input ndarray image ((height * width) x channel).
    cst: a 3x3 CST matrix.

  Returns:
    transformed image.
  """

  result = im
  for c in range(3):
    result[:, :, c] = (cst[c, 0] * im[:, :, 0] + cst[c, 1] * im[:, :, 1] +
                       cst[c, 2] * im[:, :, 2])
  return result


def assert_target_camera(camera_models):
  """ Asserts target camera model name.

  Args:
    camera_models: a list of camera model names.
  """
  for c in range(len(camera_models)):
    target_cam = camera_models[c]
    assert (target_cam.lower() in cameras or target_cam.lower() == 'all')


def assert_xyz_img_dir(xyz_dir, ids):
  """ Asserts XYZ images in the xyz_dir directory.

  Args:
    xyz_dir: directory of the XYZ images for data augmentation.
    ids: a list of scene IDs.
  """
  for i in range(len(ids)):
    if not os.path.exists(os.path.join(xyz_dir, '%04d.png' % (ids[i] + 1))):
      print('Image %s not found!' % os.path.join(xyz_dir,
                                                 '%04d.png' % (ids[i] + 1)))
      raise FileExistsError
