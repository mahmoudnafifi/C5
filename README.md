# Cross-Camera Convolutional Color Constancy, ICCV 2021 (Oral)

*[Mahmoud Afifi](https://sites.google.com/view/mafifi)*<sup>1,2</sup>, *[Jonathan T. Barron](https://jonbarron.info/)*<sup>2</sup>, *[Chloe LeGendre](http://www.chloelegendre.com/)*<sup>2</sup>, *[Yun-Ta Tsai](https://scholar.google.com/citations?user=7fUcF9UAAAAJ&hl=en)*<sup>2</sup>, and *[Francois Bleibel](https://www.linkedin.com/in/fbleibel)*<sup>2</sup>

<sup>1</sup>York University  &ensp; <sup>2</sup>Google Research


[Paper](https://arxiv.org/pdf/2011.11890.pdf) | [Poster](https://drive.google.com/file/d/1j3FyeZpoGv638qGslARCEv3ue4Wt-38y/view) | [PPT](https://docs.google.com/presentation/d/1MTidjeFEwvoOBNAYgC3r8P9y60Axej0f/edit?rtpof=true&sd=true) | [Video](https://www.youtube.com/watch?v=BFYhbxo9jK8)

![C5_teaser](https://user-images.githubusercontent.com/37669469/103726597-4de63f80-4fa7-11eb-851a-7c35b38d8806.gif)


Reference code for the paper [Cross-Camera Convolutional Color Constancy.](https://arxiv.org/pdf/2011.11890.pdf) Mahmoud Afifi, Jonathan T. Barron, Chloe LeGendre, Yun-Ta Tsai, and Francois Bleibel. In ICCV, 2021. If you use this code, please cite our paper:
```
@InProceedings{C5,
  title={Cross-Camera Convolutional Color Constancy},
  author={Afifi, Mahmoud and Barron, Jonathan T and LeGendre, Chloe and Tsai, Yun-Ta and Bleibel, Francois},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

![C5_figure](https://user-images.githubusercontent.com/37669469/103725576-e3cc9b00-4fa4-11eb-8b63-e23de06f3673.jpg)


## Code
#### Prerequisite
* Pytorch
* opencv-python
* tqdm


#### Training 
To train C5, training/validation data should have the following formatting:
```
- train_folder/
       | image1_sensorname_camera1.png
       | image1_sensorname_camera1_metadata.json
       | image2_sensorname_camera1.png
       | image2_sensorname_camera1_metadata.json
       ...
       | image1_sensorname_camera2.png
       | image1_sensorname_camera2_metadata.json
       ...
```

In `src/ops.py`, the function `add_camera_name(dataset_dir)` can be used to rename image filenames and corresponding ground-truth JSON files. Each JSON file should include a key named either `illuminant_color_raw` or `gt_ill` that has the ground-truth illuminant color of the corresponding image. 

The training code is given in `train.py`. The following parameters are required to set model configuration and training data information. 
  * `--data-num`: the number of images used for each inference (additional images + input query image). This was mentioned in the main paper as `m`. 
  * `--input-size`: number of histogram bins. 
  * `--learn-G`: to use a `G` multiplier as explained in the paper.
  * `--training-dir-in`: training image directory.
  * `--validation-dir-in`: validation image directory; when this variable is `None` (default), the validation set will be taken from the training data based on the `--validation-ratio`.
  * `--validation-ratio`: when `--validation-dir-in` is `None`, this argument determines the validation set ratio of the image set in `--training-dir-in` directory.
  * `--augmentation-dir`: directory(s) of augmentation data (optional). 
  * `--model-name`: name of the trained model. 

The following parameters are useful to control training settings and hyperparameters:
  * `--epochs`: number of epochs
  * `--batch-size`: batch size
  * `--load-hist`: to load histogram if pre-computed (recommended). 
  * `-optimizer`: optimization algorithm for stochastic gradient descent; options are: `Adam` or `SGD`.
  * `--learning-rate`: Learning rate
  * `--l2reg`: L2 regularization factor
  * `--load`: to load C5 model from a .pth file; default is `False`
  * `--model-location`: when `--load` is True, this variable should point to the fullpath of the .pth model file.
  * `--validation-frequency`: validation frequency (in epochs).
  * `--cross-validation`: To use three-fold cross-validation. When this variable is `True`, `--validation-dir-in` and `--validation-ratio` will be ignored and 3-fold cross-validation, on the data provided in the `--training-dir-in`, will be applied. 
  * `--gpu`: GPU device ID. 
  * `--smoothness-factor-*`: smoothness loss factor of the following model components: F (conv filter), B (bias), G (multiplier layer). For example, `--smoothness-factor-F` can be used to set the smoothness loss for the conv filter. 
  * `--increasing-batch-size`: for increasing batch size during training.
  * `--grad-clip-value`: gradient clipping value; if it's set to 0 (default), no clipping is applied.


#### Testing
To test a pre-trained C5 model, testing data should have the following formatting:
```
- test_folder/
       | image1_sensorname_camera1.png
       | image1_sensorname_camera1_metadata.json
       | image2_sensorname_camera1.png
       | image2_sensorname_camera1_metadata.json
       ...
       | image1_sensorname_camera2.png
       | image1_sensorname_camera2_metadata.json
       ...
```

The testing code is given in `test.py`. The following parameters are required to set model configuration and testing data information. 
  * `--model-name`: name of the trained model.
  * `--data-num`: the number of images used for each inference (additional images + input query image). This was mentioned in the main paper as `m`.
  * `--input-size`: number of histogram bins. 
  * `--g-multiplier`: to use a `G` multiplier as explained in the paper.
  * `--testing-dir-in`: testing image directory.
  * `--batch-size`: batch size
  * `--load-hist`: to load histogram if pre-computed (recommended). 
  * `--multiple_test`: to apply multiple tests (ten as mentioned in the paper) and save their results.
  * `--white-balance`: to save white-balanced testing images.
  * `--cross-validation`: to use three-fold cross-validation. When it is set to `True`, it is supposed to have three pre-trained models saved with a postfix of the fold number. The testing image filenames should be listed in .npy files located in the `folds` directory with the same name of the dataset, which should be the same as the folder name in `--testing-dir-in`. 
  * `--gpu`: GPU device ID. 

In the `images` directory, there are few examples captured by Mobile Sony IMX135 from the [INTEL-TAU](https://etsin.fairdata.fi/dataset/f0570a3f-3d77-4f44-9ef1-99ab4878f17c) dataset. To white balance these raw images, as shown in the figure below, using a C5 model (trained on DSLR cameras from [NUS](http://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html) and [Gehler-Shi](https://www2.cs.sfu.ca/~colour/data/shi_gehler/) datasets), use the following command:

`python test.py --testing-dir-in ./images --white-balance True --model-name C5_m_7_h_64`

  
 ![c5_examples](https://user-images.githubusercontent.com/37669469/128657485-b93e47b5-c52d-46a3-86b7-0cdd771084bc.jpg)
 
 To test with the gain multiplie, use the following command:
 
 `python test.py --testing-dir-in ./images --white-balance True --g-multiplier True --model-name C5_m_7_h_64_w_G`
 
  
Note that in testing, C5 does not require any metadata. The testing code only uses JSON files to load ground-truth illumination for comparisons with our estimated values.


#### Data augmentation
The raw-to-raw augmentation functions are provided in `src/aug_ops.opy`. Call the `set_sampling_params` function to set sampling parameters (e.g., excluding certain camera/dataset from the soruce set, determine the number of augmented images, etc.). Then, call the `map_raw_images` function to generate a new augmentation set with the determined parameters. The function `map_raw_images` takes four arguments:
* `xyz_img_dir`: directory of XYZ images; you can download the CIE XYZ images from [here](https://drive.google.com/file/d/1ylf1AnjcdNBbSINS4t6rlfb5U2RJeKQT/view?usp=sharing). All images were transformed to the CIE XYZ space after applying the black-level normalization and masking out the calibration object (i.e., the color rendition chart or SpyderCUBE). 
* `target_cameras`: a list of one or more of the following camera models:
`Canon EOS 550D`, `Canon EOS 5D`, `Canon EOS-1DS`, `Canon EOS-1Ds Mark III`, `Fujifilm X-M1`, `Nikon D40`, `Nikon D5200`, `Olympus E-PL6`, `Panasonic DMC-GX1`, `Samsung NX2000`, `Sony SLT-A57`, or `All`.
* `output_dir`: output directory to save the augmented images and their metadata files.
* `params`: sampling parameters set by the `set_sampling_params` function.


