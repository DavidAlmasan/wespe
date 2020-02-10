# deep-learning-marchantia/SCRIPTS

This folder contains the relevant files for training and testing [segmentation](#segmentation) models and [creating synthetic data](#synthetic-data-generation). 

The `color_index_palette.npy` file is used for the instance segmentation labels - each cell is assigned a different color according to this file, which maps up to 4000 cells to distinct colors.

## Segmentation

The [segmentation](./segmentation) folder contains the files for training and testing the segmentation models.

The final network architecture used in this project is located in [segmentation/model_arch.py](./segmentation/model_arch.py). It handles both semantic and instance segmentation, with only the final layer differing to give the corresponding output. This is outlined in more detail in the Methods section of the [project report](../final_report_v3.pdf).

### [train_model.py](./segmentation/train_model.py)

This script trains a segmentation model. There are options for semantic segmentation and instance segmentation.
The base architecture for the neural network is a UNet with a pre-trained VGG16 encoder.

***Important!! Format of training file names***

The directory of the training data is named below. This is the directory that the [synthetic generation scripts](./synthetic_generation) use to save the training images.
- `training_images_filepath = '../data/synthesized_imgs'`

This set-up assumes the following format for the files in this directory:
`XXX_synth.png` -- the synthetic Marchantia images
`XXX_ins_gt.npy` -- the instance mask ground truth arrays
`XXX_invert_sem_gt.png` -- the semantic ground truth images with the inner cells labelled

**In order to add your own hand-labelled Marchantia images, please ensure that they are labelled in this way in this directory and cropped to the correct size.**

The following parameters must be set for both semantic and instance segmentation training:

- `image_shape = (128, 128)` The image shape of the training images. Here it is set to 128x128.
- Other training parameters
  - `learning_rate = 0.001`
  - `num_epochs = 1000` Number of epochs to run.
  - `batch_size = 20`

In order to train a *semantic* segmentation network, you must set the following parameters in this file:

- `train_semantic_network = True`
- `train_instance_network = False`
- `seg_criterion = nn.BCEWithLogitsLoss()` Specify the loss. It is recommended to use nn.BCEWithLogitsLoss(), but other losses can be used as well.
- `semseg_network_filepath = './semseg_results/final_model.pt'` Location and name of where to save the semantic segmentation network after training. This has been preset to the [semseg_results](./segmentation/semseg_results) folder. You may need to create this directory if it has not already been created.
- `semseg_results_filepath = './semseg_results'` Directory to save other semantic segmentation results. This has again been preset to the [semseg_results](./segmentation/semseg_results) folder.


To train an *instance* segmentation network, you must first ensure that a semantic segmentation model has *already been trained*. The instance segmentation model takes the pre-trained semantic segmentation network as its initialisation. Thus, you must set the following parameters in this script in order for the training to occur correctly:

- `train_semantic_network = False`
- `train_instance_network = True`
- `freeze_encoder = False` Do you want to freeze the encoder for instance segmentation training? It is recommended not to (default).
- `insseg_results_filepath = './insseg_results'` Specify the directory to save instance results. During training, the network is saved automatically every 100 epochs as well as the loss data to this location.
- Parameters for the discriminative loss function. The default setting is what was used in the original [paper](https://arxiv.org/abs/1708.02551):
  - `delta_v = 0.5`
  - `delta_d = 1.5`
  - `feature_dim = 16` Feature dimension to embed pixels for the instance segmentation


### [semantic_model_testing.py](./segmentation/semantic_model_testing.py)
This script allows you to test a specified semantic segmentation model that you've trained and save the corresponding output image. The parameters needed to run this script are outlined at the top of the file.


### [instance_model_testing.py](./segmentation/instance_model_testing.py)
This script allows you to test a specified instance segmentation model that you've trained and save the corresponding output image. The parameters needed to run this script are outlined at the top of the file.


### [recursive_seg.py](./segmentation/recursive_seg.py)

This script will recursively segment a given image by reconsidering each class that the model outputs and then inputting an image with only those pixels being left (the rest blacked out) and segment again.

Note: this algorithm can take a very long time - might be worth using a smaller or downscaled image.

The parameters needed to run this script are outlined at the top of the file.

## Synthetic Data Generation

The [synthetic_generation](./synthetic_generation) folder contains the files for generating synthetic images of Marchantia polymorpha.

Synthetic image generation is a 3 step process:
1. Create Voronoi diagrams - this also serves as the ground truth image
2. Vary the grayscale value of the Voronoi diagrams to emulate the variable brightness in the Marchantia images
3. Perform neural style transfer, using a real Marchantia image as the 'style' image

To create the synthetic images, run these scripts in this order:
1. voronoi_generation.py
2. generate_instance_gt.py
3. neural_style_transfer.py


### [voronoi_generation.py](./synthetic_generation/voronoi_generation.py)

This script generates Voronoi diagrams that serve as ground truth data for the synthetic dataset. It also generates the content images to be input into the neural style transfer script by varying the grayscale value of the diagram.

It is important that the location to where the images will be saved is consistent with the location specified in [train_model.py](./segmentation/train_model.py). The ground truth images will be labelled XXX_sem_gt.png, and the grayscale content images will be labelled XXX_gray.png. The inverted ground truth image (with inner cells labelled) will also be saved as XXX_invert_sem_gt.png
- `save_image_path = '../data/synthesized_imgs'`

The parameters needed to run this script are outlined at the top of the file.

### [neural_style_transfer.py](./synthetic_generation/neural_style_transfer.py)

This script runs the neural style transfer, given a content and style image. 

The content image will be the generated Voronoi diagram image with the grayscale variation. If created by the voronoi_generation.py script, this will have the form XXX_gray.png

The style image will be a Marchantia image, cropped to the same size as the desired synthetic image. It is currently set as

- `style_img_filepath = 'g2_t017_c001_cropped2.png'`

The other parameters needed to run this script are detailed at the top of the file.

### [generate_instance_gt.py](./synthetic_generation/generate_instance_gt.py)

This script generates the ground truth data for instance segmentation, given the ground truth images for the semantic segmentation (cells themselves labelled white, everything else is black).

Locations of the semantic ground truth data - **IMPORTANT**: must be ground truth with inner cells labelled! NOT cell walls
- `file_names = glob.glob('../data/synthesized_imgs/*_invert_sem_gt.png')`

The other parameters needed to run this script are detailed at the top of the file.

