import subprocess
import deep_learning_marcanthia.SCRIPTS.segmentation.semantic_model_testing as sem_model
import os
import numpy as np 
import imageio
import re

curFolder = os.path.abspath(os.path.dirname(__file__))
relImgFolder = './images_bulk_segmentation'
relImgSaveFolder = os.path.join(relImgFolder, 'segmented')
os.makedirs(relImgSaveFolder, exist_ok=True)
imgFolder = os.path.join(curFolder, relImgFolder)
images = list()
names = list()
for root, dirnames, filenames in os.walk(imgFolder):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                filepath = os.path.join(root, filename)
                names.append(filename)
                image = imageio.imread(filepath)
                if len(image.shape) == 2: image = np.expand_dims(image, axis = -1)
                images.append(image)

# Segment the images
seg_folder = './deep_learning_marcanthia/SCRIPTS/segmentation'
rel_to_main_folder = '../../../'
relModelPath = '../semseg_model_100epochs_16fdim_unet16_bce.pt'

for img, name in zip(images, names):
    saveImgPath = os.path.join(relImgSaveFolder, name[:-4] + '_segmented.png')
    relTestImgPath = os.path.join(relImgFolder, name)


    inputPath = os.path.join(rel_to_main_folder, relTestImgPath)
    saveImgPath = os.path.join(rel_to_main_folder, saveImgPath)
    sem_model.test_semantic_model(inputImgPath=inputPath,
                                  modelPath = relModelPath,
                                  saveImgPath = saveImgPath)
