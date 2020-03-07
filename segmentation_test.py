import subprocess
import deep_learning_marcanthia.SCRIPTS.segmentation.semantic_model_testing as sem_model
import os

subprocess.call(['python', 'model.py'])

# Segment original image
seg_folder = './deep_learning_marcanthia/SCRIPTS/segmentation'
rel_to_main_folder = '../../../'
relModelPath = '../semseg_model_100epochs_16fdim_unet16_bce.pt'
saveImgPath = './model_tests/' + 'original_segmented.png'
relTestImgPath = './test_data/img.tiff'

inputPath = os.path.join(rel_to_main_folder, relTestImgPath)
saveImgPath = os.path.join(rel_to_main_folder, saveImgPath)
sem_model.test_semantic_model(inputImgPath=inputPath,
                                modelPath = relModelPath,
                                saveImgPath = saveImgPath)

# Segment enhanced image
saveImgPath = './model_tests/' +  'enhanced_segmented.png'
inputPath = os.path.join(rel_to_main_folder, './model_tests/' +  'enhanced_image.png')
saveImgPath = os.path.join(rel_to_main_folder, saveImgPath)
sem_model.test_semantic_model(inputImgPath=inputPath,
                              modelPath = relModelPath,
                              saveImgPath = saveImgPath)
