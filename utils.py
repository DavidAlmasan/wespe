import time
# import tensorflow as tf
import os
from scipy import ndimage, misc
import re
import imageio
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import scipy.stats as st
# import tensorflow_datasets as tfds
import cv2
import shutil
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
import subprocess
# import deep_learning_marcanthia.SCRIPTS.segmentation.semantic_model_testing as sem_model


def _instance_norm(net):

    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift

def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter

def rgb2grey(img):
    if img.shape[-1] == 4: img = img[:, :, 0]
    if img.shape[-1] == 3:
        return np.expand_dims(np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]), axis = -1)
    elif len(img.shape) == 2 or img.shape[-1] == 1:
        return img
    else: raise ValueError('not known img type')

def blur_img(img):
    return cv2.blur(img, (3, 3))

def noisy(noise_typ,image, var = 40):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = np.clip(image + gauss, 0, 255)
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

def pad_image(img, padSize = 1):
    if padSize == 0:
        return img
    return np.expand_dims(cv2.copyMakeBorder(img.copy(), padSize, padSize, padSize, padSize, borderType = cv2.BORDER_REFLECT), axis = - 1)

def remove_padding(img, padSize = 1):
    if padSize == 0:
        return img
    if len(img.shape) == 2: img = np.expand_dims(img, axis = -1)
    elif len(img.shape) == 4:
     img = img[0]
    elif len(img.shape) == 3: img = np.expand_dims(rgb2grey(img[:, :, 0]), axis = -1)
    return img[padSize:img.shape[0] - padSize, padSize:img.shape[1] - padSize, :]

def img_to_patches(img, patchSize, padding = 0, verbose = False):
    if len(img.shape) == 2: img = np.expand_dims(img, axis = -1)
    elif len(img.shape) == 3: pass
    # elif len(img.shape) == 4 and img.shape[0] == 1: img = img.reshape((img.shape[1], img.shape[2], img.shape[3]))
    elif len(img.shape) == 4 and img.shape[0] == 1: img = img[0]
    else: 
        return 0  # image not possible to reshape to patches so just send ignore signal
    if verbose:
        print('------------------')
        print('In img_to_patches:')
        print('img shape: ', img.shape)
        print('patchSize:', patchSize)
        print('padding: ', padding)
    paddingX = (img.shape[0] // patchSize + 1 ) * patchSize - img.shape[0]
    if paddingX == patchSize:
        paddingX = 0
    paddingY = (img.shape[1] // patchSize + 1 ) * patchSize - img.shape[1]
    if paddingY == patchSize:
        paddingY = 0
    # print('paddings: ', paddingY, paddingX)
    # print(paddingY, paddingX)
    newImg = cv2.copyMakeBorder(img.copy(), 0, paddingY, 0, paddingX, borderType = cv2.BORDER_REFLECT)
    # print(newImg.shape, "newimg shape")
    # print('newImg: ', newImg)
    patches = []
    for i in range(newImg.shape[0] // patchSize):
        for j in range(newImg.shape[1] // patchSize):
            if padding == 0:
                patch = newImg[i * patchSize: (i + 1) * patchSize, j * patchSize: (j + 1) * patchSize]
            else:
                patch = pad_image(newImg[i * patchSize: (i + 1) * patchSize, j * patchSize: (j + 1) * patchSize], padding)
            patches.append(patch)
    if padding == 0:
        return np.expand_dims(np.array(patches), axis = -1), paddingX, paddingY
    else: 
        return np.array(patches), paddingX, paddingY

    # return image.extract_patches_2d(np.expand_dims(newImg, axis = -1), (patchSize, patchSize))

def patches_to_img(patches, patchSize, crop = None, padding = 0, verbose = False):
    if verbose:
        print('-----------------')
        print('In patches_to_img: ')
        print('patches shape: ', patches.shape)
        print('patchSize:', patchSize)
        print('crop:', crop)
    rows, cols = int(np.sqrt(patches.shape[0])), int(np.sqrt(patches.shape[0]))
    if rows == 1 and cols == 1:
        return remove_padding(patches.reshape((patches.shape[-3], patches.shape[-2], 1)), padSize = padding)
    patches = patches.reshape((rows, cols, patchSize + padding * 2, patchSize + padding * 2, 1))
    newImg = None
    for row in range(rows):
        newRow = None
        for col in range(cols):
            try:
                a = remove_padding(patches[row, col, :, :, :].reshape((patchSize + padding * 2, patchSize + padding * 2, 1)), padSize = padding)
                newRow = np.concatenate((newRow, a), axis = 1)
            except:
                newRow = remove_padding(patches[row, col, :, :, :], padSize = padding)
        try: newImg = np.concatenate((newImg, np.array(newRow)), axis = 0)
        except:
            newImg = newRow
    if crop is not None:
        if 0 in crop:
            return newImg
        return newImg[:-crop[0], :-crop[1], :]
    return newImg    

def load_data(folder, patchSize = 100, verbose = False, kSize = 9, lim_ = None):
    images = list()
    for root, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                filepath = os.path.join(root, filename)
                image = imageio.imread(filepath)
                if len(image.shape) == 2: image = np.expand_dims(image, axis = -1)
                images.append(image)
    data = list()
    for image in images:
        if lim_ == 0:
            break         
        greyImg = rgb2grey(image)  
        greyImg = cv2.copyMakeBorder(greyImg.copy(), kSize//2, kSize//2, kSize//2, kSize//2, borderType = cv2.BORDER_REFLECT)    
        if len(greyImg.shape) == 2: greyImg = np.expand_dims(greyImg, axis = -1)
        data.append(greyImg)
        if lim_ is not None:
            lim_ -= 1
    img1 = np.expand_dims(np.array(data.pop(0)), axis = 0)
    patches = tf.image.extract_patches(images=img1,
                                    sizes=[1, patchSize + kSize - 1, patchSize + kSize - 1, 1],
                                    strides=[1, patchSize, patchSize, 1],
                                    rates=[1, 1, 1, 1],
                                    padding='VALID').numpy().reshape((-1, patchSize + kSize - 1, patchSize + kSize - 1, 1))
    print('in load data:', patches.shape)
    for img in data:
        newPatches = tf.image.extract_patches(images=np.expand_dims(img, axis = 0),
                                        sizes=[1, patchSize + kSize - 1, patchSize + kSize - 1, 1],
                                        strides=[1, patchSize, patchSize, 1],
                                        rates=[1, 1, 1, 1],
                                        padding='VALID').numpy().reshape((-1, patchSize + kSize - 1, patchSize + kSize - 1, 1))
        patches = np.concatenate((patches, newPatches), axis = 0)
    
    return (np.array(patches) - 127.5 ) / 127.5

def load_test_img_patches(testImg, patchSize = 100, kSize = 9):
    if len(testImg.shape) == 4:
        testImg = testImg[-1]
    if testImg.shape[-1] != 1:
        testImg = rgb2grey(testImg)
    testImg = cv2.copyMakeBorder(testImg.copy(), kSize//2, kSize//2, kSize//2, kSize//2, borderType = cv2.BORDER_REFLECT)    
    if len(testImg.shape) == 2: testImg = np.expand_dims(testImg, axis = -1)
    if len(testImg.shape) == 3: testImg = np.expand_dims(testImg, axis = 0)
    return tf.image.extract_patches(images=testImg,
                                    sizes=[1, patchSize + kSize - 1, patchSize + kSize - 1, 1],
                                    strides=[1, patchSize, patchSize, 1],
                                    rates=[1, 1, 1, 1],
                                    padding='VALID').numpy().reshape((-1, patchSize + kSize - 1, patchSize + kSize - 1, 1))

def load_dummy_data(dataset, patchSize = 100, overlap = True, kSize = 9, corrupt_types = ['gauss']):
    for corrupt_type in corrupt_types:
        assert corrupt_type in ['s&p', 'gauss', 'poisson', 'speckle', 'blur']
    data = tfds.load(dataset)
    domA, domB = list(), list()
    try:
        train, test = data["train"], data["test"]
    except: train = data["train"]
    for example in tfds.as_numpy(train):
        greyImg = rgb2grey(example['image'])
        if overlap:    
            greyImg = cv2.copyMakeBorder(greyImg.copy(), kSize//2, kSize//2, kSize//2, kSize//2, borderType = cv2.BORDER_REFLECT)    
            if len(greyImg.shape) == 2: greyImg = np.expand_dims(greyImg, axis = -1)
            if len(greyImg.shape) == 3: greyImg = np.expand_dims(greyImg, axis = 0)  # 4d Tensor
            patches =  tf.image.extract_patches(images=greyImg,
                                                sizes=[1, patchSize + kSize - 1, patchSize + kSize - 1, 1],
                                                strides=[1, patchSize, patchSize, 1],
                                                rates=[1, 1, 1, 1],
                                                padding='VALID').numpy().reshape((-1, patchSize + kSize - 1, patchSize + kSize - 1, 1))
            
            for patch in patches:
                corrupted_patch = patch
                for corrupt_type in corrupt_types:
                    if corrupt_type == 'blur':
                        corrupted_patch = np.expand_dims(blur_img(rgb2grey(corrupted_patch)), axis = -1)
                        assert(len(corrupted_patch.shape) == 3)
                    else:
                        corrupted_patch = noisy(corrupt_type, rgb2grey(corrupted_patch))

                domA.append(rgb2grey(corrupted_patch))
                domB.append(rgb2grey(patch))
                
        else:
            patches, _, _= img_to_patches(greyImg, patchSize, verbose = False)
            for patch in patches:
                corrupted_patch = patch
                for corrupt_type in corrupt_types:
                    if corrupt_type == 'blur':
                        corrupted_patch = np.expand_dims(blur_img(rgb2grey(corrupted_patch)), axis = -1)
                        assert(len(corrupted_patch.shape) == 3)
                    else:
                        corrupted_patch = noisy(corrupt_type, rgb2grey(corrupted_patch))

                domA.append(rgb2grey(corrupted_patch))
                domB.append(rgb2grey(patch))
    
    domA, domB = (np.array(domA) - 127.5 ) / 127.5, (np.array(domB) - 127.5 ) / 127.5
    return domA, domB

def generate_and_save_images(model, epoch = None, test_input = None, patchSize = 100, kSize = 9, ckpt_folder = None, test = False, type_ = 'orig'):
    saveFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    saveFolder = os.path.join(saveFolder, ckpt_folder)
    saveFolder = os.path.join(saveFolder, 'images')
    # TODO save test image and save the output from the model
    try: os.makedirs(saveFolder, exist_ok = True)
    except:pass

    fig = plt.figure()
    if test:
        #plt.imshow(test_input[0, :, :, 0] * 127.5 + 127.5, cmap='gray')
        #plt.axis('off')
        imgPath = os.path.join(saveFolder, 'test_image' + type_ + '.png')
        # plt.savefig(imgPath, bbox_inches='tight', pad_inches=0)
        cv2.imwrite(imgPath, test_input[0, :, :, 0] * 127.5 + 127.5)
    else:
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        print('Generating enhanced image...')
        predictions = model(test_input, training=False).numpy()[:, kSize//2:-(kSize//2),kSize//2:-(kSize//2) :]
        print('Predictions shape: ', predictions.shape)
        newImg = patches_to_img(predictions, patchSize, verbose = False)
        # print('Output image pixel values (zero mean)', np.unique(newImg))
        #plt.imshow(newImg[:, :, 0] * 127.5 + 127.5, cmap='gray')
        #plt.axis('off')
        imgPath = os.path.join(saveFolder, 'image_at_epoch_{:04d}.png'.format(epoch))
        cv2.imwrite(imgPath, newImg[:, :, 0] * 127.5 + 127.5)
        # print('Image path: {}'.format(imgPath))
        # plt.savefig(imgPath, bbox_inches='tight', pad_inches=0)
        
        return newImg

def save_metrics(folder, name, title, metricVec):
    plt.figure()
    plt.plot(metricVec)
    plt.title(title)
    saveName = os.path.join(folder, name + '_' + title + '.png')
    plt.savefig(saveName, bbox_inches='tight')
    plt.close()

def plot_metrics(path):
    with open(path) as logFile:
        lines = logFile.readlines()
    for line in lines:
        line =  line.rstrip().split(':')
        if 'Generator loss' in line:
            try: genLoss.append(float(line[-1]))
            except: genLoss = [float(line[-1])]
        if 'Texture disc loss' in line:
            try: textDisc.append(float(line[-1]))
            except: textDisc = [float(line[-1])]
        if 'Color disc loss' in line:
            try: colDisc.append(float(line[-1]))
            except: colDisc = [float(line[-1])]
        if 'Content loss' in line:
            try: contentLoss.append(float(line[-1]))
            except: contentLoss = [float(line[-1])]
        if 'TV loss' in line:
            try: TV_loss.append(float(line[-1]))
            except: TV_loss = [float(line[-1])]
    return genLoss, textDisc, colDisc, contentLoss, TV_loss

def variance_map():
    # # Create images with fake noise
    # # Make sure config.py is set accordingly
    # subprocess.call(['python', 'model.py'])

    # curFolder = os.path.abspath(os.path.dirname(__file__))
    # relImgFolder = './imgs_for_variance'
    # relImgSaveFolder = os.path.join(relImgFolder, 'segmented')
    # if os.isdir(relImgSaveFolder):
    #     shutil.rmtree(relImgSaveFolder)
    # os.makedirs(relImgSaveFolder)
    # imgFolder = os.path.join(curFolder, relImgFolder)
    # images = list()
    # names = list()
    # for root, dirnames, filenames in os.walk(imgFolder):
    #         for filename in filenames:
    #             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
    #                 filepath = os.path.join(root, filename)
    #                 names.append(filename)
    #                 image = imageio.imread(filepath)
    #                 if len(image.shape) == 2: image = np.expand_dims(image, axis = -1)
    #                 images.append(image)

    # # Segment the images
    # seg_folder = './deep_learning_marcanthia/SCRIPTS/segmentation'
    # rel_to_main_folder = '../../../'
    # relModelPath = '../semseg_model_100epochs_16fdim_unet16_bce.pt'

    # for img, name in zip(images, names):
    #     saveImgPath = os.path.join(relImgSaveFolder, name[:-4] + '_segmented.png')
    #     relTestImgPath = os.path.join(relImgFolder, name)

    #     inputPath = os.path.join(rel_to_main_folder, relTestImgPath)
    #     saveImgPath = os.path.join(rel_to_main_folder, saveImgPath)
    #     sem_model.test_semantic_model(inputImgPath=inputPath,
    #                                 modelPath = relModelPath,
    #                                 saveImgPath = saveImgPath)



    # retrieve segmented images and create the variance plot 
    ### DUMMY 
    relImgFolder = './images_bulk_segmentation'
    relImgSaveFolder = os.path.join(relImgFolder, 'segmented')
    images = list()
    print('a')
    for root, dirnames, filenames in os.walk(relImgSaveFolder):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                filepath = os.path.join(root, filename)
                image = imageio.imread(filepath)
                if len(image.shape) == 2: image = np.expand_dims(image, axis = -1)
                images.append(image[:, :, 0]) #black and white
    images = np.sign(np.asarray(images))
    var = np.var(images, axis = 0)
    print('Shape of array containing segmented images is {}'.format(var.shape))
    cv2.imshow('var', var)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    path = 'checkpoints/cycle_loss_scratch_contd/logs/3-11-20-12-59.txt'  #the good one
    # path = 'checkpoints/cycle_loss_scratch_contd_2/logs/3-12-19-39-24.txt'
    # a, b, c, d, e = plot_metrics(path)
    # #%%
    # plt.plot(e)
    # plt.show()
    variance_map()




    # a, b = load_dummy_data('horses_or_humans', overlap = True, kSize = 9, patchSize = 100)
    # print(type(a))
    # print(a.shape)
    # print(a.shape, b.shape)

    #a = cv2.imread('domB/test_image_orig.png')
    #a = a[:, :, 0]
    #print(a.shape)
    #cv2.imshow("GausQsian Smoothing",a)
    #cv2.waitKey(0) # waits until a key is pressed
    #cv2.destroyAllWindows() # destroys the window showing image
    #b = blur_img(a)
    #cv2.imshow("Gaussian Smoothing",np.hstack((a, b)))
    #cv2.waitKey(0) # waits until a key is pressed
    #cv2.destroyAllWindows() # destroys the window showing image
    #print(b.shape)
