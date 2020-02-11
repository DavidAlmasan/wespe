import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import PIL
import json
import tensorflow as tf
import datetime
try:import tensorflow_addons as tfa
except: pass
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Add, LeakyReLU, Lambda, ReLU
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, DepthwiseConv2D, GaussianNoise
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose, DepthwiseConv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.backend import clear_session
import os
from scipy import ndimage, misc
import time
import re
from IPython import display
import cv2 

from utils import load_data, generate_and_save_images, load_dummy_data, img_to_patches, patches_to_img, remove_padding, load_test_img_patches
from utils import gauss_kernel, _instance_norm, save_metrics
from loss_functions import content_loss, color_loss, texture_loss, content_loss_v2, tv_loss, total_loss_agg
import deep_learning_marcanthia.SCRIPTS.segmentation.semantic_model_testing as sem_model
class WESPE():
    def __init__(self, configFilePath, dummyData = False, trainMode = True, laptop = False):
        # Config file
        self.laptop = laptop
        self.curFolder = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.curFolder, configFilePath)) as configTXT:
            self.config = json.load(configTXT)

        # Params
        self.BATCH_SIZE = self.config['batch_size']
        self.EPOCHS = self.config['epochs']
        self.genEta = self.config['gen_eta']
        self.discEta = self.config['disc_eta']
        self.init = RandomNormal(stddev = self.config.get('init_stddev', 0.01))
        self.genTypes = ['G', 'F', 'CYCLEGAN']
        self.discTypes = ['color', 'texture']
        self.numFilters = self.config.get('num_filters', 64)
        self.patchSize = self.config.get('patch_size', 100)
        self.save_ckpt_dir = self.config.get('save_ckpt_folder', None)
        self.load_ckpt_dir = self.config.get('load_ckpt_folder', None)
        self.dummyData = self.config.get('dummy_dataset', None)
        self.contentW = self.config.get('content_w', 1) # default from WESPE paper
        self.textureW = self.config.get('texture_w', 1e-3)
        self.colorW = self.config.get('color_w', 1e-3)
        self.tvW = self.config.get('tv_w', 10)
        self.useCycleGAN = str(self.config.get('use_cycle_gan', 'no')).lower() in ['yes', '1', 'true']
        self.testImgPath = self.config.get('test_img_path', None)
        self.discrimEpochs = self.config.get('discrim_epochs', 1)
        self.genEpochs = self.config.get('generator_epochs', 1)
        self.labelFlippingPeriod = int(self.config.get('label_flipping_period', None))
        self.data_corruption_types = self.config.get('data_corruption_types', ['gauss'])
        if self.labelFlippingPeriod == 0:
            self.labelFlippingPeriod = None
        self.discrimNoiseSTDDEV = self.config.get('discrim_noise_stddev', 0.01)
        self.kSize = 9  # TODO automatically assume somehow

        # Blur kernels  #TODO make this config
        self.kernel_size=23
        self.std = 3
        self.blur_kernel_weights = gauss_kernel(self.kernel_size, self.std, 1)
        # Optimizers
        self.G_optimizer = tf.keras.optimizers.Adam(self.genEta)
        self.F_optimizer = tf.keras.optimizers.Adam(self.genEta)
        self.colorDisc_optimizer = tf.keras.optimizers.Adam(self.discEta)
        self.textDisc_optimizer = tf.keras.optimizers.Adam(self.discEta)

        # Logger
        self.logFolder = os.path.join(self.curFolder, 'checkpoints')
        self.logFolder = os.path.join(self.logFolder, self.save_ckpt_dir)
        self.logFolder = os.path.join(self.logFolder, 'logs')
        try: os.makedirs(self.logFolder, exist_ok = True)
        except: pass  # exists
        now = datetime.datetime.now()
        timeName = str(now.month) +  '-' + str(now.day) + '-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second)
        self.logFilePath = os.path.join(self.logFolder, timeName + '.txt')
        with open(self.logFilePath, 'w') as logFile:
            logFile.write('CONFIG: \n\n\n')
            logFile.write("Batch size: " + str(self.BATCH_SIZE) + '\n')
            logFile.write('Epochs: ' + str(self.EPOCHS) + '\n')
            logFile.write('Generator Learning Rate: ' + str(self.genEta) + '\n')
            logFile.write('Discriminator Learning Rate: ' + str(self.discEta) + '\n')
            logFile.write('Number of filters: ' + str(self.numFilters) + '\n')
            logFile.write('Saving checkpoints to folder: ' + str(self.save_ckpt_dir) + '\n')
            logFile.write('Loading checkpoints from folder: ' + str(self.load_ckpt_dir) + '\n')
            logFile.write('Cyclegan used in generator: ' + str(self.useCycleGAN) + '\n')
            logFile.write('Patch size: ' + str(self.patchSize) + '\n')
            logFile.write('Dummy data used: '+ str(self.dummyData))
            logFile.write('Test image used: ' + str(self.testImgPath) + '\n')
            logFile.write(' \n LOSS PARAMETERS: \n')
            logFile.write('Content weight: ' + str(self.contentW) + '\n')
            logFile.write('Texture weight: ' + str(self.textureW) + '\n')
            logFile.write('Color weight: ' + str(self.colorW) + '\n')
            logFile.write('TV weight: ' + str(self.tvW) + '\n')       
            logFile.write('---------------------------\n' + '\n' + '\n')
        
        # Plotting of params
        self.metricsFolder = os.path.join(self.curFolder, 'checkpoints') 
        self.metricsFolder = os.path.join(self.metricsFolder, self.save_ckpt_dir)
        self.metricsFolder = os.path.join(self.metricsFolder, 'metrics_plots')
        try: os.makedirs(self.metricsFolder, exist_ok = True)
        except: pass  # exists

        self.colorLoss_hist = []
        self.textLoss_hist = []
        self.contentLoss_hist = []
        self.tvLoss_hist = []
        self.totalLoss_hist = []

        # Load data
        if self.dummyData is not None:
            self.domA, self.domB = load_dummy_data(self.dummyData, self.patchSize, overlap = True, kSize = self.kSize, corrupt_types=self.data_corruption_types)
            

        else:
            domA_folder = os.path.join(self.curFolder, self.config['domA_folder'])
            domB_folder = os.path.join(self.curFolder, self.config['domB_folder'])
            self.domA = load_data(domA_folder, self.patchSize, kSize = self.kSize, lim_ = 20)
            self.domB = load_data(domB_folder, self.patchSize, kSize = self.kSize, lim_ = 20)
            len_ = min(self.domA.shape[0], self.domB.shape[0])
            self.domA = self.domA[:len_]
            self.domB = self.domB[:len_]


        # Check data shapes
        aShape, bShape = self._get_data_shape()
        print('Data shape: \n', aShape, bShape)
        
        assert len(list(aShape)) == 4 and len(list(bShape)) == 4 and \
            aShape[-1] == 1 and bShape[-1] == 1
        self.imgShape = self.domA[0].shape
        assert self.imgShape[0] == self.patchSize + 2 * (self.kSize // 2), self.imgShape[1] == self.patchSize + 2 * (self.kSize // 2) 
        self.testImg_orig = np.expand_dims(self.domB[4], axis = 0)
        self.testImg_noise = np.expand_dims(self.domA[4], axis = 0)
        # Test image
        if self.testImgPath is not None:
            print('Loading test data from test data folder')
            self.relTestImgPath = self.testImgPath
            self.testImgPath = os.path.join(self.curFolder, self.testImgPath)
            images = list()
            for root, dirnames, filenames in os.walk(self.testImgPath):
                for filename in filenames:
                    if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                        filepath = os.path.join(root, filename)
                        image = imageio.imread(filepath)
                        if len(image.shape) == 2: image = np.expand_dims(image, axis = -1)
                        image = (image - 127.5 ) / 127.5
                        images.append(image)
            assert len(images) == 1
            image = images[-1]
            self.testImg_patches = load_test_img_patches(image, patchSize = self.patchSize, kSize = self.kSize)
        else:
            print('Using fourth patch form train data as test image')
            self.testImg_patches = self.testImg_noise

        # VGG19 model
        img_input = Input(shape=self.imgShape)
        input_tensor = Concatenate()([img_input, img_input, img_input])
        vgg_model = VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)
        print('Compiled VGG19 model...')
        layer_name="block2_conv2"
        self.inter_VGG_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer(layer_name).output)
        self.inter_VGG_model.trainable=False
        self.inter_VGG_model.compile(optimizer='rmsprop', loss='mse')
        self.inter_VGG_model.trainable=False
        
        # Create models
        self.create_models()
        
        # Checkpoints
        self.save_checkpoint_prefix = os.path.join(self.curFolder, 'checkpoints')
        self.save_checkpoint_prefix = os.path.join(self.save_checkpoint_prefix, self.save_ckpt_dir)
        self.save_checkpoint_prefix = os.path.join(self.save_checkpoint_prefix, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(g_optimizer = self.G_optimizer,
                                             f_optimizer = self.F_optimizer,
                                             colorDisc_optimizer = self.colorDisc_optimizer, 
                                             textDisc_optimizer = self.textDisc_optimizer,
                                             G = self.G,
                                             F = self.F,
                                             textDisc = self.textDisc,
                                             colorDisc = self.colorDisc)
        if self.load_ckpt_dir is not None:
            load_ckpt_dir = os.path.join(self.curFolder, 'checkpoints')
            self.load_ckpt_dir = os.path.join(load_ckpt_dir, self.load_ckpt_dir)
            if trainMode:
                status = self.checkpoint.restore(tf.train.latest_checkpoint(self.load_ckpt_dir))
            else:
                status = self.checkpoint.restore(tf.train.latest_checkpoint(self.load_ckpt_dir)).expect_partial()
            print('Restored latest checkpoint from folder {}'.format((tf.train.latest_checkpoint(self.load_ckpt_dir))))


        # Train or test
        self.domA = tf.data.Dataset.from_tensor_slices(self.domA).batch(self.BATCH_SIZE, drop_remainder=True)
        self.domB = tf.data.Dataset.from_tensor_slices(self.domB).batch(self.BATCH_SIZE, drop_remainder=True)


        if trainMode:
            print("Training...")
            self.train(self.domA, self.domB, self.EPOCHS)
        else:
            print('Testing...')
            output = self.colorDisc(self.testImg_patches, training = False).numpy()
            testFolder = os.path.join(self.curFolder, 'model_tests')
            # timeName = timeName = str(now.month) +  '-' + str(now.day) + '-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second)
            # testFolder = os.path.join(testFolder, timeName)
            try: os.makedirs(testFolder, exist_ok = True)
            except: pass

            # Enhance image
            print('Enhancing image...')
            predictions = self.G(self.testImg_patches, training=False).numpy()[:, self.kSize//2:-(self.kSize//2),self.kSize//2:-(self.kSize//2) :]
            print('img enhanced')
            newImg = patches_to_img(predictions, self.patchSize, verbose = False)
            print('a')
            plt.imshow(newImg[:, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
            enhImgPath = os.path.join(testFolder, 'enhanced_image.png')
            print('b')
            cv2.imwrite(enhImgPath, newImg[:, :, 0])
            

    def test_model(self, _type, testImgPatches, crop = None, ploting = False):
        # assert testImg.shape == self.imgShape
        # print('Testing model: ', model.name)
        _type = _type.upper()
        assert _type in ['G', 'CYCLEGAN']
        generator = self.get_generator_model(self.numFilters, _type)
        status = tf.train.Checkpoint(net = generator).restore(tf.train.latest_checkpoint(self.latest_checkpoint_dir)).expect_partial()
        predictions = generator(testImgPatches, training=False).numpy()
        # predictions = model(testImgPatches, training=False).numpy()
        print(np.unique(predictions))
        print(predictions.shape)
        newImg = patches_to_img(predictions, self.patchSize, crop, padding = self.patchPadding, verbose = False)
        # print(prediction.shape)
        # return
        if ploting:
            plt.figure()
            # plt.imshow(testImg[0, :, :, 0] * 127.5 + 127.5, cmap='gray')
            # plt.title("Orig")
            # plt.figure()
            plt.imshow(newImg[:, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.title("Enhanced")
            plt.show()

    def create_models(self):
        if self.useCycleGAN:
            self.G = self.get_generator_model(self.numFilters, 'CYCLEGAN')
            self.F = self.get_generator_model(self.numFilters, 'CYCLEGAN')
        else:
            self.G = self.get_generator_model(self.numFilters, 'G')
            self.F = self.get_generator_model(self.numFilters, 'F')

        self.colorDisc = self.get_discriminator_model(self.numFilters, 'color')
        self.textDisc = self.get_discriminator_model(self.numFilters, 'texture')

    def get_generator_model(self, numFilters, _type):
        _type = _type.upper()
        assert _type in self.genTypes
        def ResBlock(featuresIn, numFilters, num):
            temp = Conv2D(numFilters,
                          (3, 3),
                          strides = 1,
                          padding = 'SAME',
                          name = ('ResBlock_%d_CONV_1' %num),
                          kernel_initializer=self.init)(featuresIn)
            #if self.laptop: temp = BatchNormalization(axis=-1, scale = False)(temp)
            #else: temp = tfa.layers.normalizations.InstanceNormalization(axis=-1)(temp)

            # temp = LeakyReLU(alpha=0.2)(temp)
            temp = ReLU()(temp)

            temp = Conv2D(numFilters,
                          (3, 3),
                          strides = 1,
                          padding = 'SAME',
                          name = ('ResBlock_%d_CONV_2' %num),
                          kernel_initializer=self.init)(temp)
            #if self.laptop: temp = BatchNormalization(axis=-1, scale = False)(temp)
            #else: temp = tfa.layers.normalizations.InstanceNormalization(axis=-1)(temp)
            # temp = LeakyReLU(alpha=0.2)(temp)
            temp = ReLU()(temp)

            return Add()([temp, featuresIn])

        # Create the generator model
        image = Input(self.imgShape)
        proc = Lambda(lambda x: x, output_shape=lambda x:x)(image)


        # Processing layers
        if _type == 'CYCLEGAN':
            proc = Conv2D(numFilters,
                        (9, 9),
                        strides = 1,
                        padding = 'SAME',
                        name = ('Generator_%s_CYCLE_CONV_1_1' %_type),
                        # activation = 'relu',
                        kernel_initializer=self.init)(proc)
            proc = LeakyReLU(alpha=0.2)(proc)

            proc = Conv2D(numFilters,
                        (3, 3),
                        strides = 1,
                        padding = 'SAME',
                        name = ('Generator_%s_CYCLE_CONV_1_2' %_type),
                        # activation = 'relu',
                        kernel_initializer=self.init)(proc)
            proc = LeakyReLU(alpha=0.2)(proc)

            proc = Conv2D(numFilters,
                        (3, 3),
                        strides = 1,
                        padding = 'SAME',
                        name = ('Generator_%s_CYCLE_CONV_1_3' %_type),
                        # activation = 'relu',
                        kernel_initializer=self.init)(proc)
            proc = LeakyReLU(alpha=0.2)(proc)

        else:
            proc = Conv2D(numFilters,
                     (9, 9),
                     strides = 1,
                     padding = 'SAME',
                     name = ('Generator_%s_CONV_1' %_type),
                     activation = 'relu',
                     kernel_initializer=self.init)(proc)
            # proc = LeakyReLU(alpha=0.2)(proc)
        # Residual blocks
        proc = ResBlock(proc, numFilters, 1)
        proc = ResBlock(proc, numFilters, 2)
        proc = ResBlock(proc, numFilters, 3)
        proc = ResBlock(proc, numFilters, 4)
        
        if _type == 'CYCLEGAN':
        ### CycleGAN architecture with transposeConv2d layers
            proc = Conv2DTranspose(numFilters,
                        (3, 3),
                        strides = 1,
                        padding = 'SAME',
                        name = ('Generator_%s_CYCLE_DECONV_1' %_type),
                        # activation = 'relu',
                        kernel_initializer=self.init)(proc)
            proc = LeakyReLU(alpha=0.2)(proc)

            proc = Conv2DTranspose(numFilters,
                        (3, 3),
                        strides = 1,
                        padding = 'SAME',
                        name = ('Generator_%s_CYCLE_DECONV_2' %_type),
                        # activation = 'relu',
                        kernel_initializer=self.init)(proc)
            proc = LeakyReLU(alpha=0.2)(proc)

            proc = Conv2DTranspose(numFilters,
                        (9, 9),
                        strides = 1,
                        padding = 'SAME',
                        name = ('Generator_%s_CYCLE_DECONV_3' %_type),
                        # activation = 'relu',
                        kernel_initializer=self.init)(proc)
            proc = LeakyReLU(alpha=0.2)(proc)

        else:
        ### Orig architecture with onvolutional layers
            proc = Conv2D(numFilters,
                         (3, 3),
                         strides = 1,
                         padding = 'SAME',
                         name = ('Generator_%s_CONV_2' %_type),
                         activation = 'relu',
                         kernel_initializer=self.init)(proc)
            # proc = LeakyReLU(alpha=0.2)(proc)

            proc = Conv2D(numFilters,
                         (3, 3),
                         strides = 1,
                         padding = 'SAME',
                         name = ('Generator_%s_CONV_3' %_type),
                         activation = 'relu',
                         kernel_initializer=self.init)(proc)
            # proc = LeakyReLU(alpha=0.2)(proc)

            # proc = Conv2D(numFilters,
            #              (3, 3),
            #              strides = 1,
            #              padding = 'SAME',
            #              name = ('Generator_%s_CONV_4' %_type),
            #              activation = 'relu',
            #              kernel_initializer=self.init)(proc)
            # # proc = LeakyReLU(alpha=0.2)(proc)

        proc = Conv2D(1,
                     (9, 9),
                     strides = 1,
                     padding = 'SAME',
                     name = ('Generator_%s_CONV_FINAL' %_type),
                     activation = 'tanh',
                     kernel_initializer=self.init)(proc)
        # proc = Lambda(lambda x: 0.58 * x + 0.5, output_shape=lambda x:x)(proc)

        return Model(inputs=image, outputs=proc, name=_type)

    def get_discriminator_model(self, numFilters, _type):
        _type = _type.lower()
        assert _type in self.discTypes

         # Create the generator model
        image = Input(self.imgShape)
        proc = Lambda(lambda x: x, output_shape=lambda x:x)(image) 

        if _type == 'color':  #blur image
            g_layer = DepthwiseConv2D(self.kernel_size, use_bias=False, padding='SAME')
            proc = g_layer(image)
            
            g_layer.set_weights([self.blur_kernel_weights])
            g_layer.trainable = False
        else:
            proc = image

        # Some noise to improve the training
        # proc = GaussianNoise(stddev = 0.01)(proc)

        #Convolutions
        proc = Conv2D(48,
                    (11, 11),
                    strides = 4,
                    padding = 'SAME',
                    name = ('Discriminator_%s_CONV_1' %_type),
                    kernel_initializer=self.init)(proc)
        proc = LeakyReLU(alpha=0.2)(proc)

        proc = Conv2D(128,
                    (5, 5),
                    strides = 2,
                    padding = 'SAME',
                    name = ('Discriminator_%s_CONV_2' %_type),
                    kernel_initializer=self.init)(proc)
        proc = LeakyReLU(alpha=0.2)(proc)
        if self.laptop: proc = BatchNormalization(axis=-1, scale = False)(proc)
        else: proc = tfa.layers.normalizations.InstanceNormalization(axis=-1)(proc)

        
        

        proc = Conv2D(192,
                    (3, 3),
                    strides = 1,
                    padding = 'SAME',
                    name = ('Discriminator_%s_CONV_3' %_type),
                    kernel_initializer=self.init)(proc)
        proc = LeakyReLU(alpha=0.2)(proc)
        if self.laptop: proc = BatchNormalization(axis=-1, scale = False)(proc)
        else: proc = tfa.layers.normalizations.InstanceNormalization(axis=-1)(proc)
        
        
        
        proc = Conv2D(192,
                    (3, 3),
                    strides = 1,
                    padding = 'SAME',
                    name = ('Discriminator_%s_CONV_4' %_type),
                    kernel_initializer=self.init)(proc)
        proc = LeakyReLU(alpha=0.2)(proc)
        if self.laptop: proc = BatchNormalization(axis=-1, scale = False)(proc)
        else: proc = tfa.layers.normalizations.InstanceNormalization(axis=-1)(proc)
         
        proc = Conv2D(128,
                    (3, 3),
                    strides = 2,
                    padding = 'SAME',
                    name = ('Discriminator_%s_CONV_5' %_type),
                    kernel_initializer=self.init)(proc)
        proc = LeakyReLU(alpha=0.2)(proc)
        if self.laptop: proc = BatchNormalization(axis=-1, scale = False)(proc)
        else: proc = tfa.layers.normalizations.InstanceNormalization(axis=-1)(proc)

        proc = Flatten()(proc)
        proc = Dense(1024,
                    kernel_initializer=RandomNormal(stddev = 0.01))(proc) , #TODO add to config, this is github code on DPED (randomnormal)
        proc = LeakyReLU(alpha=0.2)(proc)
        proc = Dropout(0.2)(proc)
        proc = Dense(1,
                    kernel_initializer=self.init,
                    activation='sigmoid')(proc)
        return Model(inputs=image, outputs=proc, name=_type)

    def _get_data_shape(self):
        try:
            return self.domA.shape, self.domB.shape
        except:
            print('ERROR: Data not present')
            sys.exit()
    
    @tf.function
    def train_step(self, domainA_imgs, domainB_imgs):
        # fake = np.zeros((self.BATCH_SIZE, ) + self.imgShape).astype('float32')
        # true = np.ones((self.BATCH_SIZE, ) + self.imgShape).astype('float32')
        
        #fake = np.zeros((self.BATCH_SIZE, ) + (7, 7, 1)).astype('float32')
        #true = np.ones((self.BATCH_SIZE, ) + (7, 7, 1)).astype('float32')
        # fake_flat = tf.convert_to_tensor(np.zeros((self.BATCH_SIZE, ) + tuple([1])).astype('float32'))
        # true_flat = tf.convert_to_tensor(np.ones((self.BATCH_SIZE, ) + tuple([1])).astype('float32'))

        with tf.GradientTape() as genG_tape, tf.GradientTape() as textDisc_tape, \
             tf.GradientTape() as colorDisc_tape, tf.GradientTape() as genF_tape:
            enhanced_images = self.G(domainA_imgs, training=True)
            reverted_images = self.F(enhanced_images, training=True)
            textDisc_real_output = self.textDisc(domainB_imgs, training=True)
            textDisc_fake_output = self.textDisc(enhanced_images, training=True)

            colorDisc_real_output = self.colorDisc(domainB_imgs, training=True)
            colorDisc_fake_output = self.colorDisc(enhanced_images, training=True)
            
            
            gen_loss = content_loss_v2(domainA_imgs, reverted_images, self.inter_VGG_model)
            TV_loss = tv_loss(enhanced_images, self.imgShape)

            if self.flattenDiscriminator:
                textDisc_loss = texture_loss(tf.ones_like(textDisc_real_output), textDisc_real_output) + \
                                texture_loss(tf.zeros_like(textDisc_fake_output), textDisc_fake_output)
                colorDisc_loss = color_loss(tf.ones_like(colorDisc_real_output), colorDisc_real_output) + \
                                color_loss(tf.zeros_like(textDisc_fake_output), colorDisc_fake_output)
                # print(colorDisc_loss)    
            else:
                textDisc_loss = texture_loss(true, textDisc_real_output) + \
                                texture_loss(fake,textDisc_fake_output)

                colorDisc_loss = color_loss(true, colorDisc_real_output) + \
                                color_loss(fake, colorDisc_fake_output)

            total_gen_loss = self.contentW * gen_loss - \
                            self.textureW * textDisc_loss - \
                            self.colorW * colorDisc_loss + \
                            self.tvW * TV_loss

        
        gradients_of_G = genG_tape.gradient(total_gen_loss, self.G.trainable_variables)
        gradients_of_F = genF_tape.gradient(total_gen_loss, self.F.trainable_variables)

        self.G_optimizer.apply_gradients(zip(gradients_of_G, self.G.trainable_variables))
        self.F_optimizer.apply_gradients(zip(gradients_of_F, self.F.trainable_variables))
        gradients_of_textDisc = textDisc_tape.gradient(textDisc_loss, self.textDisc.trainable_variables)
            # print(gradients_of_textDisc)
        gradients_of_colorDisc = colorDisc_tape.gradient(colorDisc_loss, self.colorDisc.trainable_variables)
            
        self.colorDisc_optimizer.apply_gradients(zip(gradients_of_colorDisc, self.colorDisc.trainable_variables))
        self.textDisc_optimizer.apply_gradients(zip(gradients_of_textDisc, self.textDisc.trainable_variables))
        # return 0, 0, 0, 0
        # return np.array(total_gen_loss), np.array(textDisc_loss), np.array(colorDisc_loss), np.array(TV_loss), np.array(gen_loss)
        return total_gen_loss, textDisc_loss, colorDisc_loss, TV_loss, gen_loss, textDisc_fake_output, textDisc_real_output, colorDisc_fake_output, colorDisc_real_output

    @tf.function
    def train_generator_step(self, domainA_imgs, domainB_imgs):
        with tf.GradientTape() as genG_tape, tf.GradientTape() as textDisc_tape, \
             tf.GradientTape() as colorDisc_tape, tf.GradientTape() as genF_tape:

            enhanced_images = self.G(domainA_imgs, training=True)
            reverted_images = self.F(enhanced_images, training=True)

            textDisc_real_output = self.textDisc(domainB_imgs, training=False)
            textDisc_fake_output = self.textDisc(enhanced_images, training=False)

            colorDisc_real_output = self.colorDisc(domainB_imgs, training=False)
            colorDisc_fake_output = self.colorDisc(enhanced_images, training=False)
            
            # Losses
            gen_loss = content_loss_v2(domainA_imgs, reverted_images, self.inter_VGG_model)
            TV_loss = tv_loss(enhanced_images, self.imgShape)
            textDisc_loss = texture_loss(tf.ones_like(textDisc_real_output), textDisc_real_output) + \
                            texture_loss(tf.zeros_like(textDisc_fake_output), textDisc_fake_output)
            colorDisc_loss = color_loss(tf.ones_like(colorDisc_real_output), colorDisc_real_output) + \
                            color_loss(tf.zeros_like(textDisc_fake_output), colorDisc_fake_output)
            total_gen_loss = self.contentW * gen_loss - \
                            self.textureW * textDisc_loss - \
                            self.colorW * colorDisc_loss + \
                            self.tvW * TV_loss

        
        gradients_of_G = genG_tape.gradient(total_gen_loss, self.G.trainable_variables)
        gradients_of_F = genF_tape.gradient(total_gen_loss, self.F.trainable_variables)

        self.G_optimizer.apply_gradients(zip(gradients_of_G, self.G.trainable_variables))
        self.F_optimizer.apply_gradients(zip(gradients_of_F, self.F.trainable_variables))
        
        return total_gen_loss, textDisc_loss, colorDisc_loss, TV_loss, gen_loss, textDisc_fake_output, textDisc_real_output, colorDisc_fake_output, colorDisc_real_output

    
    
    @tf.function
    def train_discriminator_step(self, domainA_imgs, domainB_imgs):
        with tf.GradientTape() as genG_tape, tf.GradientTape() as textDisc_tape, \
             tf.GradientTape() as colorDisc_tape, tf.GradientTape() as genF_tape:

            enhanced_images = self.G(domainA_imgs, training=False)
            reverted_images = self.F(enhanced_images, training=False)

            textDisc_real_output = self.textDisc(domainB_imgs, training=True)
            textDisc_fake_output = self.textDisc(enhanced_images, training=True)

            colorDisc_real_output = self.colorDisc(domainB_imgs, training=True)
            colorDisc_fake_output = self.colorDisc(enhanced_images, training=True)
            
            # Losses
            gen_loss = content_loss_v2(domainA_imgs, reverted_images, self.inter_VGG_model)
            TV_loss = tv_loss(enhanced_images, self.imgShape)
            textDisc_loss = texture_loss(tf.ones_like(textDisc_real_output), textDisc_real_output) + \
                            texture_loss(tf.zeros_like(textDisc_fake_output), textDisc_fake_output)
            colorDisc_loss = color_loss(tf.ones_like(colorDisc_real_output), colorDisc_real_output) + \
                             color_loss(tf.zeros_like(textDisc_fake_output), colorDisc_fake_output)

            total_gen_loss = self.contentW * gen_loss - \
                            self.textureW * textDisc_loss - \
                            self.colorW * colorDisc_loss + \
                            self.tvW * TV_loss

        
        gradients_of_textDisc = textDisc_tape.gradient(textDisc_loss, self.textDisc.trainable_variables)
        gradients_of_colorDisc = colorDisc_tape.gradient(colorDisc_loss, self.colorDisc.trainable_variables)
            
        self.colorDisc_optimizer.apply_gradients(zip(gradients_of_colorDisc, self.colorDisc.trainable_variables))
        self.textDisc_optimizer.apply_gradients(zip(gradients_of_textDisc, self.textDisc.trainable_variables))
        
        return total_gen_loss, textDisc_loss, colorDisc_loss, TV_loss, gen_loss, textDisc_fake_output, textDisc_real_output, colorDisc_fake_output, colorDisc_real_output
    
    @tf.function
    def train_discriminator_step_flipped(self, domainA_imgs, domainB_imgs):
        with tf.GradientTape() as genG_tape, tf.GradientTape() as textDisc_tape, \
             tf.GradientTape() as colorDisc_tape, tf.GradientTape() as genF_tape:

            enhanced_images = self.G(domainA_imgs, training=False)
            reverted_images = self.F(enhanced_images, training=False)

            textDisc_real_output = self.textDisc(domainB_imgs, training=True)
            textDisc_fake_output = self.textDisc(enhanced_images, training=True)

            colorDisc_real_output = self.colorDisc(domainB_imgs, training=True)
            colorDisc_fake_output = self.colorDisc(enhanced_images, training=True)
            
            # Losses
            gen_loss = content_loss_v2(domainA_imgs, reverted_images, self.inter_VGG_model)
            TV_loss = tv_loss(enhanced_images, self.imgShape)
            textDisc_loss = texture_loss(tf.zeros_like(textDisc_real_output), textDisc_real_output) + \
                            texture_loss(tf.ones_like(textDisc_fake_output), textDisc_fake_output)
            colorDisc_loss = color_loss(tf.zeros_like(colorDisc_real_output), colorDisc_real_output) + \
                             color_loss(tf.ones_like(textDisc_fake_output), colorDisc_fake_output)

            total_gen_loss = self.contentW * gen_loss - \
                            self.textureW * textDisc_loss - \
                            self.colorW * colorDisc_loss + \
                            self.tvW * TV_loss

        
        gradients_of_textDisc = textDisc_tape.gradient(textDisc_loss, self.textDisc.trainable_variables)
        gradients_of_colorDisc = colorDisc_tape.gradient(colorDisc_loss, self.colorDisc.trainable_variables)
            
        self.colorDisc_optimizer.apply_gradients(zip(gradients_of_colorDisc, self.colorDisc.trainable_variables))
        self.textDisc_optimizer.apply_gradients(zip(gradients_of_textDisc, self.textDisc.trainable_variables))
        
        return total_gen_loss, textDisc_loss, colorDisc_loss, TV_loss, gen_loss, textDisc_fake_output, textDisc_real_output, colorDisc_fake_output, colorDisc_real_output

    def train(self, datasetA, datasetB, epochs):
        chckpts = np.array(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) * epochs).astype('int32')
        print('Saving at epochs:', chckpts)
        generate_and_save_images(self.G,
                                    1,
                                    self.testImg_orig, ckpt_folder = self.save_ckpt_dir,  test=True, type_ = '_orig')
        generate_and_save_images(self.G,
                                    1,
                                    self.testImg_noise, ckpt_folder = self.save_ckpt_dir,  test=True, type_ = '_noise')
        for epoch in range(epochs):
            if epoch in chckpts:
                print('Saving model at epoch{} in file {}: '.format(epoch, self.save_checkpoint_prefix))
                self.checkpoint.save(file_prefix = self.save_checkpoint_prefix)
                 # Produce images for the GIF as we go
            display.clear_output(wait=True)
            newImg = generate_and_save_images(self.G,
                                                epoch + 1,
                                                self.testImg_patches,
                                                patchSize = self.patchSize,
                                                kSize = 9,
                                                ckpt_folder = self.save_ckpt_dir)
            start = time.time()
            # Train discriminator for self.discrimEPOCHS 
            if self.laptop: len_ = 2
            else: len_ = 100000
            for i in range(self.genEpochs):
                print('Training generator for minibatch {} out of {}'.format(i + 1, self.genEpochs))
                for imageA_batch, imageB_batch, _ in zip(datasetA, datasetB, range(len_)):
                    gen_loss, text_loss, col_loss, TV_loss, cont_loss, \
                    textDisc_fake_output_np, textDisc_real_output_np, colDisc_fake_output_np, colDisc_true_output_np  = self.train_generator_step(imageA_batch, imageB_batch)
                print("Generator loss:", gen_loss.numpy())
                print("Texture disc loss:", text_loss.numpy())
                print("Color disc loss:", col_loss.numpy())
                print("TV loss:", TV_loss.numpy())
                print("Content loss: ", cont_loss.numpy())
                print("Texture discrim output w/ fake images: ", textDisc_fake_output_np.numpy()[:, :, 0])
                print("Texture discrim output w/ real images: ", textDisc_real_output_np.numpy()[:, :, 0])
                print("Color discrim output w/ fake images: ", textDisc_fake_output_np.numpy()[:, :, 0])
                print("Color discrim output w/ real images: ", textDisc_real_output_np.numpy()[:, :, 0])
                print('------------')
                # Saving images of metrics
                imgName = 'epoch_' + str(epoch * (self.genEpochs + self.discrimEpochs) + i) 
                self.contentLoss_hist.append(cont_loss.numpy())
                self.textLoss_hist.append(text_loss.numpy())
                self.colorLoss_hist.append(col_loss.numpy())
                self.tvLoss_hist.append(TV_loss.numpy())
                self.totalLoss_hist.append(gen_loss.numpy())
                save_metrics(self.metricsFolder, imgName, 'Content_Loss', self.colorLoss_hist)
                save_metrics(self.metricsFolder, imgName, 'Texture_Loss', self.textLoss_hist)
                save_metrics(self.metricsFolder, imgName, 'Color_Loss', self.colorLoss_hist)
                save_metrics(self.metricsFolder, imgName, 'TV_Loss', self.tvLoss_hist)
                save_metrics(self.metricsFolder, imgName, 'Total_Loss', self.totalLoss_hist)


            for i in range(self.discrimEpochs):
                print('Training discriminator for minibatch {} out of {}'.format(i + 1, self.discrimEpochs))
                for imageA_batch, imageB_batch, batchIndex in zip(datasetA, datasetB, range(len_)):
                    imageA_batch = imageA_batch + tf.random.normal(imageA_batch.shape, stddev = self.discrimNoiseSTDDEV, dtype = 'double')
                    if self.labelFlippingPeriod is not None:
                        if (batchIndex + 1) % self.labelFlippingPeriod == 0:
                            gen_loss, text_loss, col_loss, TV_loss, cont_loss, \
                            textDisc_fake_output_np, textDisc_real_output_np, colDisc_fake_output_np, colDisc_true_output_np  = self.train_discriminator_step_flipped(imageA_batch, imageB_batch)
                        else:
                            gen_loss, text_loss, col_loss, TV_loss, cont_loss, \
                            textDisc_fake_output_np, textDisc_real_output_np, colDisc_fake_output_np, colDisc_true_output_np  = self.train_discriminator_step(imageA_batch, imageB_batch)
                    else:
                        gen_loss, text_loss, col_loss, TV_loss, cont_loss, \
                        textDisc_fake_output_np, textDisc_real_output_np, colDisc_fake_output_np, colDisc_true_output_np  = self.train_discriminator_step(imageA_batch, imageB_batch)
                        

                print("Generator loss:", gen_loss.numpy())
                print("Texture disc loss:", text_loss.numpy())
                print("Color disc loss:", col_loss.numpy())
                print("TV loss:", TV_loss.numpy())
                print("Content loss: ", cont_loss.numpy())
                print("Texture discrim output w/ fake images: ", textDisc_fake_output_np.numpy()[:, :, 0])
                print("Texture discrim output w/ real images: ", textDisc_real_output_np.numpy()[:, :, 0])
                print("Color discrim output w/ fake images: ", textDisc_fake_output_np.numpy()[:, :, 0])
                print("Color discrim output w/ real images: ", textDisc_real_output_np.numpy()[:, :, 0])
                print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
                print('------------')
                # Saving images of metrics
                imgName = 'epoch_' + str(epoch * (self.genEpochs + self.discrimEpochs) + i) 
                self.contentLoss_hist.append(cont_loss.numpy())
                self.textLoss_hist.append(text_loss.numpy())
                self.colorLoss_hist.append(col_loss.numpy())
                self.tvLoss_hist.append(TV_loss.numpy())
                self.totalLoss_hist.append(gen_loss.numpy())
                save_metrics(self.metricsFolder, imgName, 'Content_Loss', self.colorLoss_hist)
                save_metrics(self.metricsFolder, imgName, 'Texture_Loss', self.textLoss_hist)
                save_metrics(self.metricsFolder, imgName, 'Color_Loss', self.colorLoss_hist)
                save_metrics(self.metricsFolder, imgName, 'TV_Loss', self.tvLoss_hist)
                save_metrics(self.metricsFolder, imgName, 'Total_Loss', self.totalLoss_hist)
                

            print('Logging data')
            print('------------')
            with open(self.logFilePath, 'a') as logFile:
                logFile.write('Time for epoch {} is {} sec'.format(str(epoch + 1), str(time.time()-start)) + '\n')
                logFile.write("Generator loss:" + str(gen_loss.numpy()) + '\n')
                logFile.write("Texture disc loss:" + str(text_loss.numpy()) + '\n')
                logFile.write("Color disc loss:" + str(col_loss.numpy()) + '\n')
                logFile.write("TV loss:" + str(TV_loss.numpy()) + '\n')
                logFile.write("Content loss: " + str(cont_loss.numpy()) + '\n')
                logFile.write("Texture discrim output w/ fake images: " + str(textDisc_fake_output_np.numpy()))
                # logFile.write('Min, Max pixel values in test input patches: ' + str(max(np.unique(testImgPatches))) + ' ' + str(min(np.unique(testImgPatches))))
                min_, max_ = min(np.unique(newImg)), max(np.unique(newImg))
                logFile.write('Max and min pixel values in predictions: ' + str(min_) + ', ' + str(max_) + '\n')
                logFile.write('----------------------------------\n' + '\n' + '\n')


        # Generate after the final epoch
        display.clear_output(wait=True)
        newImg = generate_and_save_images(self.G,
                                    epoch + 2,
                                    self.testImg_patches,
                                    patchSize = self.patchSize,
                                    kSize = 9,
                                    ckpt_folder = self.save_ckpt_dir)

        self.checkpoint.save(file_prefix = self.save_checkpoint_prefix)

if __name__ == "__main__":
    
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if len(gpus) == 0:
        configPath = './config_files/laptop-wespe.config'  # CPU laptop
        model = WESPE(configPath,  trainMode = False, laptop = True)
    else:
        configPath = './config_files/wespe.config'  # GPU server
        model = WESPE(configPath,  trainMode = False, laptop = False)
