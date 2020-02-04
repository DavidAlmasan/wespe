import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate
# from skimage.measure import compare_ssim
import matplotlib.pyplot as plt
from tensorflow.keras.losses import binary_crossentropy
import time
import numpy as np
from tensorflow.keras.models import Sequential, Model
import cv2 as cv
# from tensorflow.python.framework import ops
# ops.reset_default_graph()

def content_loss(y_true, y_pred, shape_):
    # print(np.unique(y_true))
    # print(np.unique(y_pred))
    # time.sleep(10)
    batch_size = list(y_true.shape)[0]
    y_true = K.concatenate([y_true, y_true, y_true], axis = -1)
    y_pred = K.concatenate([y_pred, y_pred, y_pred], axis = -1)

    input_tensor = K.concatenate([y_true, y_pred], axis=0)
    input_shape = input_tensor.numpy()[0].shape
    # print(input_tensor.get_shape().as_list())
    # time.sleep(10)
    img_input = Input(shape=input_shape)
    # input_tensor = Concatenate()([img_input, img_input, img_input])

    
    model = VGG19(input_tensor=img_input, weights='imagenet', include_top=False)
    # model = VGG19(input_shape=input_shape, weights='imagenet', include_top=False)
    layer_name="block2_conv2"
    inter_VGG_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    inter_VGG_model.trainable = False
    for l in inter_VGG_model.layers:
        l.trainable = False

    inter_VGG_model.compile(optimizer='rmsprop', loss='mse')


    

    # model.compile(optimizer='rmsprop', loss='mse')
    
    # print(model.summary())
    
    # outputs_dict = dict([(layer.name, layer.output) for layer in inter_VGG_model.layers])
    # layer_features = outputs_dict["block2_conv2"]
    y_true_features = inter_VGG_model(input_tensor.numpy())[:batch_size, :, :, :]
    y_pred_features = inter_VGG_model(input_tensor.numpy())[batch_size:, :, :, :]
    # print(K.mean(K.square(y_true_features - y_pred_features)).get_shape().as_list())
    # time.sleep(10)
    return tf.reduce_mean((y_true_features - y_pred_features ** 2))
    return K.mean(K.square(y_true_features - y_pred_features)) 

def content_loss_v2(y_true, y_pred, model):
    # return tf.Variable([0], tf.int32)
    # print(np.unique(y_true))
    # print(np.unique(y_pred))
    # time.sleep(10)
    # batch_size = list(y_true.shape)[0]
    # y_true = K.concatenate([y_true, y_true, y_true], axis = -1)
    # y_pred = K.concatenate([y_pred, y_pred, y_pred], axis = -1)
    # gpus = tf.config.experimental.list_physical_devices("GPU")
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    
    real_content = model(y_true)
    fake_content = model(y_pred)
    return K.mean(K.square([real_content - fake_content]))

def tv_loss(imgs, imgShape):
    (dX, dY) = tf.image.image_gradients(imgs)
    scalar = 1. /(imgShape[0] * imgShape[1])
    # return K.mean(K.square(dY))
    return  K.mean(tf.math.scalar_mul(x = K.square(tf.math.add(dX, dY)),  scalar = scalar))

def color_loss(y_true, y_pred):
    return K.mean(binary_crossentropy(y_true, y_pred, from_logits = False))

def texture_loss(y_true, y_pred):
    return K.mean(binary_crossentropy(y_true, y_pred, from_logits = False))

def total_loss_agg(domA_orig, reverted_imgs, enhanced_imgs, model, imgShape):
    real_content = model(domA_orig)
    fake_content = model(reverted_imgs)
    cont_loss = K.mean(K.square([real_content - fake_content]))
 
    # (dX, dY) = tf.image.image_gradients(enhanced_imgs)
    # scalar = 1. /(imgShape[0] * imgShape[1])
    return cont_loss
    # return  K.mean(tf.math.scalar_mul(x = K.square(tf.math.add(dX, dY)),  scalar = scalar))
if  __name__ == "__main__":
    img = np.ones((10, 10))
    img[5, 5] = 100
    img = np.expand_dims((img), axis = -1)
    img = np.array([img, img, img, img, img, img, img])
    print(img.shape)
    a = tv_loss(tf.convert_to_tensor(img), (10, 10))
    print(np.array(a))
    # print(np.unique(a))
    # plt.imshow(a)
    # plt.show()
