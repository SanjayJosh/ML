from keras.preprocessing.image import img_to_array, load_img
from keras.applications.inception_v3 import preprocess_input as inception_preprocess
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
import numpy as np

def inception_image_processor(imgpath,target_shape):
    img = load_img(imgpath,target_size=target_shape)
    vec = img_to_array(img)
    vec = np.expand_dims(vec, axis=0)
    vec = inception_preprocess(vec)
    return vec
def resnet_image_processor(imgpath,target_shape):
    img = load_img(imgpath,target_size=target_shape)
    vec = img_to_array(img)
    vec = np.expand_dims(vec, axis=0)
    vec = resnet_preprocess(vec)
    return vec
