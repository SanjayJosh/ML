from keras.preprocessing.image import img_to_array, load_img
from keras.applications.inception_v3 import preprocess_input as inception_preprocess
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.models import Model, load_model
from keras.layers import Input
from keras.applications.imagenet_utils import decode_predictions
from image_processor import inception_image_processor , resnet_image_processor
import numpy as np

imgpath='Clean_Data/Train/Archery/v_Archery_g01_c01/image-0001.jpg'


model=InceptionV3(weights='imagenet',include_top=True)
img = load_img(imgpath,target_size=(299,299,3))
vec = img_to_array(img)
vec = np.expand_dims(vec, axis=0)
vec = inception_preprocess(vec)
print(vec.shape)
preds=model.predict(vec)
print(decode_predictions(preds, top=3)[0])
