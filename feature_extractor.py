from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.models import Model, load_model
from keras.layers import Input
from image_processor import inception_image_processor , resnet_image_processor
import numpy as np
import tensorflow as tf
graph = tf.get_default_graph()
class Inception_Features():
    def __init__(self,denselayer=True):
        inception_model = InceptionV3(weights='imagenet',include_top=denselayer)
        self.model = Model(
            inputs=inception_model.input,
            outputs=inception_model.layers[-2].output
        )
        self.input_shape=(299,299,3)

    def get_features(self,imgpath):
        with graph.as_default():
            input_vec = inception_image_processor(imgpath,self.input_shape)
            features = self.model.predict(input_vec)
            return features.squeeze()

class Resnet_Features():
    def __init__(self,denselayer=True):
        resnet_model = ResNet50(weights='imagenet',include_top=denselayer)
        self.model = Model(
            inputs=resnet_model.input,
            outputs=resnet_model.layers[-2].output
        )
        self.input_shape=(299,299,3)

    def get_features(self,imgpath):
        input_vec = resnet_image_processor(imgpath,self.input_shape)
        features = self.model.predict(input_vec)
        return features.squeeze()
