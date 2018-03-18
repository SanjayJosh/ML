from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from lstm_model import lstm_model
from datagen import Dataset
import time
import os.path
from utils import mac_remove_file
import tensorflow as tf
from feature_extractor import graph
def train():
    with graph.as_default():
        datamodel = Dataset()
        datamodel.make_path_lists()
        nb_categories=datamodel.class_num
        print(nb_categories)
        tb = TensorBoard(log_dir=os.path.join('logs'))
        feature_dim= 2048
        sampling_rate=40
        batchsize=6
        epochs= 2
        steps_per_epoch= datamodel.trainlength//batchsize
        steps_per_epoch_test = datamodel.testlength//batchsize
        dnn_model = lstm_model(nb_categories,sampling_rate,feature_dim)
        model= dnn_model.getmodel()
        train_generator= datamodel.train_data_generator(batchsize)
        test_generator= datamodel.test_data_generator( batchsize)
        model.fit_generator(generator=train_generator,steps_per_epoch=steps_per_epoch,epochs=2,verbose=1,validation_data=test_generator,validation_steps=steps_per_epoch_test,callbacks=[tb])
        # X,y = datamodel.data_generator(datamodel.trainlist,batchsize)
        # print(X.shape)
        # print(y.shape)
train();
