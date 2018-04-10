from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from lstm_model import lstm_model
from datagen import Dataset
from create_images import sample_x_images
import time
import os
from utils import mac_remove_file
import tensorflow as tf
import multiprocessing as mp
import numpy as np
from feature_extractor import graph
from feature_extractor import Inception_Features
from keras.applications.inception_v3 import InceptionV3
from image_processor import inception_image_processor
from keras.models import Model, load_model
from keras.utils import to_categorical
def global_save_array(each_file):
    print("Woah here eh")
    imagelist = sample_x_images(each_file[0],datamodel.sampling_rate)
    X=[feature_model.predict(inception_image_processor(i,input_shape)) for i in imagelist]
    y=to_categorical(each_file[1],datamodel.class_num).squeeze()
    np.save(os.path.join(each_file[0],'X'),X)
    np.save(os.path.join(each_file[0],'y'),y)
    print("Done eh")
    return 1;
def myprint(file):
    return file[0]
def global_save_in_disk_parallel(listname):
    print("Much waw")
    pool = mp.Pool(processes=5)
    results = pool.map(global_save_array,listname)
    print(results)
def train():
    with graph.as_default():
        is_multiprocessing=True
        mac_remove_file()
        starttime=time.time()
        global datamodel
        global feature_model
        global input_shape
        datamodel = Dataset(False)
        input_shape=(299,299,3)
        inception_model = InceptionV3(weights='imagenet',include_top=True)
        feature_model = Model(
            inputs=inception_model.input,
            outputs=inception_model.layers[-2].output
        )
        feature_model._make_predict_function()
        print("Done with the file-creation")
        datamodel.make_path_lists()
        global_save_in_disk_parallel(datamodel.trainlist)
        global_save_in_disk_parallel(datamodel.testlist)
        print("Done with the path-creation")
        print("Time for path-creation is:",time.time() - starttime)
        nb_categories=datamodel.class_num
        print(nb_categories)
        tb = TensorBoard(log_dir=os.path.join('logs'))
        checkpoint= ModelCheckpoint(filepath=os.path.join('checkpoints','lstm-'+'.{epoch:03d}-{val_loss:.3f}.hdf5'),verbose=1,save_best_only=True)
        early_stopper = EarlyStopping(patience=15)
        feature_dim= 2048
        sampling_rate=30
        batchsize=40
        epochs= 1000
        steps_per_epoch= datamodel.trainlength//batchsize
        steps_per_epoch_test = datamodel.testlength//batchsize
        dnn_model = lstm_model(nb_categories,sampling_rate,feature_dim)
        model= dnn_model.getmodel()
        #train_generator= datamodel.train_data_generator(batchsize)
        #test_generator= datamodel.test_data_generator(batchsize)
        starttime=time.time()
        #X_test,y_test = datamodel.load_all_in_memory(datamodel.testlist)
        #model.fit_generator(generator=train_generator,steps_per_epoch=steps_per_epoch,epochs=epochs,verbose=1,validation_data=(X_test, y_test),workers=4,validation_steps=steps_per_epoch_test,callbacks=[tb,checkpoint,early_stopper])
        #model.fit_generator(generator=train_generator,steps_per_epoch=steps_per_epoch,epochs=epochs,verbose=1,validation_data=(X_test, y_test),workers=4,callbacks=[tb,checkpoint,early_stopper])
        #X,y = datamodel.load_all_in_memory(datamodel.trainlist)
        #print("Time for loading into memory is:",time.time() - starttime)

        #starttime=time.time()
        #model.fit(X,y,validation_data=(X_test, y_test),batch_size=batchsize,verbose=1,epochs=epochs,callbacks=[tb,checkpoint,early_stopper])
        print("Training-Validation time is",time.time() - starttime)
        # X,y = datamodel.data_generator(datamodel.trainlist,batchsize)
        # print(X.shape)
        # print(y.shape)
if __name__ == "__main__":
    mp.freeze_support()
    train();
