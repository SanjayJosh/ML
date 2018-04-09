from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from lstm_model import lstm_model
from datagen import Dataset
import time
import os.path
from utils import mac_remove_file
import tensorflow as tf
import sys
from feature_extractor import graph
def train():
    with graph.as_default():
        mac_remove_file()
        starttime=time.time()
        datamodel = Dataset(False)
        print("Done with the file-creation")
        datamodel.make_path_lists()
        print("Done with the path-creation")
        print("Time for path-creation is:",time.time() - starttime)
        nb_categories=datamodel.class_num
        print(nb_categories)
        tb = TensorBoard(log_dir=os.path.join('logs'))
        checkpoint= ModelCheckpoint(filepath=os.path.join('checkpoints','lstm-'+'.{epoch:03d}-{val_loss:.3f}.hdf5'),verbose=1,save_best_only=True)
        early_stopper = EarlyStopping(patience=15)
        feature_dim= 2048
        sampling_rate=40
        batchsize=6
        epochs= 100
        steps_per_epoch= datamodel.trainlength//batchsize
        steps_per_epoch_test = datamodel.testlength//batchsize
        dnn_model = lstm_model(nb_categories,sampling_rate,feature_dim)
        model= dnn_model.getmodel();sys.exit(0)
        # train_generator= datamodel.train_data_generator(batchsize)
        # test_generator= datamodel.test_data_generator( batchsize)
        starttime=time.time()
        # model.fit_generator(generator=train_generator,steps_per_epoch=steps_per_epoch,epochs=epochs,verbose=1,validation_data=test_generator,validation_steps=steps_per_epoch_test,callbacks=[tb])
        X,y = datamodel.load_all_in_memory(datamodel.trainlist)
        X_test,y_test = datamodel.load_all_in_memory(datamodel.testlist)
        print("Time for loading into memory is:",time.time() - starttime)

        starttime=time.time()
        model.fit(X,y,validation_data=(X_test, y_test),batch_size=batchsize,verbose=1,epochs=epochs,callbacks=[tb,checkpoint,early_stopper])
        print("Training-Validation time is",time.time() - starttime)
        # X,y = datamodel.data_generator(datamodel.trainlist,batchsize)
        # print(X.shape)
        # print(y.shape)
train();
