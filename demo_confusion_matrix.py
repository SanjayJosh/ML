import sys
import subprocess
import os
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from create_images import extract_images, sample_x_images,destination_directory,remove_and_create
from utils import mac_remove_file
from lstm_model import lstm_model
from datagen import Dataset
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import confusion_matrix
mac_remove_file()
datamodel = Dataset()
datamodel.make_path_lists()
nb_categories=datamodel.class_num
feature_dim= 2048
sampling_rate=30
def get_classes(arr):
    x,y=arr.shape
    ret_list=list()
    for i in range(x):
        temp_arr=arr[i]
        for j in range(y):
            if int(temp_arr[j]) == 1:
                ret_list.append(j)
                break

    return ret_list

dnn_model = lstm_model(nb_categories,sampling_rate,feature_dim)
model= dnn_model.getmodel()
model.load_weights('checkpoints/lstm-best.hdf5')
X_test,y_test = datamodel.load_all_in_memory(datamodel.testlist)
# y_true_list=pd.Series(get_classes(y_test))
y_true_list=pd.Series([datamodel.class_dict[i] for i in get_classes(y_test)])
y_pred=pd.Series([datamodel.class_dict[i] for i in model.predict_classes(X_test)])
print(y_true_list)
print(y_pred)
print("Accuracy is:",accuracy_score(y_true_list,y_pred)*100,"%")
# print(confusion_matrix(y_true_list,y_pred,labels=[ "disco", "jazz" ,"pop","reggae","rock"]))
print(pd.crosstab(y_true_list,y_pred))
