import sys
import subprocess
import os
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from create_images import extract_images, sample_x_images,destination_directory,remove_and_create
from utils import mac_remove_file
from lstm_model import lstm_model
from datagen import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import confusion_matrix
mac_remove_file()
datamodel = Dataset()
datamodel.make_path_lists()
nb_categories=datamodel.class_num
feature_dim= 2048
sampling_rate=30
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
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
# y_true_list=[datamodel.class_dict[i] for i in get_classes(y_test)]
# y_pred=[datamodel.class_dict[i] for i in model.predict_classes(X_test)]
y_true_list=[datamodel.class_dict[i] for i in get_classes(y_test)]
y_pred=[datamodel.class_dict[i] for i in model.predict_classes(X_test)]
# print(y_true_list)
# print(y_pred)
print("Y true:",len(y_true_list))
print("Y pred:",len(y_pred))
cnf_matrix = confusion_matrix(y_test, y_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix,datamodel.all_classes)
plt.savefig('final.png')
print("Accuracy is:",accuracy_score(y_true_list,y_pred)*100,"%")
# print(confusion_matrix(y_true_list,y_pred,labels=[ "disco", "jazz" ,"pop","reggae","rock"]))
# print(pd.crosstab(y_true_list,y_pred))
