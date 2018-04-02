import sys
import subprocess
import os
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from create_images import extract_images, sample_x_images,destination_directory,remove_and_create
from lstm_model import lstm_model
from datagen import Dataset
nb_categories=5
feature_dim= 2048
sampling_rate=40
dest_dir="Validation_Data"
dnn_model = lstm_model(nb_categories,sampling_rate,feature_dim)
model= dnn_model.getmodel()
model.load_weights('/Users/s0j00os/Documents/Proj/lstmodel.hdf5')
FNULL = open(os.devnull, 'w')
datamodel = Dataset()
videoname= sys.argv[1]
remove_and_create(dest_dir)
subprocess.call(['ffmpeg','-i',videoname,os.path.join(dest_dir,'image-%04d.jpg')],stdout=FNULL, stderr=subprocess.STDOUT)
imagelist=sample_x_images(dest_dir,sampling_rate)
cnn_sequence=datamodel.build_sequence(imagelist)

y=model.predict_classes(np.array([np.array(cnn_sequence)]))
print("The prediction is :",datamodel.class_dict[y[0]])
