from create_images import extract_images, sample_x_images,destination_directory
import sys
from split_list import make_split,getclasses,category_to_digit,train_folder,test_folder,digit_to_category
from feature_extractor import Inception_Features, Resnet_Features
from keras.utils import to_categorical
from utils import mac_remove_file
import csv
import numpy as np
import threading
import random
import tensorflow as tf
import sys
class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

class Dataset():
    all_classes=None
    class_dict=None
    sampling_rate=40
    trainlength= None
    testlength= None
    trainfile="trainlist-00.csv"
    testfile="testlist-00.csv"
    trainlist=None
    testlist=None
    def set_sampling_rate(rate):
        self.sampling_rate=rate
    def __init__(self,isinit=False,transfer="I"):
        if isinit == True:
            self.trainfile,self.testfile=make_split(80,20,"00")
            extract_images(self.trainfile,self.testfile)


        self.all_classes = getclasses()
        self.class_num = len(self.all_classes)
        self.class_dict = digit_to_category()
        if transfer == "I":
            self.feature_class = Inception_Features()
        else:
            self.feature_class = Resnet_Features()

    def make_path_lists(self):
        self.trainlist=list()
        with open(self.trainfile,"r") as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                videoname,dest_dir=destination_directory(row,train_folder)
                self.trainlist.append([dest_dir,int(row[1])])
        self.trainlength=len(self.trainlist)
        self.testlist=list()
        with open(self.testfile,"r") as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                videoname,dest_dir=destination_directory(row,test_folder)
                self.testlist.append([dest_dir,int(row[1])])
        self.testlength=len(self.testlist)
    @threadsafe_generator
    def train_data_generator(self,batchsize):
        imagelist=self.trainlist
        while True:
            X=[]
            y=[]
            for i in range(batchsize):
                filename = random.choice(imagelist)
                # print("Rofl:",filename)
                all_images = sample_x_images(filename[0],self.sampling_rate)
                sequence=self.build_sequence(all_images)
                X.append(sequence)
                y.append(to_categorical(filename[1],self.class_num))
            yield np.array(X),np.array(y)
    @threadsafe_generator
    def test_data_generator(self,batchsize):
        imagelist=self.testlist
        while True:
            X=[]
            y=[]
            for i in range(batchsize):
                filename = random.choice(imagelist)
                # print("Rofl:",filename)
                all_images = sample_x_images(filename[0],self.sampling_rate)
                sequence=self.build_sequence(all_images)
                X.append(sequence)
                y.append(to_categorical(filename[1],self.class_num))
            yield np.array(X),np.array(y)
    def load_all_in_memory(self,listname):
        X=[]
        y=[]
        for each_file in listname:
            imagelist = sample_x_images(each_file[0],self.sampling_rate)
            sequence=self.build_sequence(imagelist)
            X.append(sequence)
            y.append(to_categorical(each_file[1],self.class_num).squeeze())
        return np.array(X), np.array(y)

    def build_sequence(self,imagelist):
        sequence=[self.feature_class.get_features(i) for i in imagelist]
        # print(sequence.shape)
        # sys.exit(0)
        return sequence


if __name__ == "__main__":
    mac_remove_file()
    length = len(sys.argv)
    if length>2 and sys.argv[1] == "init":
        myclass = Dataset(True)
    else:
        myclass = Dataset()
        myclass.make_path_lists()
        myclass.load_all_in_memory(myclass.trainlist)
        print(myclass.all_classes)
        print(myclass.class_dict)
