import csv
import glob
import os
import split_list
import sys
import subprocess
import shutil
import math
from split_list import make_split,dir_name,train_folder,test_folder,getclasses,clean_data
FNULL = open(os.devnull, 'w')
def get_sep_index():
    sep_index= len(dir_name) + 1
    return sep_index

def remove_and_create(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)

def class_directories(path):
    for aclass in getclasses():
        os.mkdir(os.path.join(path,aclass))

def init_directories():
    train_path=os.path.join(clean_data,train_folder)
    test_path=os.path.join(clean_data,test_folder)
    remove_and_create(train_path)
    remove_and_create(test_path)
    class_directories(train_path)
    class_directories(test_path)

def destination_directory(row,split):
    sep_index=get_sep_index()
    videoname=row[0]
    filename=os.path.splitext(videoname)[0]
    dest_dir=os.path.join(clean_data,split,filename[sep_index:])
    return videoname,dest_dir

def sample_x_images(dest_dir,sampling_rate):
    # print("Lmao::",dest_dir)
    imgfiles=sorted(os.listdir(dest_dir))
    # print(imgfiles)
    length=len(imgfiles)

    skip=length//sampling_rate

    my_indices = range(0,length,skip)[:sampling_rate]
    # print(dest_dir,len(range(0,length,skip)))
    return [os.path.join(dest_dir,imgfiles[i]) for i in my_indices]

def extract_one(row,split):
    videoname,dest_dir=destination_directory(row,split)
    os.mkdir(dest_dir)
    subprocess.call(['ffmpeg','-i',videoname,os.path.join(dest_dir,'image-%04d.jpg')],stdout=FNULL, stderr=subprocess.STDOUT)


def extract_images(trainfile,testfile):
    #print(os.path.isfile(testfile) and os.path.isfile(trainfile))
    init_directories()
    with open(trainfile,"r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            extract_one(row,train_folder)

    with open(testfile,"r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            extract_one(row,test_folder)








if __name__ == "__main__":
    length = len(sys.argv)
    global testfile
    global trainfile

    if length >= 2 :
        if sys.argv[1] == "init" :
            make_split(6,3,"00")
            extract_images()
        else :
            trainfile="trainlist-"+sys.argv[1]+".csv"
            testfile="testlist-"+sys.argv[1]+".csv"
            if os.path.isfile(testfile) and os.path.isfile(trainfile) :
                extract_images()

            else :
                print("The version you specified doesn't exist!")
    else:
        trainfile="trainlist-"+"00"+".csv"
        testfile="testlist-"+"00"+".csv"
        if not (os.path.isfile(testfile) and os.path.isfile(trainfile)) :
            make_split(6,3,"00")
        extract_images()

