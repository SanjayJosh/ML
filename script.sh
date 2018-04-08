mkdir Proj
cd Proj
git clone https://github.com/SanjayJosh/ML.git
sudo apt-get update
sudo apt-get install unrar
git checkout ami
cd ML
wget http://crcv.ucf.edu/data/UCF101/UCF101.rar
unrar x UCF101.rar
mv UCF-101/ Data/
mkdir Clean_Data
source activate tensorflow_p36
conda uninstall PIL
conda install pillow
nohup python train_large.py &
