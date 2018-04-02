import subprocess
import platform
from keras.utils import to_categorical
ignore_file=".DS_Store"
def mac_remove_file():
    subprocess.call(('find . -name '+ignore_file+'  -delete').split())
    print("DS Store deleted.")
