import shutil
from shutil import copyfile
from shutil import copy2
from os import listdir
from os.path import isfile, join, exists
import time
import os

time_delay = 2/28

def copyFiles(old_dir, new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    all_images = [f for f in listdir(old_dir) if isfile(join(old_dir, f))]
    for i in range(0, len(all_images)):
        start = time.time()
        img_file = old_dir + "/" + all_images[i]
        imgdat_tmp_dir = new_dir + "/" + all_images[i] + ".tmp"
        shutil.copy2(img_file, imgdat_tmp_dir)
        final_image_name = imgdat_tmp_dir[:-4]
        os.rename(imgdat_tmp_dir, final_image_name)
        end = time.time()
        time.sleep(time_delay-(start-end))
        #print(end-start)
                

if __name__ == "__main__":
    copyFiles("ref", "dump")