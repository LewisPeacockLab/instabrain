import shutil
from shutil import copyfile
from shutil import copy2
from os import listdir
from os.path import isfile, join, exists
import time
import os
import subprocess

time_delay = 2/28
num_slices = 28
num_reps = 160
copy_delay = 0.01

def copyFiles(old_dir, new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    os.system('rm '+new_dir+'/*')

    for rep in range(num_reps):
        for slc in range(num_slices):
            start_time = time.time()
            img_file = 'SB-R'+str(rep+1).zfill(4)+'-E1-S'+str(slc+1).zfill(3)+'.imgdat'
            tmp_file = img_file+'.tmp'
            shutil.copy(old_dir+'/'+img_file, new_dir+'/'+tmp_file)
            time.sleep(copy_delay)
            shutil.move(new_dir+'/'+tmp_file, new_dir+'/'+img_file)
            end_time = time.time()
            time.sleep(time_delay-(start_time-end_time))

if __name__ == "__main__":
    copyFiles("ref", "dump")
