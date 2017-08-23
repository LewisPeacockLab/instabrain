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
            shutil.move(new_dir+'/'+tmp_file, new_dir+'/'+img_file)
            end_time = time.time()
            time.sleep(time_delay-(start_time-end_time))
    
    # all_images = [f for f in listdir(old_dir) if isfile(join(old_dir, f))]
    # print('copying files...')
    # for i in range(0, len(all_images)):
    #     img_file = old_dir + "/" + all_images[i]
    #     imgdat_tmp_dir = new_dir + "/" + all_images[i] + ".tmp"
    #     shutil.copy2(img_file, imgdat_tmp_dir)
    # print('done copying files!')

    # print('renaming files...')
    # for i in range(0, len(all_images)):
    #     start = time.time()
    #     # img_file = old_dir + "/" + all_images[i]
    #     imgdat_tmp_dir = new_dir + "/" + all_images[i] + ".tmp"
    #     # shutil.copy2(img_file, imgdat_tmp_dir)
    #     final_image_name = imgdat_tmp_dir[:-4]
    #     # shutil.move(imgdat_tmp_dir, final_image_name)
    #     os.system('mv '+imgdat_tmp_dir+' '+final_image_name)
    #     # subprocess.Popen(['mv', imgdat_tmp_dir, final_image_name])
    #     # subprocess.Popen('mv '+imgdat_tmp_dir+' '+final_image_name,shell=True)
    #     end = time.time()
    #     time.sleep(time_delay-(start-end))
    #     #print(end-start)
    # print('done renaming files!')
                

if __name__ == "__main__":
    copyFiles("ref", "dump")
