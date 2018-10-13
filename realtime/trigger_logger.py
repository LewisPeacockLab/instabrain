import pygame
import os, sys, time
import numpy as np

def check_for_scanner_trigger(log_file, log_file_time, rep_count):
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_5:
                write_log(log_file, log_file_time, 'trigger', rep_count)
                rep_count+=1
            elif event.key == pygame.K_ESCAPE:
                sys.exit()
    return rep_count

def write_log_header(log_file):
    log_file.write('time,event,tr\n')

def write_log(log_file, start_time, event_name, count):
    log_file.write(str(time.time()-start_time)+','+event_name+','+str(count)+'\n')

if __name__ == '__main__':
    pygame.init()
    pygame.display.set_mode((1,1))
    rep_count = 0
    log_file_time = int(np.floor(time.time()))
    log_file_name = os.getcwd()+'/log/'+str(log_file_time)+'_trigger.log'
    log_file = open(os.path.normpath(log_file_name),'w')
    write_log_header(log_file)
    start_time = 1.539e9 #hacky
    while True:
        rep_count = check_for_scanner_trigger(log_file, start_time, rep_count)
