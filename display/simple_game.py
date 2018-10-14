import pygame
from pygame.gfxdraw import aacircle
import os, sys, time
import numpy as np
import multiprocessing as mp
import requests as r
import game_server as gs

class neurofeedbackGame(object):

    def __init__(self):
        # initialize pygame and graphics
        pygame.init()
        self.clock = pygame.time.Clock()
        self.FRAME_RATE = 60
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 800, 800
        self.CIRCLE_POS = 0.5*self.SCREEN_WIDTH, 0.5*self.SCREEN_HEIGHT
        self.MAX_CIRCLE_DIAMETER = 0.8*min(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.screen = pygame.display.set_mode(
            (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.BG_COLOR = 40,40,40
        self.CIRCLE_COLOR = 60,150,60

        # initialize logic
        self.BASELINE_TRS = 5
        self.NUM_CLASSES = 4
        self.feedback_score = 0.1
        self.tr_count = 0
        self.feedback_count = 0
        self.start_time = int(np.floor(time.time()))
        log_file_name = os.getcwd()+'/log/'+str(self.start_time )+'_event.log'
        self.log_file = open(os.path.normpath(log_file_name),'w')
        self.write_log_header()

        # networking
        self.shutdown_url = 'http://127.0.0.1:5000/shutdown'
        self.target_class = mp.Value('i', 1)
        self.feedback_calc_trial = mp.Value('i', -1)
        self.clf_outs = mp.Array('d', self.NUM_CLASSES)
        self.server_running_bool = mp.Value('i', 1)
        self.server_process = mp.Process(target = gs.start_server,
                                         args = (self.target_class,
                                            self.feedback_calc_trial,
                                            self.clf_outs,))
        self.server_process.start()

    def check_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_5: self.write_trigger()
                elif event.key == pygame.K_f: self.send_fake_insta()
                elif event.key == pygame.K_ESCAPE: self.quit()
        self.check_for_feedback()

    def run(self):
        while True:
            time_passed = self.clock.tick_busy_loop(self.FRAME_RATE)
            self.check_input()
            self.draw_background()
            self.draw_feedback()
            pygame.display.flip()

    def check_for_feedback(self):
        if self.feedback_calc_trial.value == self.feedback_count:
            self.write_log('feedback',self.feedback_count+self.BASELINE_TRS-1)
            self.feedback_count += 1
            target_class = 0
            self.feedback_score = self.clf_outs[target_class]

    def write_log_header(self):
        self.log_file.write('time,event,tr\n')

    def write_trigger(self):
        self.write_log('trigger', self.tr_count)
        self.tr_count += 1

    def write_log(self, event_name, count):
        self.log_file.write(str(time.time()-self.start_time)+','+event_name+','+str(count)+'\n')

    def send_fake_insta(self):
        post_url = 'http://127.0.0.1:5000/rt_data'
        out_data = np.random.uniform(0,1,self.NUM_CLASSES)
        trial_count = self.feedback_count
        target_class = self.target_class.value
        payload = {"clf_outs": list(out_data.flatten()),
            "feedback_num": trial_count,
            "target_class": target_class}
        status_code = 404
        while status_code != 200:
            post_status = r.post(post_url, json=payload)
            status_code = post_status.status_code

    def draw_feedback(self):
        draw_filled_aacircle(self.screen,
             self.feedback_score*self.MAX_CIRCLE_DIAMETER/2.,
             self.CIRCLE_COLOR,
             self.CIRCLE_POS[0],
             self.CIRCLE_POS[1]) 

    def draw_background(self):
        self.screen.fill(self.BG_COLOR)

    def quit(self):
        r.post(self.shutdown_url)
        sys.exit()   

def draw_filled_aacircle(screen, radius, color, xpos, ypos):
    pygame.gfxdraw.filled_circle(screen,
                                 int(xpos),
                                 int(ypos),
                                 int(radius),
                                 color)
    pygame.gfxdraw.aacircle(screen,
                            int(xpos),
                            int(ypos),
                            int(radius),
                            color)

if __name__ == "__main__":
    game = neurofeedbackGame()
    game.run()
