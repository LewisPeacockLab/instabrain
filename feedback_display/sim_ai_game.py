import game_init as gi
import game_run as gr
import numpy as np
import fileutils as fu
from target import Target
from controller import Controller
from tqdm import *

class SimAiGame(object):
    def __init__(self):
        gi.generate_constants(self, 'sim')
        gi.generate_variables(self, 'sim')
        self.target = Target(self.FRAME_RATE, self.SUCCESS_SCORE)
        self.controller = Controller(self.tr,
                                     self.SUCCESS_SCORE,
                                     self.target.MIN_ERROR_METRIC,
                                     self.target.SS_ERROR_METRIC)
        self.target.set_fb_mode('hrf')
        self.target.target_list = self.target.SIM_AI_TARGET_LIST

    def set_next_orientation(self):
        self.ori = self.target.set_next_target(self.trial_count)
        
    def set_noise(self, noise_std):
        self.noise_std = noise_std

    def run(self):
        for controller in tqdm(self.controller_list):
            fu.timed_controller_record_sim(self, self.f_timed_controller)
            self.controller.set_control_params(
                controller['sin_amplitude_max'],
                controller['sin_amplitude_min'],
                controller['sin_frequency'],
                controller['integrator_gain'],
                controller['estimate_samples'])
            while True:
                if self.trial_count < self.target.SIM_AI_TARGET_COUNT:
                    if not(self.trial_started):
                        self.set_next_orientation()
                        self.trial_started = True

                        self.ori = self.DEFAULT_ORIENTATION

                        game.tr_frame_counter = 0
                        game.signal_frame_counter = 0
                        game.tr_counter = 0

                    elif (not(self.target_reached) and
                            self.tr_counter < self.TR_LIMIT):
                        self.signal_frame_counter += 1
                        self.tr_frame_counter += 1
                        if self.signal_frame_counter >= self.SAMPLE_FRAMES:
                            gr.signal_based_updates(self)
                            self.signal_frame_counter -= self.SAMPLE_FRAMES
                        if self.tr_frame_counter >= self.FRAME_RATE:
                            gr.tr_based_updates_game(self)
                            self.tr_frame_counter -= self.FRAME_RATE
                            self.ori = self.controller.update(self.target.error_metric)
                            self.tr_counter += 1
                        gr.frame_based_updates(self)
                        self.target.update(self.ori)
                    else:
                        if self.target_reached == False:
                            fu.timed_trial_record_sim(self, self.f_timed_trial)
                        self.controller.reset()
                        self.trial_started = False
                        self.target_reached = False
                        self.trial_count += 1
                else:
                    break
            self.controller_type += 1
            self.trial_count = 0

if __name__ == "__main__":
    game = SimAiGame()
    game.run()