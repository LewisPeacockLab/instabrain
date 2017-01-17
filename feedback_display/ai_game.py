from psychopy.iohub import EventConstants
from psychopy import core

import game_init as gi
import game_run as gr
import numpy as np
from target import Target
from controller import Controller

class AiGame(object):

    def __init__(self):
        gi.generate_constants(self, 'ai')
        gi.generate_variables(self, 'ai')
        self.target = Target(self.FRAME_RATE, self.SUCCESS_SCORE)
        self.controller = Controller(self.tr,
                                     self.SUCCESS_SCORE,
                                     self.target.MIN_ERROR_METRIC,
                                     self.target.SS_ERROR_METRIC)
        self.set_random_orientation()
        self.target.set_fb_mode('hrf')
        self.trial_type_count = 2

    def check_input(self):
        for event in self.keyboard.getEvents():
            if event.type == EventConstants.KEYBOARD_PRESS:
                if event.key == u'5':
                    gr.start_run(game)
                elif event.key == u' ':
                    gr.start_run(game)
                elif event.key == u'escape':
                    self.quit()
            if event.type == EventConstants.KEYBOARD_RELEASE:
                pass

    def set_random_orientation(self):
        self.target_grating.ori = self.target.set_new_random_target()

    def set_next_orientation(self):
        self.target_grating.ori = self.target.set_next_target(self.trial_count)

    def run(self):
        while True:
            self.check_input()
            if self.run_trials == True:
                if self.begin_rest_clock.getTime() < 0:
                    gr.run_rest(self)
                elif not(self.trial_started):
                    self.set_next_orientation()
                    gr.start_trial(self)
                elif not(self.target_reached):
                    if self.ENABLE_BLINK:
                        if self.blink_clock.getTime() > 0:
                            gr.blink_based_updates(self)
                            self.blink_clock.add(self.BLINK_TIME)
                    if self.signal_clock.getTime() > 0:
                        gr.signal_based_updates(self)
                        self.signal_clock.add(self.SAMPLE_PERIOD)
                    if self.tr_clock.getTime() > 0:
                        gr.tr_based_updates_game(self)
                        self.tr_clock.add(self.tr)
                        self.grating.ori = self.controller.update(self.target.error_metric)
                        self.debug_msg.text = self.debug_msg_base.format(
                                                sin_in=str(np.around(self.controller.sin_input,2)),
                                                sin_conv=str(np.around(self.controller.conv_sin[self.controller.time_count-1],2)),
                                                delt_est=str(np.around(self.controller.integrator_gain*self.controller.output_change_estimate,2)),
                                                ang_est=str(np.around(self.controller.angle_estimate,2)))
                    gr.run_trial(self)
                elif self.success_show_clock.getTime() < 0:
                    gr.run_score(self)        
                else:
                    self.controller.reset()
                    gr.reset_for_next_trial(self)

                self.draw()
            else:
                self.draw_splash()
            self.screen.flip()

    def draw(self):
        if self.grating_blink_visible:
            if self.show_grating:
                self.grating.draw()
                self.grating_mask.draw()
            if self.show_target:
                self.target_grating.draw()
        self.success_score_circ.draw()
        if self.show_score:
            self.score_circ.draw()
        self.fixation.draw()
        self.debug_msg.draw()


    def draw_splash(self):
        game.splash_msg.draw()

    def quit(self):
        core.quit()

if __name__ == "__main__":
    game = AiGame()
    game.run()