from psychopy.iohub import EventConstants
from psychopy import core

import game_init as gi
import game_run as gr
from target import Target

class Game(object):

    def __init__(self):
        gi.generate_constants(self, 'game')
        gi.generate_variables(self, 'game')
        self.target = Target(self.FRAME_RATE, self.SUCCESS_SCORE)
        self.set_random_orientation()

    def check_input(self):
        for event in self.keyboard.getEvents():
            if event.type == EventConstants.KEYBOARD_PRESS:
                if event.key == u'1':
                    self.direction = 'ccw'
                elif event.key == u'2':
                    self.direction = 'cw'
                # elif event.key == u'3':
                #     game.run_trials = not(game.run_trials)
                # elif event.key == u'4':
                #     game.target_visible_trial = not(game.target_visible_trial)
                elif event.key == u'5':
                    if not(self.run_trials):
                        gr.start_run(game)
                # elif event.key == u'6':
                #     self.set_random_orientation()
                elif event.key == u' ':
                    if not(self.run_trials):
                        gr.start_run(game)
                elif event.key == u'escape':
                    self.quit()
            if event.type == EventConstants.KEYBOARD_RELEASE:
                # fixthis could add keydown state
                # (holding 1, press then release 2)
                if event.key == u'1':
                    if self.direction == 'ccw':
                        self.direction = 'none'
                elif event.key == u'2':
                    if self.direction == 'cw':
                        self.direction = 'none'

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
                    # self.set_random_orientation()
                    self.set_next_orientation()
                    gr.start_trial(self)
                elif not(self.target_reached):
                    if self.ENABLE_BLINK:
                        if self.blink_clock.getTime() > 0:
                            gr.blink_based_updates(self)
                            self.blink_clock.add(self.BLINK_TIME)
                    if self.signal_clock.getTime() > 0:
                        gr.signal_based_updates(self)
                        self.signal_clock.add(game.SAMPLE_PERIOD)
                    if self.tr_clock.getTime() > 0:
                        gr.tr_based_updates_game(self)
                        self.tr_clock.add(self.tr)
                    gr.run_trial(self)
                elif self.success_show_clock.getTime() < 0:
                    gr.run_score(self)        
                else:
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

    def draw_splash(self):
        game.splash_msg.draw()

    def quit(self):
        core.quit()

if __name__ == "__main__":
    game = Game()
    game.run()