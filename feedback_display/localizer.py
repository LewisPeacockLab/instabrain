from psychopy.iohub import EventConstants
from psychopy import core

import game_init as gi
import game_run as gr

class Localizer(object):

    def __init__(self):
        gi.generate_constants(self, 'localizer')
        gi.generate_variables(self, 'localizer')

    def check_input(self):
        for event in self.keyboard.getEvents():
            if event.type == EventConstants.KEYBOARD_PRESS:
                if event.key == u'3':
                    game.run_trials = not(game.run_trials)
                elif event.key == u'5':
                    if not(self.run_trials):
                        gr.start_run(game)
                elif event.key == u' ':
                    if not(self.run_trials):
                        gr.start_run(game)
                elif event.key == u'escape':
                    self.quit()
            if event.type == EventConstants.KEYBOARD_RELEASE:
                if event.key == u'1':
                    if self.direction == 'ccw':
                        self.direction = 'none'
                elif event.key == u'2':
                    if self.direction == 'cw':
                        self.direction = 'none'

    def run(self):
        while True:
            self.check_input()
            if game.run_trials == True:
                # gr.frame_based_updates(self)
                if game.ENABLE_BLINK:
                    if game.blink_clock.getTime() > 0:
                        gr.blink_based_updates(self)
                        self.blink_clock.add(self.BLINK_TIME)
                if game.tr_clock.getTime() > 0:
                    gr.tr_based_updates_localizer(self)
                    self.tr_clock.add(self.tr)
                if game.trial_clock.getTime() > 0:
                    self.trial_count += 1
                    if self.trial_count >= self.TRIALS_PER_RUN_LOCALIZER:
                        self.run_trials = False
                    else:
                        gr.trial_based_updates_localizer(self)
                        self.trial_clock.add(game.TRIAL_TIME_LOCALIZER)
                self.draw()
            else:
                self.draw_splash()
            self.screen.flip()

    def draw(self):
        if self.grating_visible:
            if self.grating_blink_visible:
                self.grating.draw()
                self.grating_mask.draw()
        self.success_score_circ.draw()
        if self.grating_visible:
            self.score_circ.draw()
        self.fixation.draw()

    def draw_splash(self):
        game.splash_msg.draw()

    def quit(self):
        core.quit()

if __name__ == "__main__":
    game = Localizer()
    game.run()