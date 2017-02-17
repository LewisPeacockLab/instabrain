from psychopy.iohub import EventConstants
from psychopy import core
import numpy as np
import requests as r
import StringIO as sio
import game_init as gi

class Game(object):
    def __init__(self):
        gi.generate_constants(self, 'demo')
        gi.generate_variables(self, 'demo')

    def check_input(self):
        for event in self.keyboard.getEvents():
            if event.type == EventConstants.KEYBOARD_PRESS:
                if event.key == u'5':
                    if not(self.run_trials):
                        self.start_run()
                elif event.key == u' ':
                    if not(self.run_trials):
                        self.start_run()
                elif event.key == u'escape':
                    self.quit()

    def run(self):
        while True:
            self.check_input()
            if self.run_trials == True:
                if self.zscore_clock.getTime() < 0:
                    self.run_rest()
                elif self.trial_count < self.TRIALS_PER_RUN:
                    if self.trial_clock.getTime() < -self.CUE_TIME_LIMIT:
                        self.run_cue()
                    elif self.trial_clock.getTime() < -self.WAIT_TIME_LIMIT:
                        self.run_rest()
                    elif not(self.feedback_calc_bool):
                        # could run this earlier?
                        # depends on how long request takes
                        # and when data is available from smoker
                        self.get_next_feedback_value()
                    elif self.trial_clock.getTime() < -self.FEEDBACK_TIME_LIMIT:
                        self.run_feedback()
                    elif self.trial_clock.getTime() < 0:
                        self.run_rest()
                    else:
                        self.reset_for_next_trial()
                else:
                    self.run_trials = False
                self.draw()
            else:
                self.draw_splash()
            self.screen.flip()

    def start_run(self):
        self.trial_count = 0
        self.zscore_clock.reset()
        self.zscore_clock.add(self.ZSCORE_TIME)
        self.trial_clock.reset() 
        self.trial_clock.add(self.ZSCORE_TIME+self.TRIAL_TIME)
        self.run_count += 1
        self.splash_msg.text = self.SPLASH_MSG_BASE.format(current=str(self.run_count+2),
            total=str(game.RUNS))
        self.run_trials = True

    def reset_for_next_trial(self):
        self.trial_count += 1
        self.trial_clock.add(self.TRIAL_TIME)
        self.feedback_calc_bool = False

    def get_next_feedback_value(self):
        next_feedback = ''
        while next_feedback == '':
            try:
                next_feedback = r.get(self.NETWORK_TARGET+str(game.trial_count)+'.txt',timeout=(0.01,0.01)).text
            except:
                pass
        np_data = np.loadtxt(sio.StringIO(next_feedback))
        self.feedback_score_history.append(next_feedback)
        # add file writing here? to datasets/subjectid/...
        self.feedback_update_clock.reset()
        self.feedback_update_clock.add(self.FEEDBACK_UPDATE_TIME)
        self.feedback_calc_bool = True

    def update_score(self, score=.5):
        self.score_circ.size = 0.5*(self.SCORE_DIAMETER_MIN
                                + score*self.SCORE_DIAMETER_RANGE)

    def run_rest(self):
        self.show_cue = False
        self.show_feedback = False

    def run_cue(self):
        self.show_cue = True
        self.show_feedback = False

    def run_feedback(self):
        self.show_cue = False
        self.show_feedback = True
        time_ratio = max(0,-self.feedback_update_clock.getTime()/float(self.FEEDBACK_UPDATE_TIME))
        self.update_score(time_ratio*self.feedback_score_history[-2]
                          +(1-time_ratio)*self.feedback_score_history[-1])

    def draw(self):
        if self.show_feedback:
            self.max_score_circ.draw()
            self.max_score_circ_mask.draw()
            self.score_circ.draw()
            self.score_circ_mask.draw()
        if self.show_cue:
            self.cue.draw()
        else:
            self.fixation.draw()

    def draw_splash(self):
        self.splash_msg.draw()

    def quit(self):
        core.quit()

if __name__ == "__main__":
    game = Game()
    game.run()
