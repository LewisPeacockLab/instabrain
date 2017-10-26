from psychopy.iohub import EventConstants
from psychopy import core
import numpy as np
import game_init as gi
import argparse, time
import requests as r

parser = argparse.ArgumentParser(description='Function arguments')
parser.add_argument('-s','--subjectid', help='Subject ID',default='demo')
parser.add_argument('-n','--sessionnum', help='Session number',default='1')
args = parser.parse_args()

class Game(object):
    def __init__(self):
        gi.generate_constants(self, args.subjectid, args.sessionnum)
        gi.generate_variables(self)

    def check_input(self):
        for event in self.keyboard.getEvents():
            if event.type == EventConstants.KEYBOARD_PRESS:
                if event.key == u'5' or event.key == u' ':
                    if self.trial_stage == 'splash':
                        self.start_run()
                elif event.key == u'escape':
                    self.quit()
                elif event.key == self.key_codes[0]: self.keydown[0] = True
                elif event.key == self.key_codes[1]: self.keydown[1] = True
                elif event.key == self.key_codes[2]: self.keydown[2] = True
                elif event.key == self.key_codes[3]: self.keydown[3] = True
                elif event.key == self.key_codes[4]: self.keydown[4] = True
            elif event.type == EventConstants.KEYBOARD_RELEASE:
                if event.key == self.key_codes[0]: self.keydown[0] = False
                elif event.key == self.key_codes[1]: self.keydown[1] = False
                elif event.key == self.key_codes[2]: self.keydown[2] = False
                elif event.key == self.key_codes[3]: self.keydown[3] = False
                elif event.key == self.key_codes[4]: self.keydown[4] = False

    def run(self):
        while True:
            self.check_input()
            if self.trial_stage == 'splash':
                self.draw_splash()
            else:
                self.timer_based_updates()
                if (self.trial_stage == 'wait' and
                        (self.feedback_status == 'idle' or self.feedback_status == 'wait')):
                    self.check_for_next_feedback_value()
                if self.trial_stage in ('zscore','wait','iti'):
                    self.run_rest()
                elif self.trial_stage == 'cue':
                    self.run_cue()
                elif self.trial_stage == 'feedback':
                    self.run_feedback()
                self.draw_trials()
            self.screen.flip()

    def start_run(self):
        self.feedback_score_history = [self.feedback_score_history[-1]]
        self.run_reward_history = []
        self.trial_clock.reset() 
        self.trial_clock.add(self.ZSCORE_TIME)
        self.reset_for_next_trial()
        self.feedback_calc_trial.value = -1
        self.trial_count = 0
        self.run_count += 1
        if self.run_count < self.RUNS-1:
            self.splash_msg.text = self.SPLASH_MSG_BASE.format(current=str(self.run_count+2),
                total=str(self.RUNS))
        else:
            self.splash_msg.text = self.QUIT_MSG
        self.trial_stage = 'zscore' 

    def timer_based_updates(self):
        if self.trial_clock.getTime() < -self.TRIAL_TIME:
            self.trial_stage = 'zscore'
        elif self.trial_clock.getTime() < -self.CUE_TIME_LIMIT:
            self.trial_stage = 'cue'
        elif self.trial_clock.getTime() < -self.WAIT_TIME_LIMIT:
            self.trial_stage = 'wait'
        elif self.trial_clock.getTime() < -self.FEEDBACK_TIME_LIMIT:
            self.trial_stage = 'feedback'
        elif self.trial_clock.getTime() < 0:
            self.trial_stage = 'iti'
        else:
            self.reset_for_next_trial()
        if self.trial_count >= self.TRIALS_PER_RUN:
            self.reset_for_splash()

    def reset_for_next_trial(self):
        self.trial_count += 1
        self.trial_clock.add(self.TRIAL_TIME)
        self.feedback_status = 'idle'

    def write_trial_header(self, clf_data):
        self.TRIAL_FILE.write('run_num,')
        for class_num in range(clf_data.size-1):
            self.TRIAL_FILE.write('class_'+str(class_num+1)+',')
        self.TRIAL_FILE.write('delay,')
        self.TRIAL_FILE.write('dollars_reward\n')

    def write_trial_data(self, clf_data, delay):
        self.TRIAL_FILE.write(str(game.run_count+1)+',')
        for class_num in range(clf_data.size-1):
            self.TRIAL_FILE.write(str(clf_data[class_num])+',')
        self.TRIAL_FILE.write(str(delay)+',')
        self.TRIAL_FILE.write(str(self.run_reward_history[-1])+'\n')

    def reset_for_splash(self):
        self.reward_msg.text = self.REWARD_MSG_BASE.format(
            run_reward=sum(self.run_reward_history),
            total_reward=self.total_reward)
        self.self_pace_start_clock.reset()
        self.self_pace_start_clock.add(self.SELF_PACE_START_TIME)
        self.trial_stage = 'splash'

    def check_for_next_feedback_value(self):
        if self.feedback_status == 'idle':
            self.feedback_status = 'wait'
            self.begin_wait_time = time.time()
        if self.feedback_calc_trial.value == self.trial_count:
            self.update_feedback_value()
            self.feedback_status = 'calculated'

    def update_feedback_value(self):
        clf_data = np.zeros(self.num_classes+1)
        clf_data[:-1] = self.clf_outs[:]
        clf_data[-1] = self.target_class.value
        target_score = clf_data[int(clf_data[-1]-1)]
        self.feedback_score_history.append(target_score)
        self.run_reward_history.append(target_score*self.MAX_TRIAL_REWARD)
        self.total_reward += self.run_reward_history[-1]
        if not(self.header_written):
            self.write_trial_header(clf_data)
            self.header_written = True
        delay = time.time()-self.begin_wait_time
        self.write_trial_data(clf_data, delay)

    def reset_feedback_clock(self):
        self.feedback_update_clock.reset()
        self.feedback_update_clock.add(self.FEEDBACK_UPDATE_TIME)

    def run_rest(self):
        self.show_cue = False
        self.show_feedback = False

    def run_cue(self):
        self.show_cue = True
        self.show_feedback = False

    def run_feedback(self):
        if self.feedback_status == 'calculated':
            self.reset_feedback_clock()
            self.feedback_status = 'display'
        if self.feedback_status == 'display':
            self.show_cue = False
            self.show_feedback = True
            time_ratio = max(0,-self.feedback_update_clock.getTime()/float(self.FEEDBACK_UPDATE_TIME))
            self.update_score(time_ratio*self.feedback_score_history[-2]
                              +(1-time_ratio)*self.feedback_score_history[-1])

    def update_score(self, score=.5):
        self.score_circ.size = 0.5*(self.SCORE_DIAMETER_MIN
                                + score*self.SCORE_DIAMETER_RANGE)

    def draw_trials(self):
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
        if not(any(self.keydown)):
            if self.self_pace_start_clock.getTime() < -self.SELF_PACE_START_TIME:
                self.self_pace_start_clock.reset()
                self.self_pace_start_clock.add(self.SELF_PACE_START_TIME)
            else:
                self.self_pace_start_clock.add(0.1*(self.self_pace_start_clock.getTime()+self.SELF_PACE_START_TIME))
        continue_ratio = max(0,min(1,
            1+self.self_pace_start_clock.getTime()/self.SELF_PACE_START_TIME))
        self.splash_msg.color = (game.BG_COLOR
            + (1-continue_ratio)*(game.TEXT_COLOR-game.BG_COLOR))
        self.reward_msg.color = self.splash_msg.color
        self.splash_msg.draw()
        self.reward_msg.draw()
        if self.self_pace_start_clock.getTime() > 0:
            if self.run_count >= self.RUNS-1:
                self.quit()
            for flip in range(2):
                self.screen.flip()
                self.fixation.draw()
            while self.trial_stage == 'splash':
                self.check_input()

    def quit(self):
        r.post(self.shutdown_url)
        core.quit()

if __name__ == "__main__":
    game = Game()
    game.run()
