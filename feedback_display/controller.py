import numpy as np
from target import gen_hrf

class Controller(object):

    def __init__(self, tr, success_score,
                 min_score, max_score):
        self.SIN_AMPLITUDE = 60.#45. # in degrees
        self.SIN_AMPLITUDE_MAX = 90. # 90. # in degrees
        self.SIN_AMPLITUDE_MIN = 0. # 0 # in degrees
        self.SIN_AMPLITUDE_RANGE = self.SIN_AMPLITUDE_MAX-self.SIN_AMPLITUDE_MIN
        self.SIN_FREQUENCY = 0.01 #0.025 # in Hz
        self.INTEGRATOR_GAIN = 35. #15.0
        self.INIT_WAIT_TIME = 10
        self.INIT_ESTIMATE_TIME = 4
        self.ESTIMATE_TIME = 6
        self.SCORE_BUFFER_LENGTH = 100
        self.FRAME_RATE = 1/float(tr)
        self.HRF = gen_hrf(self.FRAME_RATE)
        self.HRF = self.HRF[::-1]

        self.SUCCESS_SCORE = success_score
        self.MIN_SCORE = min_score
        self.MAX_SCORE = max_score
        self.SCORE_RANGE = float(self.MAX_SCORE-self.MIN_SCORE)
        self.TR = float(tr)
        self.init_wait_samples = self.INIT_WAIT_TIME/tr
        self.init_estimate_samples = self.INIT_ESTIMATE_TIME/tr
        self.estimate_samples = self.ESTIMATE_TIME/tr

        self.sin_amplitude = self.SIN_AMPLITUDE  
        self.integrator_gain = self.INTEGRATOR_GAIN
        self.conv_sin = gen_conv_sin(self.HRF, self.SIN_FREQUENCY)
        self.reset()

    def reset(self):
        self.angle_estimate = 0
        self.output_change_estimate = 0
        self.init_estimate_bool = False
        self.init_time_count = 0
        self.init_score_estimate = 0
        self.time_count = 0
        self.sin_input = 0
        self.plant_input = 0
        self.score_buffer = self.MIN_SCORE*np.ones(self.SCORE_BUFFER_LENGTH)

    def update(self, score):
        self.score_buffer = np.roll(self.score_buffer, -1)
        self.score_buffer[-1] = score
        # if self.init_time_count < self.init_wait_samples:
        #     self.init_time_count += 1
        # elif not(self.init_estimate_bool):
        #     self.init_estimate()
        if False:
            pass
        else:
            self.update_sin_amplitude(np.mean(
                                  self.score_buffer[-self.estimate_samples:-1]))
            self.time_count += self.TR
            self.sin_input = np.sin(self.time_count*self.SIN_FREQUENCY*2*np.pi)
            self.output_change_estimate = (score - np.mean(
                                  self.score_buffer[-self.estimate_samples:-1]))
            # self.output_change_estimate = (score - self.init_score_estimate)
            # self.angle_estimate += (self.integrator_gain
            #                         *self.sin_input
            #                         *self.output_change_estimate)
            self.angle_estimate += (self.integrator_gain
                                    *self.conv_sin[self.time_count-1]
                                    *self.output_change_estimate)
            self.plant_input = self.sin_amplitude*self.sin_input+self.angle_estimate
        return self.plant_input + 90

    def init_estimate(self):
        self.init_score_estimate = np.mean(
                                self.score_buffer[-self.init_estimate_samples:])
        self.score_buffer[:] = self.init_score_estimate 
        # self.update_sin_amplitude(self.init_score_estimate)
        self.init_estimate_bool = True

    def update_sin_amplitude(self, score):
        self.sin_amplitude = (self.SIN_AMPLITUDE_MIN
                              +(self.SCORE_RANGE-(score-self.MIN_SCORE))
                                /self.SCORE_RANGE*self.SIN_AMPLITUDE_RANGE)

    def set_control_params(self,
                           sin_amplitude_max=90.,
                           sin_amplitude_min=0.,
                           sin_frequency=0.025,
                           integrator_gain=35,
                           estimate_samples=6):
        self.SIN_AMPLITUDE_MAX = sin_amplitude_max
        self.SIN_AMPLITUDE_MIN = sin_amplitude_min
        self.SIN_AMPLITUDE_RANGE = self.SIN_AMPLITUDE_MAX-self.SIN_AMPLITUDE_MIN
        self.SIN_FREQUENCY = sin_frequency
        self.integrator_gain = integrator_gain
        self.estimate_samples = estimate_samples
        self.conv_sin = gen_conv_sin(self.HRF, self.SIN_FREQUENCY)
        self.reset()

def gen_conv_sin(hrf, sin_frequency, length=1000):
    in_sin = np.sin(np.arange(length)*sin_frequency*2*np.pi)
    conv_sin = np.convolve(hrf, in_sin)
    return conv_sin
