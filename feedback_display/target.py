import numpy as np
from scipy.stats import vonmises
from numpy.random import permutation

class Target(object):

    def __init__(self, frame_rate, success_score):
        self.VARIANCE_DEGREES = 20**2
        self.VARIANCE = 2*np.deg2rad(self.VARIANCE_DEGREES) # 2* b/c vonmises scale=0.5
        self.KAPPA = 1/self.VARIANCE
        self.MIN_SUCCESS_SCORE = success_score
        self.VONMISES_MIN = vonmises.pdf(x=.5*np.pi,
                                         kappa=self.KAPPA,
                                         loc=0,
                                         scale=.5)
        self.VONMISES_MAX = vonmises.pdf(x=0,
                                         kappa=self.KAPPA,
                                         loc=0,
                                         scale=.5)
        self.VONMISES_RANGE = self.VONMISES_MAX-self.VONMISES_MIN
        self.MIN_ERROR_METRIC = 0.1
        self.SS_ERROR_METRIC = 0.9
        self.ERROR_METRIC_RANGE = self.SS_ERROR_METRIC-self.MIN_ERROR_METRIC

        self.ANGLE_BETWEEN_TARGETS = 22.5
        self.ALL_TARGET_LIST = self.transform_angle_list([1,2,3,4,5,6,7])

        SHORT_LIST = [1,2,3,4,5,6,7]
        self.SHORT_TARGET_LIST = self.transform_angle_list(SHORT_LIST)
        LONG_LIST = [1,2,3,4,5,6,7,
                     1,2,3,4,5,6,7,
                     1,2,3,4,5,6,7,
                     1,2,3,4,5,6,7]
        self.LONG_TARGET_LIST = self.transform_angle_list(LONG_LIST)
        self.target_list = self.SHORT_TARGET_LIST
        self.shuffle_target_list()

        # SIM_AI_LIST = [1,1,1,1,1,1,
        #                2,2,2,2,2,2,
        #                3,3,3,3,3,3,
        #                4,4,4,4,4,4,
        #                5,5,5,5,5,5,
        #                6,6,6,6,6,6,
        #                7,7,7,7,7,7]
        SIM_AI_LIST = [1,1,1,1,1,1,1,#1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       2,2,2,2,2,2,2,#2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                       3,3,3,3,3,3,3,#3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                       4,4,4,4,4,4,4,#4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
                       5,5,5,5,5,5,5,#5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                       6,6,6,6,6,6,6,#6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                       7,7,7,7,7,7,7]#,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]
        self.SIM_AI_TARGET_LIST = self.transform_angle_list(SIM_AI_LIST)
        self.SIM_AI_TARGET_COUNT = len(SIM_AI_LIST)

        self.hrf = gen_hrf(frame_rate)
        self.impulse = np.zeros(len(self.hrf))
        self.impulse[-1] = 1
        self.set_fb_mode('impulse')
        # self.set_fb_mode('hrf')

        self.pos = 45
        self.error = 0
        self.error_metric = 0

        ########################
        # error metric buffers #
        ########################
        self.error_metric_buffer = np.zeros(len(self.hrf))
        self.error_metric_conv_buffer = np.zeros(len(self.hrf))
        self.error_metric_conv_sampled_buffer = np.zeros(len(self.hrf))


    def update(self, cursor_pos):
        self.error_metric_buffer = np.roll(self.error_metric_buffer, -1) 
        self.error_metric_conv_buffer = np.roll(self.error_metric_conv_buffer, -1) 
        self.error_metric_buffer[-1] = self.error_calc(cursor_pos)
        self.error_metric_conv_buffer[-1] = np.dot(self.impulse_response,
                                                   self.error_metric_buffer)

    def transform_angle_list(self, angle_list):
        return 90 - self.ANGLE_BETWEEN_TARGETS*np.array(angle_list)

    def set_fb_mode(self, mode):
        if mode == 'impulse':
            self.fb_mode = 'impulse'
            self.impulse_response = self.impulse
        elif mode == 'hrf':
            self.fb_mode = 'hrf'
            self.impulse_response = self.hrf

    def set_new_random_target(self):
        self.pos = np.random.choice(self.ALL_TARGET_LIST)
        self.reset_error_metric()
        return self.pos

    def set_next_target(self, index):
        self.pos = self.target_list[index]
        self.reset_error_metric()
        return self.pos

    def shuffle_target_list(self):
        self.target_list = permutation(self.target_list)

    def reset_error_metric(self):
        self.error_metric = self.MIN_ERROR_METRIC
        self.error_metric_buffer[:] = self.MIN_ERROR_METRIC
        self.error_metric_conv_buffer[:] = self.MIN_ERROR_METRIC
        self.error_metric_conv_sampled_buffer[:] = self.MIN_ERROR_METRIC

    def error_calc(self, cursor_pos):
        target_radians = np.deg2rad(self.pos)
        cursor_radians = np.deg2rad(cursor_pos)
        error_metric_vm = vonmises.pdf(x=cursor_radians,
                                       kappa=self.KAPPA,
                                       loc=target_radians,
                                       scale=0.5)
        error_metric = (self.MIN_ERROR_METRIC + (error_metric_vm - self.VONMISES_MIN)
                        * self.ERROR_METRIC_RANGE/self.VONMISES_RANGE)
        return error_metric       

def gen_hrf(frame_rate=120, tf=30,
            c=1/6.0, a1=6, a2=16, A=1/0.833657):
    ts = 1/float(frame_rate)
    A = A*ts
    t = np.arange(0,tf,ts)
    h = (A*np.exp(-t)*(t**(a1-1)/np.math.factorial(a1-1)
         - c*t**(a2-1)/np.math.factorial(a2-1)))
    return(h[::-1])
