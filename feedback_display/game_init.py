# from psychopy.iohub import launchHubServer
# from psychopy import core, visual, monitors
from psychopy import core
import fileutils as fu
import numpy as np
import yaml

def generate_constants(game, mode):
    with open('config.yaml') as f:
        game.CONFIG = yaml.load(f)
    game.LOCATION = game.CONFIG['location']
    game.SUBJECT_ID = game.CONFIG['subject-id']
    game.FULLSCREEN = game.CONFIG['fullscreen']
    game.RUNS = game.CONFIG['runs'][mode]
    game.SPLASH_MSG_BASE = 'Ready for Run {current} of {total}'
    game.ENABLE_BLINK = game.CONFIG['enable-blink']
    game.FRAME_RATE = 60
    game.VISIBLE_TRIALS = game.CONFIG['visible-trials']
    game.INVISIBLE_TRIALS = game.CONFIG['invisible-trials']
    game.NUM_TRIALS = game.VISIBLE_TRIALS
    game.TRIAL_TYPES = 9
    game.MODE = mode

    # game parameters
    game.ROTATION_SPEED = 45 # degrees/sec
    game.BLINK_TIME = .25
    game.DEFAULT_ORIENTATION = 90
    game.SAMPLE_PERIOD = 1.
    game.SUCCESS_SCORE = .85
    game.REACH_SUCCESS_TIME = 10. #6. #4.
    if game.MODE == 'sim':
        game.SIM_ADD_REACH_TIME = 0. #10.
        game.REACH_SUCCESS_TIME += game.SIM_ADD_REACH_TIME
    game.SAMPLE_FRAMES = game.SAMPLE_PERIOD*game.FRAME_RATE

    game.BEGIN_REST_TIME_LAB = 3
    game.SUCCESS_SHOW_TIME_LAB = 2
    game.TRIALS_PER_RUN_LAB = 10

    game.BEGIN_REST_TIME_SCANNER = 10
    game.SUCCESS_SHOW_TIME_SCANNER = 2
    game.TRIALS_PER_RUN_SCANNER = 999

    game.NOISE_STD_LIST = [0.01, 0.15] #[0.025, 0.1]
    # game.NOISE_STD_LIST = [0.15, 0.15] #[0.025, 0.1]
    game.TR_LIST = [2, 6] #[1,6] #[1,2,4]
    game.TRS_SAMPLES_DICT = dict((tr, tr/game.SAMPLE_PERIOD)
                                 for tr in game.TR_LIST)
    game.TRS_SUCCESS_DICT = dict((tr, game.REACH_SUCCESS_TIME/tr)
                                 for tr in game.TR_LIST)

    if game.LOCATION == 'lab':
        game.BEGIN_REST_TIME = game.BEGIN_REST_TIME_LAB
        game.SUCCESS_SHOW_TIME = game.SUCCESS_SHOW_TIME_LAB 
        game.TRIALS_PER_RUN = game.TRIALS_PER_RUN_LAB
    elif game.LOCATION == 'scanner':
        game.BEGIN_REST_TIME = game.BEGIN_REST_TIME_SCANNER
        game.SUCCESS_SHOW_TIME = game.SUCCESS_SHOW_TIME_SCANNER
        game.TRIALS_PER_RUN = game.TRIALS_PER_RUN_SCANNER

    # ai sim nfb possible parameters
    game.sin_amplitude_max_array = [165] #[180] #[165]
    game.sin_amplitude_min_array = [0.]
    game.sin_frequency_array = [0.01] #[0.015]
    game.integrator_gain_array = [75] #[45] #[65]
    game.estimate_samples_array = [6.]
    game.controller_list = []
    for amp_max in game.sin_amplitude_max_array:
        for amp_min in game.sin_amplitude_min_array:
            for sin_freq in game.sin_frequency_array:
                for int_gain in game.integrator_gain_array:
                    for est_samples in game.estimate_samples_array:
                        game.controller_list.append({
                            'sin_amplitude_max':amp_max,
                            'sin_amplitude_min':amp_min,
                            'sin_frequency':sin_freq,
                            'integrator_gain':int_gain,
                            'estimate_samples':est_samples})


    # localizer parameters
    game.ANGLE_BETWEEN_TARGETS = 22.5
    game.TRIAL_TIME_LOCALIZER = 16
    game.TRIALS_PER_RUN_LOCALIZER = 18 # 2 rest trials
    game.LOC_LIST = np.array([[-1,3,5,4,0,6,2,1,7,2,7,1,3,0,5,4,6,-1],
                              [-1,6,7,5,2,1,4,0,3,2,1,5,4,0,6,3,7,-1],
                              [-1,4,0,5,2,1,7,3,6,7,3,5,0,4,6,2,1,-1],
                              [-1,2,5,0,1,7,6,4,3,5,7,6,0,4,1,2,3,-1],
                              [-1,5,0,4,3,6,2,7,1,5,4,3,1,6,0,7,2,-1],
                              [-1,6,7,2,5,4,0,1,3,5,1,6,4,3,7,0,2,-1],
                              [-1,3,0,5,1,7,4,6,2,4,3,1,7,6,2,5,0,-1],
                              [-1,7,1,2,5,3,4,6,0,5,6,3,4,0,1,7,2,-1]])

    # screen proportions
    game.SCANNER_SCREEN_DIMS = [1024, 768]
    game.SCANNER_DISTANCE = 136 # cm
    game.SCANNER_WIDTH = 85.7 # cm
    game.LAB_SCREEN_DIMS = [1920, 1200]
    game.LAB_DISTANCE = 68.7 # cm
    game.LAB_WIDTH = 52 # cm
    if game.LOCATION == 'scanner':
        game.SPATIAL_FREQUENCY = 1.5
    elif game.LOCATION == 'lab':
        game.SPATIAL_FREQUENCY = 0.5
    game.GRATING_SIZE_MAX = 20
    game.GRATING_SIZE_MIN = 4 #3
    game.GRATING_SIZE_TARGET = 4.5
    game.FIXATION_SIZE = .5
    game.SCORE_SIZE_MIN = .6
    game.SCORE_SIZE_MAX = 3.5
    game.SCORE_SIZE_RANGE = game.SCORE_SIZE_MAX - game.SCORE_SIZE_MIN
    game.SCORE_SIZE_SUCCESS = (game.SCORE_SIZE_MIN
                               + game.SUCCESS_SCORE*game.SCORE_SIZE_RANGE)

    # colors
    game.SUCCESS_SCORE_COLOR = (100,100,100)
    game.CURRENT_SCORE_COLOR = (80,150,80)
    game.LAB_CONTRAST = 0.5
    game.SCANNER_CONTRAST = 1

    if game.LOCATION == 'lab':
        game.CONTRAST = game.LAB_CONTRAST
        game.SCREEN_DIMS = game.LAB_SCREEN_DIMS
        game.SCREEN_WIDTH = game.LAB_WIDTH
        game.SCREEN_DISTANCE = game.LAB_DISTANCE
    elif game.LOCATION == 'scanner':
        game.CONTRAST = game.SCANNER_CONTRAST
        game.SCREEN_DIMS = game.SCANNER_SCREEN_DIMS
        game.SCREEN_WIDTH = game.SCANNER_WIDTH
        game.SCREEN_DISTANCE = game.SCANNER_DISTANCE

    # visual stim parameters
    # game.GRATING_TYPE = 'raisedCos'
    # game.GRATING_TEXTURE = 'sin'
    # game.GRATING_PARAMS = {'fringeWidth':0.05}
    game.GRATING_TYPE = 'raisedCos'
    game.GRATING_TEXTURE = 'sqr'
    game.GRATING_PARAMS = {'fringeWidth':0.05}


def generate_variables(game, mode):
    # file recording
    game.subj_id = game.CONFIG['subject-id']
    game.subj_dir = 'datasets/' + game.subj_id
    if game.MODE == 'sim':
        fu.write_all_headers_timed_sim(game)
    else:
        fu.write_all_headers_timed(game)

    # game variables
    game.direction = 'none'
    game.grating_blink_visible = True
    game.grating_visible = True
    game.show_target = False
    game.target_visible_trial = True
    game.show_grating = True
    game.show_score = True
    game.trial_started = False
    game.last_time = 0
    game.tr = game.TR_LIST[0]
    game.run_trials = False
    game.target_reached = False
    game.noise_std = game.NOISE_STD_LIST[0]

    # io and display
    if mode == 'sim':
        game.ori = 0
        game.tr_frame_counter = 0
        game.signal_frame_counter = 0
        game.tr_counter = 0
        game.TR_LIMIT = 180
        game.controller_type = 0
        game.set_noise(game.CONFIG['sim-noise-std'])
    else:
        from psychopy.iohub import launchHubServer
        from psychopy import visual, monitors
        io=launchHubServer()
        game.keyboard = io.devices.keyboard
        game.monitor = monitors.Monitor(game.LOCATION,
                                        width=game.SCREEN_WIDTH,
                                        distance=game.SCREEN_DISTANCE)
        game.monitor.setSizePix(game.SCREEN_DIMS)
        game.screen = visual.Window(game.SCREEN_DIMS,
                                    monitor=game.monitor,
                                    fullscr=game.FULLSCREEN,
                                    units='deg')

        # visual stims
        game.grating = visual.GratingStim(game.screen,
                                          sf=game.SPATIAL_FREQUENCY,
                                          tex=game.GRATING_TEXTURE,
                                          size=game.GRATING_SIZE_MAX,
                                          mask=game.GRATING_TYPE,
                                          maskParams=game.GRATING_PARAMS,
                                          contrast=game.CONTRAST,
                                          ori=game.DEFAULT_ORIENTATION,
                                          interpolate=True) 
        game.target_grating = visual.GratingStim(game.screen,
                                                 sf=game.SPATIAL_FREQUENCY,
                                                 tex=game.GRATING_TEXTURE,
                                                 size=game.GRATING_SIZE_TARGET,
                                                 mask=game.GRATING_TYPE,
                                                 maskParams=game.GRATING_PARAMS,
                                                 contrast=game.CONTRAST,
                                                 ori=90,
                                                 interpolate=True) 
        game.grating_mask = visual.GratingStim(game.screen,
                                                color=0,
                                                colorSpace='rgb',
                                                tex=None,
                                                # mask='circle',
                                                mask=game.GRATING_TYPE,
                                                maskParams=game.GRATING_PARAMS,
                                                size=game.GRATING_SIZE_MIN,
                                                interpolate=True)
        game.success_score_circ = visual.GratingStim(game.screen,
                                                    color=game.SUCCESS_SCORE_COLOR,
                                                    colorSpace='rgb255',
                                                    tex=None,
                                                    mask='circle',
                                                    size=game.SCORE_SIZE_SUCCESS,
                                                    interpolate=True)
        game.score_circ = visual.GratingStim(game.screen,
                                                    color=game.CURRENT_SCORE_COLOR,
                                                    colorSpace='rgb255',
                                                    tex=None,
                                                    mask='circle',
                                                    size=game.SCORE_SIZE_MIN*2,
                                                    interpolate=True)
        game.fixation = visual.GratingStim(game.screen,
                                           color=-1,
                                           colorSpace='rgb',
                                           tex=None,
                                           mask='circle',
                                           size=game.FIXATION_SIZE,
                                           interpolate=True)

        game.splash_msg = visual.TextStim(game.screen,
                                          text=game.SPLASH_MSG_BASE.format(current=str(1),
                                                                           total=str(game.RUNS)))

        game.debug_msg = visual.TextStim(game.screen,
                                         text='',
                                         pos=(-0.25,-0.25))

        game.debug_msg_base = ('sin_in: {sin_in}\n'
                               +'sin_conv: {sin_conv}\n'
                               +'delt_est: {delt_est}\n'
                               +'ang_est: {ang_est}\n')

    # timers
    game.global_clock = core.Clock()
    game.blink_clock = core.Clock()
    game.tr_clock = core.Clock()
    game.signal_clock = core.Clock()
    game.trial_clock = core.Clock()
    game.begin_rest_clock = core.Clock()
    game.success_show_clock = core.Clock()
    game.trial_count = 0
    game.trial_type_count = 0
    game.run_count = -1
