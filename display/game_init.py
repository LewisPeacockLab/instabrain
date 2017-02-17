from psychopy.iohub import launchHubServer
from psychopy import visual, monitors
from psychopy import core
import numpy as np
import yaml

def generate_constants(game, subject_id):
    with open('game_config.yml') as f:
        game.CONFIG = yaml.load(f)

    # game parameters
    game.SUBJECT_ID = subject_id
    game.SUBJECT_DIR = 'datasets/' + game.SUBJECT_ID
    game.FULLSCREEN = game.CONFIG['fullscreen']
    game.RUNS = game.CONFIG['runs']
    game.TRIALS_PER_RUN = game.CONFIG['trials-per-run']
    game.SPLASH_MSG_BASE = 'Ready for Run {current} of {total}'

    # timing
    game.TR_TIME = game.CONFIG['tr-time']
    game.ZSCORE_TIME = game.TR_TIME*game.CONFIG['zscore-trs']
    game.CUE_TIME = game.TR_TIME*game.CONFIG['cue-trs']
    game.WAIT_TIME = game.TR_TIME*game.CONFIG['wait-trs']
    game.FEEDBACK_TIME = game.TR_TIME*game.CONFIG['feedback-trs']
    game.ITI_TIME = game.TR_TIME*game.CONFIG['iti-trs']

    game.WAIT_TIME_THRESHOLD = game.CUE_TIME + game.WAIT_TIME
    game.FEEDBACK_TIME_THRESHOLD = game.WAIT_TIME_THRESHOLD + game.FEEDBACK_TIME
    game.TRIAL_TIME = game.FEEDBACK_TIME_THRESHOLD + game.ITI_TIME

    game.CUE_TIME_LIMIT = game.TRIAL_TIME - game.CUE_TIME
    game.WAIT_TIME_LIMIT = game.ITI_TIME + game.FEEDBACK_TIME
    game.FEEDBACK_TIME_LIMIT = game.ITI_TIME

    game.FEEDBACK_UPDATE_TIME = game.TR_TIME*game.CONFIG['feedback-update-trs']

    # networking
    game.NETWORK_TARGET = 'http://'+str(game.CONFIG['server-ip'])+':'+str(game.CONFIG['server-port'])+'/'

    # screen dimensions 
    game.SCREEN_DIMS = [1024, 768]
    game.SCREEN_DISTANCE = 136 # cm
    game.SCREEN_WIDTH = 85.7 # cm
    game.FIXATION_DIAMETER = game.CONFIG['fixation-diameter']
    game.SCORE_DIAMETER_MIN = game.CONFIG['min-feedback-diameter']
    game.SCORE_DIAMETER_MAX = game.CONFIG['max-feedback-diameter']
    game.SCORE_DIAMETER_MAX_INDICATOR = game.SCORE_DIAMETER_MAX + game.CONFIG['feedback-indicator-diameter']
    game.SCORE_DIAMETER_MIN_INDICATOR = game.SCORE_DIAMETER_MIN - game.CONFIG['feedback-indicator-diameter']
    game.SCORE_DIAMETER_RANGE = game.SCORE_DIAMETER_MAX - game.SCORE_DIAMETER_MIN

    # colors
    game.SCORE_COLOR = (40,110,40)
    game.BG_COLOR = -1
    game.FIXATION_COLOR = -0.4

def generate_variables(game, mode):
    # game logic
    game.run_trials = False
    game.show_feedback = False
    game.show_cue = False
    game.feedback_score_history = [0]
    game.feedback_calc_bool = False

    # input/output
    io=launchHubServer()
    game.keyboard = io.devices.keyboard
    game.monitor = monitors.Monitor('projector',
                                    width=game.SCREEN_WIDTH,
                                    distance=game.SCREEN_DISTANCE)
    game.monitor.setSizePix(game.SCREEN_DIMS)
    game.screen = visual.Window(game.SCREEN_DIMS,
                                monitor=game.monitor,
                                fullscr=game.FULLSCREEN,
                                units='deg',
                                color=game.BG_COLOR, colorSpace='rgb')

    # visual stims
    game.max_score_circ = visual.Circle(game.screen, edges=64,
        lineWidth=1, lineColor=game.BG_COLOR, lineColorSpace='rgb', fillColor=game.SCORE_COLOR, fillColorSpace='rgb255',
        radius=0.5*game.SCORE_DIAMETER_MAX_INDICATOR,
        interpolate=True)
    game.max_score_circ_mask = visual.Circle(game.screen, edges=64,
        lineWidth=1, lineColor=game.BG_COLOR, lineColorSpace='rgb', fillColor=game.BG_COLOR, fillColorSpace='rgb',
        radius=0.5*game.SCORE_DIAMETER_MAX,
        interpolate=True)
    game.score_circ = visual.Circle(game.screen, edges=64,
        lineWidth=1, lineColor=game.SCORE_COLOR, lineColorSpace='rgb255', fillColor=game.SCORE_COLOR, fillColorSpace='rgb255',
        radius=0.5*game.SCORE_DIAMETER_MIN,
        interpolate=True)
    game.score_circ_mask = visual.Circle(game.screen, edges=64,
        lineWidth=1, lineColor=game.BG_COLOR, lineColorSpace='rgb', fillColor=game.BG_COLOR, fillColorSpace='rgb',
        radius=0.5*game.SCORE_DIAMETER_MIN_INDICATOR,
        interpolate=True)
    game.fixation = visual.Circle(game.screen,
        lineWidth=1, lineColor=game.FIXATION_COLOR, lineColorSpace='rgb', fillColor=game.FIXATION_COLOR, fillColorSpace='rgb',
        radius=0.5*game.FIXATION_DIAMETER,
        interpolate=True)
    game.cue = visual.Circle(game.screen,
        lineWidth=1, lineColor=game.SCORE_COLOR, lineColorSpace='rgb255', fillColor=game.SCORE_COLOR, fillColorSpace='rgb255',
        radius=0.5*game.FIXATION_DIAMETER,
        interpolate=True)

    game.splash_msg = visual.TextStim(game.screen,
        text=game.SPLASH_MSG_BASE.format(current=str(1),
                                         total=str(game.RUNS)))

    game.debug_msg = visual.TextStim(game.screen,
                                     text='',
                                     pos=(-0.25,-0.25))

    game.debug_msg_base = ('last volume: {last_vol}\n'
                           +'last classifier: {last_clf}\n')

    # timers
    game.zscore_clock = core.Clock()
    game.trial_clock = core.Clock()
    game.feedback_update_clock = core.Clock()
    game.trial_count = 0
    game.run_count = -1
