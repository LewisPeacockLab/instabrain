from psychopy.iohub import launchHubServer
from psychopy import visual, monitors
from psychopy import core
import multiprocessing as mp
import numpy as np
import os, yaml
import SocketServer
import game_server as gs

def generate_constants(game, subject_id, session_num):
    with open('demo_game_config.yml') as f:
        game.CONFIG = yaml.load(f)

    # game parameters
    game.SUBJECT_ID = subject_id
    game.SESSION_NUM = session_num
    game.SUBJECT_DIR = 'datasets/' + game.SUBJECT_ID
    if not os.path.exists(os.path.normpath(game.SUBJECT_DIR)):
        os.mkdir(os.path.normpath(game.SUBJECT_DIR))
    tr_file_name = game.SUBJECT_DIR+'/'+game.SUBJECT_ID+'-sesh'+str(game.SESSION_NUM)+'.txt'
    game.TR_FILE = open(os.path.normpath(tr_file_name),'w')
    game.FULLSCREEN = game.CONFIG['fullscreen']
    game.RUNS = game.CONFIG['runs']
    game.TRS_PER_RUN = game.CONFIG['feedback-trs']
    game.REWARD_MSG_BASE = 'You earned ${run_reward:.2f} last run,\nand have earned ${total_reward:.2f} total.'
    game.SPLASH_MSG_BASE = 'Ready for Run {current} of {total}\nHold any key to continue...'
    game.QUIT_MSG = 'Done!\nHold any key to quit...'

    # timing
    game.TR_TIME = game.CONFIG['tr-time']
    game.FEEDBACK_OFFSET_TIME = game.CONFIG['feedback-offset-time']
    game.ZSCORE_TIME = game.TR_TIME*game.CONFIG['zscore-trs']
    game.SELF_PACE_START_TIME = 1.0

    game.FEEDBACK_UPDATE_TIME = game.TR_TIME*game.CONFIG['feedback-update-trs']

    # screen dimensions 
    game.SCREEN_DIMS = [1024, 768]
    game.SCREEN_DISTANCE = 136; game.SCREEN_WIDTH = 85.7 # cm
    # game.SCREEN_DIMS = [1920, 1080]
    # game.SCREEN_DISTANCE = 136 # cm
    # game.SCREEN_WIDTH = 160.7 # cm
    game.FIXATION_DIAMETER = game.CONFIG['fixation-diameter']
    game.SCORE_DIAMETER_MIN = game.CONFIG['min-feedback-diameter']
    game.SCORE_DIAMETER_MAX = game.CONFIG['max-feedback-diameter']
    game.SCORE_DIAMETER_MAX_INDICATOR = game.SCORE_DIAMETER_MAX + game.CONFIG['feedback-indicator-diameter']
    game.SCORE_DIAMETER_MIN_INDICATOR = game.SCORE_DIAMETER_MIN - game.CONFIG['feedback-indicator-diameter']
    game.SCORE_DIAMETER_RANGE = game.SCORE_DIAMETER_MAX - game.SCORE_DIAMETER_MIN

    # colors
    game.SCORE_COLOR = (40,110,40)
    game.BG_COLOR = -0.8
    game.TEXT_COLOR = 0.8
    game.FIXATION_COLOR = -0.4

    # key presses
    game.keydown = [False,False,False,False,False]
    game.key_codes = [u'9',
                      u'8',
                      u'7',
                      u'6',
                      u'4']

    # reward
    game.MAX_EXPERIMENT_REWARD = game.CONFIG['max-experiment-reward']
    game.MAX_TRIAL_REWARD = game.MAX_EXPERIMENT_REWARD/float(game.RUNS)/float(game.TRS_PER_RUN)

def generate_variables(game):
    # game logic
    game.show_feedback = False
    game.show_cue = False
    game.header_written = False
    game.feedback_score_history = [0]
    game.trial_stage = 'splash' # splash, zscore, cue, wait, feedback, iti
    game.feedback_status = 'idle' # idle, calculated

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

    game.reward_msg = visual.TextStim(game.screen,
        text='',
        pos=(0,2.5))

    game.debug_msg = visual.TextStim(game.screen,
        text='',
        pos=(-0.25,-0.25))

    game.debug_msg_base = ('last volume: {last_vol}\n'
                           +'last classifier: {last_clf}\n')

    # timers
    game.tr_clock = core.Clock()
    game.feedback_update_clock = core.Clock()
    game.self_pace_start_clock = core.Clock()
    game.tr_count = 0
    game.trial_count = 0
    game.run_count = -1
    game.begin_wait_time = 0

    # reward
    game.run_reward_history = []
    game.total_reward = 0

    # networking
    game.shutdown_url = game.CONFIG['shutdown-url']
    # game.target_class = mp.Value('i', 0)
    game.target_class = mp.Value('i', 1)
    game.feedback_calc_trial = mp.Value('i', -1)
    game.num_classes = game.CONFIG['num-classes']
    game.clf_outs = mp.Array('d', game.num_classes)
    game.server_running_bool = mp.Value('i', 1)
    game.server_process = mp.Process(target = gs.start_server,
                                     args = (game.target_class,
                                        game.feedback_calc_trial,
                                        game.clf_outs,))
    game.server_process.start()
