import numpy as np
import fileutils as fu
from psychopy import core, visual

def run_rest(game):
    game.show_grating = False
    game.show_target = False
    game.show_score = False

def run_trial(game):
    game.show_grating = True
    if game.target_visible_trial:
        game.show_target = True 
    else:
        game.show_target = False
    game.show_score = True
    frame_based_updates(game)
    game.target.update(game.grating.ori)

def run_score(game):
    game.show_grating = False
    game.show_target = True
    game.show_score = False

def frame_based_updates(game):
    if game.MODE == 'sim':
        # pass
        fu.timed_frame_record_sim(game, game.f_timed_frame)
    else:
        fu.timed_frame_record(game, game.f_timed_frame)
    new_last_time = game.global_clock.getTime()
    time_passed = new_last_time - game.last_time
    game.last_time = new_last_time
    if game.direction == 'ccw':
        game.grating.ori -= time_passed*game.ROTATION_SPEED
    elif game.direction == 'cw':
        game.grating.ori += time_passed*game.ROTATION_SPEED

def blink_based_updates(game):
    new_phase = np.random.random()
    game.grating.phase = new_phase
    game.target_grating.phase = new_phase
    game.grating_blink_visible = not(game.grating_blink_visible)

def tr_based_updates_localizer(game):
    update_score(game, score=np.random.random())

def trial_based_updates_localizer(game):
    next_loc = game.LOC_LIST[game.run_count, game.trial_count]
    if next_loc < 0:
        game.grating_visible = False
    else:
        game.grating_visible = True
        game.grating.ori = 90 - game.ANGLE_BETWEEN_TARGETS*next_loc

def tr_based_updates_game(game):
    game.target.error_metric = np.mean(
            game.target.error_metric_conv_sampled_buffer[
                -game.TRS_SAMPLES_DICT[game.tr]:])
    score = min(max(0,game.target.error_metric),1)
    if game.MODE != 'sim':
        update_score(game, score)
    if check_error_metric(game):
        game.target_reached = True
        game.success_show_clock.reset()
        game.success_show_clock.add(game.SUCCESS_SHOW_TIME)
        if game.MODE == 'sim':
            fu.timed_trial_record_sim(game, game.f_timed_trial)
        else:
            fu.timed_trial_record(game, game.f_timed_trial)

def check_error_metric(game):
    score = np.mean(game.target.error_metric_conv_sampled_buffer[
                    -game.REACH_SUCCESS_TIME:])
    if score >= game.SUCCESS_SCORE:
        return True
    else:
        return False

# def tr_based_updates(game):
#     if (check_error_metric(game)
#             and game.exp_type == 'timed'):
#         game.cursor.has_left = False
#         game.cursor.start_ready = False
#         game.timers['reset_hold'].reset()
#         fu.timed_trial_record(game, game.f_timed_trial)
#         game.trial_count += 1
#         trial_type_based_updates(game)
#     else:
#         game.target.error_metric = np.mean(
#             game.target.error_metric_conv_sampled_buffer[
#                 -game.TRS_SAMPLES_DICT[game.tr]:])
#         game.timers['tr'].time_limit_hit = False
#         if game.exp_type == 'block':
#             if game.block_tr_count < len(game.block_nfb_buffer):
#                 game.block_nfb_buffer[game.block_tr_count] = (
#                     game.target.error_metric)
#             game.playback_nfb_points[game.block_tr_count] = (
#                 game.target.error_metric)
#             game.block_tr_count += 1

def update_score(game, score=.5):
    game.score_circ.size = (game.SCORE_SIZE_MIN
                            + score*game.SCORE_SIZE_RANGE)

def signal_based_updates(game):
    game.target.error_metric_conv_sampled_buffer = np.roll(
        game.target.error_metric_conv_sampled_buffer, -1)
    game.target.error_metric_conv_sampled_buffer[-1] = (np.mean(
        game.target.error_metric_conv_buffer[-game.SAMPLE_FRAMES:])
        + game.noise_std*np.random.normal())

def reset_grating(game, orientation):
    game.grating.ori = orientation
    game.grating_blink_visible = True
    game.blink_clock.reset()
    game.blink_clock.add(game.BLINK_TIME)

def reset_for_next_trial(game):
    game.trial_started = False
    game.target_reached = False
    game.begin_rest_clock.reset()
    game.begin_rest_clock.add(game.BEGIN_REST_TIME)
    game.trial_count += 1
    trial_type_based_updates(game)

def start_trial(game):
    game.trial_started = True

    game.grating.ori = game.DEFAULT_ORIENTATION
    if game.MODE != 'sim':
        update_score(game, game.target.MIN_ERROR_METRIC)

    game.signal_clock.reset() 
    game.signal_clock.add(game.SAMPLE_PERIOD)

    game.tr_clock.reset() 
    game.tr_clock.add(game.tr)

    game.trial_clock.reset() 

def start_run(game):
    game.trial_started = False
    game.trial_count = 0
    game.last_time = 0
    game.global_clock.reset() 
    reset_grating(game, game.DEFAULT_ORIENTATION)

    game.signal_clock.reset() 
    game.signal_clock.add(game.SAMPLE_PERIOD)

    game.tr_clock.reset() 
    game.tr_clock.add(game.tr)

    game.trial_clock.reset() 
    game.trial_clock.add(game.BEGIN_REST_TIME)

    game.begin_rest_clock.reset()
    game.begin_rest_clock.add(game.BEGIN_REST_TIME)

    trial_based_updates_localizer(game)
    game.run_count += 1
    game.splash_msg.text = game.SPLASH_MSG_BASE.format(current=str(game.run_count+2),
                                                       total=str(game.RUNS))
    game.run_trials = True


def trial_type_based_updates(game):
    # trial types:
    # 0) visible, no hrf
    # 1) invisible, no hrf
    # 2) visible, hrf
    # 3) invisible, hrf, snr=high, tr=1
    # 4) invisible, hrf, snr=high, tr=6
    # 5) invisible, hrf, snr=low, tr=1
    # 6) invisible, hrf, snr=low, tr=6

    if np.mod(game.trial_count, game.NUM_TRIALS) == 0:
        game.trial_type_count += 1
        game.trial_count = 0
        game.run_trials = False
        if (game.trial_type_count == 0
                or game.trial_type_count == 2):
            game.target.target_list = game.target.SHORT_TARGET_LIST
        else:
            game.target.target_list = game.target.LONG_TARGET_LIST
        game.target.shuffle_target_list()

    if game.trial_type_count > game.TRIAL_TYPES:
        game.quit()

    if (game.trial_type_count == 0
       or game.trial_type_count == 2):
        game.target_visible_trial = True
        game.NUM_TRIALS = game.VISIBLE_TRIALS
    else:
        game.target_visible_trial = False
        game.NUM_TRIALS = game.INVISIBLE_TRIALS

    if (game.trial_type_count == 0
       or game.trial_type_count == 1):
        game.target.set_fb_mode('impulse')
    else:
        game.target.set_fb_mode('hrf')

    if (game.trial_type_count == 5
            or game.trial_type_count == 6):
        game.noise_std = game.NOISE_STD_LIST[1]
    else:
        game.noise_std = game.NOISE_STD_LIST[0]

    if (game.trial_type_count == 4
            or game.trial_type_count == 6):
        game.tr = game.TR_LIST[1]
    else:
        game.tr = game.TR_LIST[0]
