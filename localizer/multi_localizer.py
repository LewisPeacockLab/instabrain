import multiprocessing as mp
from fingfind_localizer import *

def generate_all_subj_clfs():
    pool = mp.Pool()
    subjs = ['ff001','ff002','ff003','ff004','ff005']
    for subj in subjs:
        pool.apply_async(func = generate_subj_clfs,
            args = (subj,))

def generate_all_subj_realtime_lims_within_sess():
    pool = mp.Pool()
    subjs = ['ff001','ff002','ff003','ff004','ff005']
    for subj in subjs:
        pool.apply_async(func = generate_subj_realtime_lims_within_sess,
            args = (subj,))

def generate_all_subj_between_session():
    pool = mp.Pool()
    subjs = ['ff001','ff002','ff003','ff004','ff005']
    for subj in subjs:
        pool.apply_async(func = generate_subj_between_session,
            args = (subj,))       

def generate_all_subj_clf_out():
    pool = mp.Pool()
    subjs = ['ff001','ff002','ff003','ff004','ff005']
    for subj in subjs:
        pool.apply_async(func = generate_subj_clf_out,
            args = (subj,))     

def generate_all_compare_within_to_between():
    pool = mp.Pool()
    subjs = ['ff001','ff002','ff003','ff004','ff005']
    for subj in subjs:
        pool.apply_async(func = generate_subj_compare_within_to_between,
            args = (subj,))     

def generate_all_time_points_between_sess():
    pool = mp.Pool()
    subjs = ['ff002','ff003','ff004','ff005','ff006']
    for subj in subjs:
        pool.apply_async(func = generate_subj_time_points_between_sess,
            args = (subj,))     

def generate_subj_time_points_between_sess(subj_id):
    loc = FingfindLocalizer(subj_id)
    loc.test_time_points_between_session()

def generate_subj_clfs(subj_id):
    loc = FingfindLocalizer(subj_id)
    loc.train_and_save_multi_session_clf()

def generate_subj_realtime_lims_within_sess(subj_id):
    loc = FingfindLocalizer(subj_id)
    loc.fingfind_realtime_limitations_within_session()

def generate_subj_between_session(subj_id):
    loc = FingfindLocalizer(subj_id)
    loc.test_between_session()

def generate_subj_clf_out(subj_id):
    loc = FingfindLocalizer(subj_id)
    loc.fingfind_roi_corrs()

def generate_subj_compare_within_to_between(subj_id):
    loc = FingfindLocalizer(subj_id)
    loc.fingfind_compare_within_to_between()
