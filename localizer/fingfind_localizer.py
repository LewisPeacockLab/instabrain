import numpy as np
import yaml,os,glob,pickle,copy
import subprocess,shlex,copy
from mvpa2.clfs.smlr import SMLR
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.base.dataset import vstack
from mvpa2.mappers.zscore import zscore
from mvpa2.mappers.detrend import poly_detrend
from mvpa2.mappers.fx import mean_group_sample
from mvpa2.measures.base import CrossValidation
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.clfs.smlr import SMLR
from scipy.signal import detrend
from mvpa2.datasets.eventrelated import find_events
from mvpa2.datasets.eventrelated import fit_event_hrf_model

class FingfindLocalizer(object):
    def __init__(self, subject_id):
        with open('fingfind_localizer_config.yml') as f:
            self.CONFIG = yaml.load(f)
        
        # directories
        self.subject_id = subject_id
        self.fs_subject_id = self.subject_id+'fs'
        self.base_dir = self.CONFIG['SUBJECT_DIR']+'/'+self.subject_id
        self.ref_dir = self.base_dir+'/ref'
        self.rfi_img = self.ref_dir+'/rfi.nii'
        self.rai_img = self.CONFIG['SUBJECT_DIR']+'/pycortex/'+self.subject_id+'/anatomicals/raw.nii.gz'
        self.reg_mat = self.ref_dir+'/rai2rfi.dat'
        self.fsl_reg_mat = self.ref_dir+'/rai2rfi.mat'
        self.base_bold_dir = self.base_dir+'/bold'
        self.bold_dir_sess1 = self.base_bold_dir+'/sess1'
        self.bold_dir_sess2 = self.base_bold_dir+'/sess2'

        # timing
        self.tr = self.CONFIG['TR']
        self.vols_per_run = self.CONFIG['VOLS_PER_RUN']
        self.num_runs = self.CONFIG['NUM_RUNS']
        self.trs_per_trial = self.CONFIG['TRS_PER_TRIAL']
        self.trial_feature_trs = self.CONFIG['TRIAL_FEATURE_TRS']
        self.presses_per_trial = self.CONFIG['PRESSES_PER_TRIAL']
        self.zscore_trs = self.CONFIG['ZSCORE_TRS']
        self.trials_per_run = (self.vols_per_run-self.zscore_trs)/self.trs_per_trial

        # labels
        self.behav_data_sess1 = np.loadtxt(self.ref_dir+'/ft-data-sess1.txt',delimiter=',',skiprows=1)
        try:
            self.behav_data_sess2 = np.loadtxt(self.ref_dir+'/ft-data-sess2.txt',delimiter=',',skiprows=1)
        except:
            self.behav_data_sess2 = self.behav_data_sess1
        self.trial_data_sess1 = self.behav_data_sess1[::self.presses_per_trial,:]
        self.trial_data_sess2 = self.behav_data_sess2[::self.presses_per_trial,:]
        self.trial_targets_sess1 = self.trial_data_sess1[:,0]
        self.trial_targets_sess2 = self.trial_data_sess2[:,0]
        self.trial_chunks = self.trial_data_sess1[:,-1]
        self.n_class = np.unique(self.trial_targets_sess1).size

        # select session
        self.select_session(1)

        # classifier
        self.clf = SMLR()

    def preprocessing(self):
        self.create_rfi()
        self.register_2_rfi()
        self.generate_sensorimotor_masks()
        self.select_session(1)
        self.motion_correct()
        self.select_session(2)
        self.motion_correct()
        self.select_session(1)

    def extract_features(self, roi_name=None, hemi='lh', zs_type='all', detrend='all',
            zs_proportion=1, trial_feature_trs='from_config'):
        datasets = []
        if trial_feature_trs == 'from_config':
            self.trial_feature_trs = self.CONFIG['TRIAL_FEATURE_TRS']
        else:
            self.trial_feature_trs = trial_feature_trs
        self.tr_targets = np.tile(self.trial_targets,(self.trs_per_trial,1)).flatten('F')
        self.tr_chunks = np.tile(self.trial_chunks,(self.trs_per_trial,1)).flatten('F')
        if roi_name == None:
            mask = None
        else:
            mask = self.ref_dir+'/mask_'+hemi+'_'+roi_name+'.nii'
        for run in range(self.num_runs):
            run_tr_targets = np.append(-2*np.ones(int((1-zs_proportion)*self.zscore_trs)), self.tr_targets[self.tr_chunks==run])
            run_tr_targets = np.append(-1*np.ones(int(zs_proportion*self.zscore_trs)), run_tr_targets)
            run_tr_chunks = np.append(run*np.ones(self.zscore_trs), self.tr_chunks[self.tr_chunks==run])
            run_dataset = fmri_dataset(self.bold_dir+'/rrun-'+str(run+1).zfill(3)+'.nii',
                mask=mask,
                targets=run_tr_targets,
                chunks=run_tr_chunks)
            if detrend == 'realtime':
                run_dataset = self.realtime_detrend(run_dataset)
            elif detrend == 'all':
                poly_detrend(run_dataset, polyord=1)
            if zs_type == 'baseline':
                zscore(run_dataset, chunks_attr='chunks', param_est=('targets',[-1]))
            elif zs_type == 'realtime':
                run_dataset = self.realtime_zscore(run_dataset)
            elif zs_type == 'active':
                zscore(run_dataset, chunks_attr='chunks', param_est=('targets',range(self.n_class)))
            elif zs_type == 'all':
                zscore(run_dataset, chunks_attr='chunks')
            datasets.append(run_dataset)
        self.fmri_data = vstack(datasets, a=0)

        self.active_trs = np.zeros(self.trs_per_trial)
        self.active_trs[self.trial_feature_trs[0]-1:self.trial_feature_trs[1]] = 1
        self.active_trs = np.tile(self.active_trs,int(self.trials_per_run))
        self.active_trs = np.append(np.zeros(self.zscore_trs), self.active_trs)
        self.active_trs = np.tile(self.active_trs,int(self.num_runs))
        self.trial_regressor = -1*np.ones(self.vols_per_run*self.num_runs)
        for run in range(self.num_runs):
            begin_trial = int(run*self.trials_per_run)
            end_trial = int((run+1)*self.trials_per_run)
            self.trial_regressor[(run*self.vols_per_run+self.zscore_trs):((run+1)*self.vols_per_run)] = (
                np.tile(range(begin_trial,end_trial),(self.trs_per_trial,1)).flatten('F'))
        self.fmri_data.sa['active_trs'] = self.active_trs
        self.fmri_data.sa['trial_regressor'] = self.trial_regressor
        self.fmri_data = self.fmri_data[self.fmri_data.sa.active_trs==1]
        trial_mapper = mean_group_sample(['targets','trial_regressor'])
        self.fmri_data = self.fmri_data.get_mapped(trial_mapper)

    def realtime_detrend(self, run_dataset):
        run_data = copy.deepcopy(run_dataset.samples)
        out_data = np.zeros(run_data.shape)
        out_data[:self.zscore_trs] = detrend(run_data[:self.zscore_trs],0)
        for tr in range(self.zscore_trs,self.vols_per_run):
            out_data[tr] = detrend(run_data[:tr+1],0)[-1]
        run_dataset.samples = out_data
        return run_dataset

    def realtime_zscore(self, run_dataset):
        run_data = copy.deepcopy(run_dataset.samples)
        out_data = np.zeros(run_data.shape)
        baseline_sigmas = np.sqrt(np.var(run_data[:self.zscore_trs],0))
        out_data[:self.zscore_trs] = run_data[:self.zscore_trs]/baseline_sigmas
        for tr in range(self.zscore_trs,self.vols_per_run):
            tr_sigmas = np.sqrt(np.var(run_data[:tr+1],0))
            out_data[tr] = (run_data[:tr+1]/tr_sigmas)[-1]
        run_dataset.samples = out_data
        return run_dataset

    def select_session(self, session=1):
        if session == 1:
            self.trial_targets = self.trial_targets_sess1
            self.bold_dir = self.bold_dir_sess1
        elif session == 2:
            self.trial_targets = self.trial_targets_sess2
            self.bold_dir = self.bold_dir_sess2
        self.session = session

    def fingfind_realtime_limitations_within_session(self, rois=['m1','s1'], sessions=[1,2]):
        # self.test_training_data(rois,sessions)
        self.test_time_points(rois,sessions)
        # self.test_between_session()
        # self.test_zscore(rois,sessions)
        # self.test_detrend(rois,sessions)

    def fingfind_compare_within_to_between(self, rois=['m1--','m1-','m1','m1s1','s1','s1-','s1--']):
        sessions = [1,2]
        time_points = [[6,8]]
        self.test_training_data(rois,sessions,leave_out_runs=range(1,2))
        self.test_between_session(rois)

    def test_training_data(self, rois, sessions, leave_out_runs=range(1,8)):
        zs_type = 'baseline'
        zs_proportion = 1
        detrend = 'realtime'
        for session in sessions:
            self.select_session(session)
            for roi in rois:
                self.df = self.create_empty_cv_df(len(leave_out_runs),self.num_runs)
                self.extract_features(roi_name=roi, hemi='lh', zs_type=zs_type, detrend=detrend,
                    zs_proportion=zs_proportion, trial_feature_trs='from_config')
                for idx, leave_out_run in enumerate(leave_out_runs):
                    self.cross_validate(leave_out_runs=leave_out_run)
                    self.df.loc[idx,['roi_name','zs_type','zs_type_2','leave_out_runs','detrend',
                        'zs_proportion','trial_feature_trs','subj','session']]=[roi,zs_type,'none',
                        leave_out_run,detrend,zs_proportion,str(self.trial_feature_trs),self.subject_id,session]
                    self.df.loc[idx,'clf_acc']=np.array(self.cv_results).T
                self.df.to_csv(self.subject_id+'_'+roi+'_sess'+str(session)+'_training_runs.csv')

    def test_time_points(self, rois, sessions, time_windows=[[1,3],[2,4],[3,5],[4,6],[5,7],[6,8]]):
        zs_type = 'baseline'
        zs_proportion = 1
        detrend = 'realtime'
        leave_out_run = 1
        for session in sessions:
            self.select_session(session)
            for roi in rois:
                self.df = self.create_empty_cv_df(len(time_windows),self.num_runs)
                for idx, time_window in enumerate(time_windows):
                    self.extract_features(roi_name=roi, hemi='lh', zs_type=zs_type, detrend=detrend,
                        zs_proportion=zs_proportion, trial_feature_trs=time_window)
                    self.cross_validate()
                    self.df.loc[idx,['roi_name','zs_type','zs_type_2','leave_out_runs','detrend',
                        'zs_proportion','trial_feature_trs','subj','session']]=[roi,zs_type,'none',
                        leave_out_run,detrend,zs_proportion,str(self.trial_feature_trs),self.subject_id,session]
                    self.df.loc[idx,'clf_acc']=np.array(self.cv_results).T
                self.df.to_csv(self.subject_id+'_'+roi+'_sess'+str(session)+'_time_points.csv')

    def test_zscore(self, rois, sessions, zs_proportions=[.25,.5,.75,1,1,1]):
        zs_types = ['baseline','baseline','baseline','baseline','baseline','baseline',
            'baseline','baseline','baseline','baseline','baseline','none','realtime','all']
        zs_proportions = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.75,1,0,1,1]
        detrend = 'realtime'
        leave_out_run = 1
        for session in sessions:
            self.select_session(session)
            for roi in rois:
                self.df = self.create_empty_cv_df(len(zs_proportions),self.num_runs)
                for idx, zs_proportion in enumerate(zs_proportions):
                    zs_type = zs_types[idx]
                    self.extract_features(roi_name=roi, hemi='lh', zs_type=zs_type, detrend=detrend,
                        zs_proportion=zs_proportion, trial_feature_trs='from_config')
                    self.cross_validate()
                    self.df.loc[idx,['roi_name','zs_type','zs_type_2','leave_out_runs','detrend',
                        'zs_proportion','trial_feature_trs','subj','session']]=[roi,zs_type,'none',
                        leave_out_run,detrend,zs_proportion,str(self.trial_feature_trs),self.subject_id,session]
                    self.df.loc[idx,'clf_acc']=np.array(self.cv_results).T
                self.df.to_csv(self.subject_id+'_'+roi+'_sess'+str(session)+'_zscore.csv')

    def test_detrend(self, rois, sessions, detrends = ['none','realtime','all']):
        zs_type = 'baseline'
        zs_proportion = 1
        leave_out_run = 1
        for session in sessions:
            self.select_session(session)
            for roi in rois:
                self.df = self.create_empty_cv_df(len(detrends),self.num_runs)
                for idx, detrend in enumerate(detrends):
                    self.extract_features(roi_name=roi, hemi='lh', zs_type=zs_type, detrend=detrend,
                        zs_proportion=zs_proportion, trial_feature_trs='from_config')
                    self.cross_validate()
                    self.df.loc[idx,['roi_name','zs_type','zs_type_2','leave_out_runs','detrend',
                        'zs_proportion','trial_feature_trs','subj','session']]=[roi,zs_type,'none',
                        leave_out_run,detrend,zs_proportion,str(self.trial_feature_trs),self.subject_id,session]
                    self.df.loc[idx,'clf_acc']=np.array(self.cv_results).T
                self.df.to_csv(self.subject_id+'_'+roi+'_sess'+str(session)+'_detrend.csv')

    def create_empty_cv_df(self, conditions_per_cv, cv_folds):
        import pandas as pd
        return pd.DataFrame(
            columns=['clf_acc','roi_name','zs_type','zs_type_2','leave_out_runs','detrend',
                'zs_proportion','trial_feature_trs','subj','session'],
            index=np.repeat(range(conditions_per_cv),cv_folds))

    def test_between_session(self, rois=['m1','s1']):
        # zs_types_sess1 = ['none','baseline','realtime','active','all','baseline','realtime','active','all']
        # zs_types_sess2 = ['none','baseline','baseline','baseline','baseline','realtime','realtime','realtime','realtime']
        # zs_proportions = [1,1,1,1,1,1,1,1,1]
        # zs_types_sess1 = ['baseline']
        # zs_types_sess2 = ['baseline']
        # zs_proportions = [1]
        zs_types = ['baseline','baseline','baseline','baseline','baseline','baseline','baseline','baseline','baseline',
            'baseline','baseline','baseline','baseline','none','realtime','all']
        zs_proportions = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.75,1,1,1,0,1,1]
        zs_types_sess1 = zs_types
        zs_types_sess2 = zs_types
        zs_proportions_sess1 = zs_proportions
        zs_proportions_sess2 = zs_proportions
        detrends = ['realtime','realtime','realtime','realtime','realtime','realtime','realtime','realtime','realtime',
            'realtime','realtime','none','all','realtime','realtime','realtime']
        leave_out_run = 0
        for sessions in [[1,2],[2,1]]:
            for roi in rois:
                self.df = self.create_empty_cv_df(len(zs_types_sess1),1)
                for idx, zs_type_sess1 in enumerate(zs_types_sess1):
                    zs_type_sess2 = zs_types_sess2[idx]
                    self.select_session(sessions[0])
                    self.extract_features(roi_name=roi, hemi='lh', zs_type=zs_type_sess1, detrend=detrends[idx],
                        zs_proportion=zs_proportions_sess1[idx], trial_feature_trs='from_config')
                    self.train_classifier()
                    self.select_session(sessions[1])
                    self.extract_features(roi_name=roi, hemi='lh', zs_type=zs_type_sess2, detrend=detrends[idx],
                        zs_proportion=zs_proportions_sess2[idx], trial_feature_trs='from_config')
                    predictions = self.clf.predict(self.fmri_data)
                    cross_sess_accuracy = np.mean(predictions==self.fmri_data.sa.targets)
                    self.df.loc[idx,['roi_name','zs_type','zs_type_2','leave_out_runs','detrend',
                        'zs_proportion','trial_feature_trs','subj','session']]=[roi,zs_type_sess1,zs_type_sess2,
                        leave_out_run,detrends[idx],zs_proportions_sess1[idx],str(self.trial_feature_trs),self.subject_id,sessions[0]]
                    self.df.loc[idx,'clf_acc']=cross_sess_accuracy
                self.df.to_csv(self.subject_id+'_'+roi+'_sess'+str(sessions[0])+'_between_session.csv')

    def fingfind_roi_corrs(self, rois=['m1--','m1-','m1','m1s1','s1','s1-','s1--']):
        zs_types_sess1 = ['baseline']
        zs_types_sess2 = ['baseline']
        # rois = ['m1','s1']
        # zs_types_sess1 = ['baseline','active','all']
        # zs_types_sess2 = ['baseline','baseline','baseline']
        zs_proportion = 1
        detrend = 'realtime'
        conditions = len(zs_types_sess1)
        for sessions in [[1,2],[2,1]]:
            for roi in rois:
                self.df = self.create_empty_clf_out_df(conditions=conditions,trials=self.num_runs*self.trials_per_run)
                for idx in range(conditions):
                    self.df.loc[idx,['roi_name','zs_type','zs_type_2','detrend',
                        'zs_proportion','trial_feature_trs','subj','session']]=[roi,zs_types_sess1[idx],zs_types_sess2[idx],
                        detrend,zs_proportion,str(self.trial_feature_trs),self.subject_id,sessions[0]]
                    self.select_session(sessions[0])
                    self.extract_features(roi_name=roi, hemi='lh', zs_type=zs_types_sess1[idx], detrend=detrend,
                        zs_proportion=zs_proportion, trial_feature_trs='from_config')
                    self.train_classifier()
                    self.select_session(sessions[1])
                    self.extract_features(roi_name=roi, hemi='lh', zs_type=zs_types_sess2[idx], detrend=detrend,
                        zs_proportion=zs_proportion, trial_feature_trs='from_config')
                    predictions = self.clf.predict(self.fmri_data)
                    self.df.loc[idx,'target'] = self.fmri_data.sa.targets
                    self.df.loc[idx,'clf_out_0'] = self.clf.ca.estimates[:,0]
                    self.df.loc[idx,'clf_out_1'] = self.clf.ca.estimates[:,1]
                    self.df.loc[idx,'clf_out_2'] = self.clf.ca.estimates[:,2]
                    self.df.loc[idx,'clf_out_3'] = self.clf.ca.estimates[:,3]
                self.df.to_csv(self.subject_id+'_'+roi+'_sess'+str(sessions[0])+'_clf_out.csv')

    def test_time_points_between_session(self, rois=['m1','s1'], time_windows=[[1,3],[2,4],[3,5],[4,6],[5,7],[6,8]]):
        zs_type = 'baseline'
        zs_proportion = 1
        detrend = 'realtime'
        leave_out_run = 0
        for sessions in [[1,2],[2,1]]:
            for roi in rois:
                self.df = self.create_empty_cv_df(len(time_windows),1)
                for idx, time_window in enumerate(time_windows):
                    self.select_session(sessions[0])
                    self.extract_features(roi_name=roi, hemi='lh', zs_type=zs_type, detrend=detrend,
                        zs_proportion=zs_proportion, trial_feature_trs=time_windows[idx])
                    self.train_classifier()
                    self.select_session(sessions[1])
                    self.extract_features(roi_name=roi, hemi='lh', zs_type=zs_type, detrend=detrend,
                        zs_proportion=zs_proportion, trial_feature_trs=time_windows[idx])
                    predictions = self.clf.predict(self.fmri_data)
                    cross_sess_accuracy = np.mean(predictions==self.fmri_data.sa.targets)
                    self.df.loc[idx,['roi_name','zs_type','zs_type_2','leave_out_runs','detrend',
                        'zs_proportion','trial_feature_trs','subj','session']]=[roi,zs_type,zs_type,
                        leave_out_run,detrend,zs_proportion,str(self.trial_feature_trs),self.subject_id,sessions[0]]
                    self.df.loc[idx,'clf_acc']=cross_sess_accuracy
                self.df.to_csv(self.subject_id+'_'+roi+'_sess'+str(sessions[0])+'_between_session_time_points.csv')

    def create_empty_clf_out_df(self, conditions, trials):
        import pandas as pd
        return pd.DataFrame(
            columns=['target','clf_out_0','clf_out_1','clf_out_2','clf_out_3',
                'roi_name','zs_type','zs_type_2','detrend',
                'zs_proportion','trial_feature_trs','subj','session'],
            index=np.repeat(range(conditions),trials))

    def train_and_save_multi_session_clf(self, roi_name='m1s1', hemi='lh', zs_type='baseline', detrend='realtime',
            zs_proportion=1, trial_feature_trs='from_config'):
        self.extract_multi_session_features(roi_name=roi_name, hemi=hemi, zs_type=zs_type, detrend=detrend,
            zs_proportion=zs_proportion, trial_feature_trs=trial_feature_trs)
        self.train_classifier()
        self.save_classifier()

    def extract_multi_session_features(self, roi_name='m1s1', hemi='lh', zs_type='baseline', detrend='realtime',
            zs_proportion=1, trial_feature_trs='from_config'):
        datasets = []
        if trial_feature_trs == 'from_config':
            self.trial_feature_trs = self.CONFIG['TRIAL_FEATURE_TRS']
        else:
            self.trial_feature_trs = trial_feature_trs
        self.trial_targets = np.concatenate([self.trial_targets_sess1,self.trial_targets_sess2])
        self.trial_chunks = np.concatenate([self.trial_data_sess1[:,-1],self.trial_data_sess1[:,-1]+self.num_runs])
        self.tr_targets = np.tile(self.trial_targets,(self.trs_per_trial,1)).flatten('F')
        self.tr_chunks = np.tile(self.trial_chunks,(self.trs_per_trial,1)).flatten('F')
        if roi_name == None:
            mask = None
        else:
            mask = self.ref_dir+'/mask_'+hemi+'_'+roi_name+'.nii'

        self.bold_dir = self.bold_dir_sess1
        for run in range(self.num_runs):
            run_tr_targets = np.append(-2*np.ones(int((1-zs_proportion)*self.zscore_trs)), self.tr_targets[self.tr_chunks==run])
            run_tr_targets = np.append(-1*np.ones(int(zs_proportion*self.zscore_trs)), run_tr_targets)
            run_tr_chunks = np.append(run*np.ones(self.zscore_trs), self.tr_chunks[self.tr_chunks==run])
            run_dataset = fmri_dataset(self.bold_dir+'/rrun-'+str(run+1).zfill(3)+'.nii',
                mask=mask,
                targets=run_tr_targets,
                chunks=run_tr_chunks)
            if detrend == 'realtime':
                run_dataset = self.realtime_detrend(run_dataset)
            elif detrend == 'all':
                poly_detrend(run_dataset, polyord=1)
            if zs_type == 'baseline':
                zscore(run_dataset, chunks_attr='chunks', param_est=('targets',[-1]))
            elif zs_type == 'realtime':
                run_dataset = self.realtime_zscore(run_dataset)
            elif zs_type == 'active':
                zscore(run_dataset, chunks_attr='chunks', param_est=('targets',range(self.n_class)))
            elif zs_type == 'all':
                zscore(run_dataset, chunks_attr='chunks')
            datasets.append(run_dataset)

        self.bold_dir = self.bold_dir_sess2
        for run in range(self.num_runs,2*self.num_runs):
            run_tr_targets = np.append(-2*np.ones(int((1-zs_proportion)*self.zscore_trs)), self.tr_targets[self.tr_chunks==run])
            run_tr_targets = np.append(-1*np.ones(int(zs_proportion*self.zscore_trs)), run_tr_targets)
            run_tr_chunks = np.append(run*np.ones(self.zscore_trs), self.tr_chunks[self.tr_chunks==run])
            run_dataset = fmri_dataset(self.bold_dir+'/rrun-'+str(run-7).zfill(3)+'.nii',
                mask=mask,
                targets=run_tr_targets,
                chunks=run_tr_chunks)
            if detrend == 'realtime':
                run_dataset = self.realtime_detrend(run_dataset)
            elif detrend == 'all':
                poly_detrend(run_dataset, polyord=1)
            if zs_type == 'baseline':
                zscore(run_dataset, chunks_attr='chunks', param_est=('targets',[-1]))
            elif zs_type == 'realtime':
                run_dataset = self.realtime_zscore(run_dataset)
            elif zs_type == 'active':
                zscore(run_dataset, chunks_attr='chunks', param_est=('targets',range(self.n_class)))
            elif zs_type == 'all':
                zscore(run_dataset, chunks_attr='chunks')
            datasets.append(run_dataset)

        self.fmri_data = vstack(datasets, a=0)

        self.active_trs = np.zeros(self.trs_per_trial)
        self.active_trs[self.trial_feature_trs[0]-1:self.trial_feature_trs[1]] = 1
        self.active_trs = np.tile(self.active_trs,int(self.trials_per_run))
        self.active_trs = np.append(np.zeros(self.zscore_trs), self.active_trs)
        self.active_trs = np.tile(self.active_trs,int(2*self.num_runs))
        self.trial_regressor = -1*np.ones(self.vols_per_run*2*self.num_runs)
        for run in range(2*self.num_runs):
            begin_trial = int(run*self.trials_per_run)
            end_trial = int((run+1)*self.trials_per_run)
            self.trial_regressor[(run*self.vols_per_run+self.zscore_trs):((run+1)*self.vols_per_run)] = (
                np.tile(range(begin_trial,end_trial),(self.trs_per_trial,1)).flatten('F'))
        self.fmri_data.sa['active_trs'] = self.active_trs
        self.fmri_data.sa['trial_regressor'] = self.trial_regressor
        self.fmri_data = self.fmri_data[self.fmri_data.sa.active_trs==1]
        trial_mapper = mean_group_sample(['targets','trial_regressor'])
        self.fmri_data = self.fmri_data.get_mapped(trial_mapper)

    def train_classifier(self):
        self.clf.train(self.fmri_data)

    def apply_classifier(self, data):
        self.clf.predict(data)
        return self.clf.ca.estimates

    def cross_validate(self, leave_out_runs=1):
        partitioner = NFoldPartitioner(cvtype=leave_out_runs,
            count=np.size(np.unique(self.fmri_data.chunks)),
            selection_strategy='random')
        cvte = CrossValidation(self.clf, partitioner,
            errorfx=lambda p, t: np.mean(p == t))
        self.cv_results = cvte(self.fmri_data) 
            
    def save_classifier(self):
        self.clf.voxel_indices = self.fmri_data.fa.voxel_indices
        pickle.dump(self.clf,open(self.ref_dir+'/clf.p','wb'))

    def save_classifier_as_nifti(self):
        import nibabel as nib
        out_name = self.ref_dir+'/'+self.subject_id+'_clf_out'
        rfi_nii = nib.load(self.rfi_img)
        rfi_data = rfi_nii.get_data()
        ref_affine = rfi_nii.get_qform()
        ref_header = rfi_nii.header
        out_img = np.zeros(rfi_data.shape)
        max_abs_clf_weights = [voxel[np.where(np.abs(voxel)==np.max(np.abs(voxel)))][0] for voxel in self.clf.weights]
        for roi_idx, voxel in enumerate(self.fmri_data.fa.voxel_indices):
            out_img[voxel[0],voxel[1],voxel[2]] = max_abs_clf_weights[roi_idx]
        nib.save(nib.Nifti1Image(out_img, ref_affine, header=ref_header), out_name)

    def save_importance_map_as_nifti(self):
        import nibabel as nib
        out_name = self.ref_dir+'/importance_map'
        rfi_nii = nib.load(self.rfi_img)
        rfi_data = rfi_nii.get_data()
        ref_affine = rfi_nii.get_qform()
        ref_header = rfi_nii.header
        out_img = np.zeros(rfi_data.shape)
        out_fingers = []
        mean_patterns = np.zeros(self.clf.weights.shape)
        for finger in range(self.clf.weights.shape[1]):
            mean_patterns[:,finger] = np.mean(self.fmri_data[self.fmri_data.targets==finger],0)
            out_fingers.append(np.zeros(rfi_data.shape))
        importances = mean_patterns*self.clf.weights*((self.clf.weights*mean_patterns)>0)*np.sign(self.clf.weights)
        max_abs_importances = [voxel[np.where(np.abs(voxel)==np.max(np.abs(voxel)))][0] for voxel in importances]
        for roi_idx, voxel in enumerate(self.fmri_data.fa.voxel_indices):
            out_img[voxel[0],voxel[1],voxel[2]] = max_abs_importances[roi_idx]
            for finger in range(self.clf.weights.shape[1]):
                out_fingers[finger][voxel[0],voxel[1],voxel[2]] = importances[roi_idx,finger]
        nib.save(nib.Nifti1Image(out_img, ref_affine, header=ref_header), out_name)
        for finger in range(self.clf.weights.shape[1]):
            nib.save(nib.Nifti1Image(out_fingers[finger], ref_affine, header=ref_header), out_name+'_finger_'+str(finger))

    def save_finger_as_nifti(self, finger=0):
        import nibabel as nib
        out_name = self.ref_dir+'/'+self.subject_id+'_clf_out_'+str(finger)
        rfi_nii = nib.load(self.rfi_img)
        rfi_data = rfi_nii.get_data()
        ref_affine = rfi_nii.get_qform()
        ref_header = rfi_nii.header
        out_img = np.zeros(rfi_data.shape)
        for roi_idx, voxel in enumerate(self.fmri_data.fa.voxel_indices):
            out_img[voxel[0],voxel[1],voxel[2]] = self.clf.weights[roi_idx,finger]
        nib.save(nib.Nifti1Image(out_img, ref_affine, header=ref_header), out_name)

    def generate_distributions(self):
        self.voxel_means = np.zeros((self.fmri_data.nfeatures,self.n_class))
        self.voxel_covs = np.zeros((self.fmri_data.nfeatures,self.fmri_data.nfeatures,self.n_class))
        for target in range(self.n_class):
            self.voxel_means[:,target] = np.mean(np.array(self.fmri_data[self.fmri_data.targets==target,:]),0)
            self.voxel_covs[:,:,target] = np.cov(np.array(self.fmri_data[self.fmri_data.targets==target,:]).T)

    def generate_new_data(self, target, n_samples=1):
        return np.random.multivariate_normal(self.voxel_means[:,target],self.voxel_covs[:,:,target],n_samples)

    def create_rfi(self):
        in_bold = glob.glob(self.bold_dir+'/*run-'+str(1).zfill(3)+'*.nii')[0]
        out_bold = self.rfi_img
        os.system('fslmaths ' + in_bold + ' -Tmean ' + out_bold)
        gunzip_cmd = 'gunzip '+self.rfi_img+'.gz'
        os.system(gunzip_cmd)

    def motion_correct(self, mode='fsl'):
        # add mv ./* ./archive
        for run in range(self.num_runs):
            print 'starting mc run '+str(run+1)
            in_bold = glob.glob(self.bold_dir+'/*run-'+str(run+1).zfill(3)+'*.nii')[0]
            out_bold = self.bold_dir+'/rrun-'+str(run+1).zfill(3)+'.nii'
            ref_bold = self.rfi_img
            if mode == 'fsl':
                cmd = 'mcflirt -in '+in_bold+' -o '+out_bold+' -r '+ref_bold
            elif mode == 'afni':
                cmd = '3dvolreg -prefix '+out_bold+' -base '+ref_bold+' '+in_bold+' 2>/dev/null'
            os.system(cmd)
        if mode == 'fsl':
            gunzip_cmd = 'gunzip '+self.bold_dir+'/*.gz'
            os.system(gunzip_cmd)

    def register_2_rfi(self):
        cmd = ('bbregister --s '+self.fs_subject_id+' --mov '+self.rfi_img
            +' --init-fsl --bold --reg '+self.reg_mat)
        os.system(cmd)

    def generate_sensorimotor_masks(self, hemi='lh'):
        self.generate_exclusive_sensorimotor_masks(hemi)
        self.combine_sensorimotor_masks(hemi)

    def generate_exclusive_sensorimotor_masks(self, hemi='lh'):
        roi_names = ['ba4a','ba4p','ba3a','ba3b']
        self.generate_exclusive_mask(roi_names,hemi)
        cmd_m1_plus = ('fslmaths '+self.ref_dir+'/mask_'+hemi+'_multi_roi -uthr 2.5 -kernel sphere 2.3 -dilM -bin '
            +self.ref_dir+'/mask_'+hemi+'_m1+')
        os.system(cmd_m1_plus)
        cmd_m1_plusplus = ('fslmaths '+self.ref_dir+'/mask_'+hemi+'_multi_roi -uthr 2.5 -kernel sphere 4.6 -dilM -bin '
            +self.ref_dir+'/mask_'+hemi+'_m1++')
        os.system(cmd_m1_plusplus)
        cmd_m1 = 'fslmaths '+self.ref_dir+'/mask_'+hemi+'_multi_roi -uthr 2.5 -bin '+self.ref_dir+'/mask_'+hemi+'_m1'
        os.system(cmd_m1)
        cmd_s1_plus = ('fslmaths '+self.ref_dir+'/mask_'+hemi+'_multi_roi -thr 2.5 -kernel sphere 2.3 -dilM -bin '
            +self.ref_dir+'/mask_'+hemi+'_s1+')
        os.system(cmd_s1_plus)
        cmd_s1_plusplus = ('fslmaths '+self.ref_dir+'/mask_'+hemi+'_multi_roi -thr 2.5 -kernel sphere 4.6 -dilM -bin '
            +self.ref_dir+'/mask_'+hemi+'_s1++')
        os.system(cmd_s1_plusplus)
        cmd_s1 = 'fslmaths '+self.ref_dir+'/mask_'+hemi+'_multi_roi -thr 2.5 -bin '+self.ref_dir+'/mask_'+hemi+'_s1'
        os.system(cmd_s1)

        cmd_m1_minus = ('fslmaths '+self.ref_dir+'/mask_'+hemi+'_m1 -sub '
            +self.ref_dir+'/mask_'+hemi+'_s1+ -bin '+self.ref_dir+'/mask_'+hemi+'_m1-')
        os.system(cmd_m1_minus)
        cmd_m1_minusminus = ('fslmaths '+self.ref_dir+'/mask_'+hemi+'_m1 -sub '
            +self.ref_dir+'/mask_'+hemi+'_s1++ -bin '+self.ref_dir+'/mask_'+hemi+'_m1--')
        os.system(cmd_m1_minusminus)
        cmd_s1_minus = ('fslmaths '+self.ref_dir+'/mask_'+hemi+'_s1 -sub '
            +self.ref_dir+'/mask_'+hemi+'_m1+ -bin '+self.ref_dir+'/mask_'+hemi+'_s1-')
        os.system(cmd_s1_minus)
        cmd_s1_minusminus = ('fslmaths '+self.ref_dir+'/mask_'+hemi+'_s1 -sub '
            +self.ref_dir+'/mask_'+hemi+'_m1++ -bin '+self.ref_dir+'/mask_'+hemi+'_s1--')
        os.system(cmd_s1_minusminus)
        gunzip_cmd = 'gunzip '+self.ref_dir+'/*.gz'
        os.system(gunzip_cmd)

    def combine_sensorimotor_masks(self, hemi='lh'):
        motor_mask = self.ref_dir+'/mask_'+hemi+'_m1.nii'
        sensory_mask = self.ref_dir+'/mask_'+hemi+'_s1.nii'
        out_mask = self.ref_dir+'/mask_'+hemi+'_m1s1.nii'
        cmd = 'fslmaths '+motor_mask+' -add '+sensory_mask+' -bin '+out_mask
        os.system(cmd)
        gunzip_cmd = 'gunzip '+self.ref_dir+'/*.gz'
        os.system(gunzip_cmd)

    def generate_exclusive_mask(self, roi_names, hemi, fillthresh=0.3, proj_delta=0.1):
        SUBJECTS_DIR = os.environ['SUBJECTS_DIR']
        cmd = 'mri_label2vol --subject '+self.fs_subject_id
        for roi_name in roi_names:
            cmd = (cmd+' --label '+SUBJECTS_DIR+'/'+self.fs_subject_id+'/label/'
            +hemi+'.'+roi_name+'_exvivo.label ')
        cmd = (cmd+'--temp '+self.rfi_img
            +' --reg '+self.reg_mat+' --proj frac 0 1 '
            +str(proj_delta)+' --fillthresh '+str(fillthresh)
            +' --hemi '+hemi+' --o '+self.ref_dir+'/mask_'+hemi+'_multi_roi.nii')
        subprocess.Popen(shlex.split(cmd)).wait()
        gunzip_cmd = 'gunzip '+self.ref_dir+'/*.gz'
        os.system(gunzip_cmd)

    def generate_mask(self, roi_name, hemi, fillthresh=1, proj_delta=0.4, run_cmd=True):
        SUBJECTS_DIR = os.environ['SUBJECTS_DIR']
        cmd = ('mri_label2vol --subject '+self.fs_subject_id
            +' --label '+SUBJECTS_DIR+'/'+self.fs_subject_id+'/label/'
            +hemi+'.'+roi_name+'_exvivo.label --temp '+self.rfi_img
            +' --reg '+self.reg_mat+' --proj frac 0 1 '
            +str(proj_delta)+' --fillthresh '+str(fillthresh)
            +' --hemi '+hemi+' --o '+self.ref_dir+'/mask_'+hemi+'_'+roi_name+'.nii')
        if run_cmd:
            subprocess.Popen(shlex.split(cmd)).wait()
        else:
            print cmd

    def combine_hemis(self, roi_name):
        rh_mask = self.ref_dir+'/mask_rh_'+roi_name+'.nii'
        lh_mask = self.ref_dir+'/mask_lh_'+roi_name+'.nii'
        bi_mask = self.ref_dir+'/mask_bi_'+roi_name+'.nii'
        cmd = 'fslmaths '+rh_mask+' -add '+lh_mask+' -bin '+bi_mask
        os.system(cmd)

    def convert_to_pycortex(self):
        import cortex; from cortex.xfm import Transform
        cortex.freesurfer.import_subj(self.fs_subject_id, sname=self.subject_id)
        cmd = ('tkregister2 --mov '+self.rfi_img+' --targ '+self.rai_img
            +' --reg '+self.reg_mat+' --fslregout '
            +self.fsl_reg_mat+' --noedit')
        subprocess.Popen(shlex.split(cmd)).wait()
        x = np.loadtxt(self.fsl_reg_mat)
        Transform.from_fsl(x, self.rfi_img, self.rai_img).save(self.subject_id, 'rai2rfi', 'coord')
