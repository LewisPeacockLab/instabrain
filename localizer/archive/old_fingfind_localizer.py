import numpy as np
import yaml,os,glob,pickle
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
        self.bold_dir = self.bold_dir_sess1

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
        self.behav_data_sess2 = np.loadtxt(self.ref_dir+'/ft-data-sess2.txt',delimiter=',',skiprows=1)
        self.trial_data_sess1 = self.behav_data_sess1[::self.presses_per_trial,:]
        self.trial_data_sess2 = self.behav_data_sess2[::self.presses_per_trial,:]
        self.trial_targets_sess1 = self.trial_data_sess1[:,0]
        self.trial_targets_sess2 = self.trial_data_sess2[:,0]
        self.trial_targets = self.trial_targets_sess1
        self.trial_chunks = self.trial_data_sess1[:,-1]
        self.n_class = np.unique(self.trial_targets).size

        # classifier
        self.clf = SMLR()

    def preprocessing(self):
        self.create_rfi()
        self.register_2_rfi()
        self.generate_motor_masks()
        self.generate_sensorimotor_masks()
        self.motion_correct()

    def extract_features(self, roi_name=None, hemi='lh', zs_proportion=1, zs_all=True, detrend=True):
        datasets = []
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
            datasets.append(fmri_dataset(self.bold_dir+'/rrun-'+str(run+1).zfill(3)+'.nii',
                mask=mask,
                targets=run_tr_targets,
                chunks=run_tr_chunks))
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
        if detrend:
            poly_detrend(self.fmri_data, polyord=1, chunks_attr='chunks')
        if zs_all:
            zscore(self.fmri_data, chunks_attr='chunks')
        else:
            zscore(self.fmri_data, chunks_attr='chunks', param_est=('targets',[-1]))
        self.fmri_data = self.fmri_data[self.fmri_data.sa.active_trs==1]
        trial_mapper = mean_group_sample(['targets','trial_regressor'])
        self.fmri_data = self.fmri_data.get_mapped(trial_mapper)

    def extract_multi_session_features(self, roi_name=None, hemi='lh', zs_proportion=1, zs_all=True, detrend=True):
        datasets = []
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
            datasets.append(fmri_dataset(self.bold_dir+'/rrun-'+str(run+1).zfill(3)+'.nii',
                mask=mask,
                targets=run_tr_targets,
                chunks=run_tr_chunks))

        self.bold_dir = self.bold_dir_sess2
        for run in range(self.num_runs,2*self.num_runs):
            run_tr_targets = np.append(-2*np.ones(int((1-zs_proportion)*self.zscore_trs)), self.tr_targets[self.tr_chunks==run])
            run_tr_targets = np.append(-1*np.ones(int(zs_proportion*self.zscore_trs)), run_tr_targets)
            run_tr_chunks = np.append(run*np.ones(self.zscore_trs), self.tr_chunks[self.tr_chunks==run])
            datasets.append(fmri_dataset(self.bold_dir+'/rrun-'+str(run-7).zfill(3)+'.nii',
                mask=mask,
                targets=run_tr_targets,
                chunks=run_tr_chunks))

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
        if detrend:
            poly_detrend(self.fmri_data, polyord=1, chunks_attr='chunks')
        if zs_all:
            zscore(self.fmri_data, chunks_attr='chunks')
        else:
            zscore(self.fmri_data, chunks_attr='chunks', param_est=('targets',[-1]))
        self.fmri_data = self.fmri_data[self.fmri_data.sa.active_trs==1]
        trial_mapper = mean_group_sample(['targets','trial_regressor'])
        self.fmri_data = self.fmri_data.get_mapped(trial_mapper)

    def test_multi_clf_params(self,lms=[0.01,0.1,1,10,20]):
        # lms = np.round(np.logspace(-2,0,10),2) 
        # lms = np.linspace(.1,.5,9)
        # roi_name = 'ba3ba4'
        roi_name = 'ba4_normal'
        zs1 = True; zs2 = False; detrend = True 
        self.df = self.create_empty_clf_param_df(conditions=len(lms))
        for idx,lm in enumerate(lms):
            self.clf = SMLR(lm=lm)
            self.test_cross_session_decoding(roi_name,zs1,zs2)
            self.df.loc[idx,['zs_all_sess1','zs_all_sess2','detrend','roi_name','lm']]=[zs1,
                zs2,detrend,roi_name,lm]
            self.df.loc[idx,'clf_acc'] = self.cross_sess_accuracy
        self.df.to_csv(self.subject_id+'_clf_params.csv')

    def test_cross_session_decoding(self, roi_name='ba4', zs_all_sess1=False, zs_all_sess2=False,
        hemi='lh', zs_proportion=1, detrend=True):
        self.bold_dir = self.bold_dir_sess1
        self.trial_targets = self.trial_targets_sess1
        self.extract_features(roi_name=roi_name, hemi=hemi, zs_proportion=zs_proportion, zs_all=zs_all_sess1, detrend=detrend)
        self.train_classifier()
        self.bold_dir = self.bold_dir_sess2
        self.trial_targets = self.trial_targets_sess2
        self.extract_features(roi_name=roi_name, hemi=hemi, zs_proportion=zs_proportion, zs_all=zs_all_sess2, detrend=detrend)
        predictions = self.clf.predict(self.fmri_data)
        self.cross_sess_accuracy = np.mean(predictions==self.fmri_data.sa.targets)

    def test_cross_session_decoding_params(self, detrend=True):
        # roi_names = ['ba6','ba4','ba3']
        roi_names = ['ba4','ba3']
        for roi_name in roi_names:
            self.df = self.create_empty_cross_session_df(conditions=3)
            for idx, zs in enumerate([[True,True],[True,False],[False,False]]):
                zs1 = zs[0]; zs2 = zs[1]
                self.test_cross_session_decoding(roi_name,zs1,zs2)
                self.df.loc[idx,['zs_all_sess1','zs_all_sess2','detrend','roi_name']]=[zs1,
                    zs2,detrend,roi_name]
                self.df.loc[idx,'clf_acc'] = self.cross_sess_accuracy
            self.df.to_csv(self.subject_id+'_'+roi_name+'_cross_sess.csv')

    def test_cross_session_roi_params(self, detrend=True):
        roi_names = ['ba4','ba4_normal','ba3ba4','ba3_normal','ba3']
        for roi_name in roi_names:
            self.df = self.create_empty_cross_session_df(conditions=1)
            for idx, zs in enumerate([[True,False]]):
                zs1 = zs[0]; zs2 = zs[1]
                self.test_cross_session_decoding(roi_name,zs1,zs2)
                self.df.loc[idx,['zs_all_sess1','zs_all_sess2','detrend','roi_name']]=[zs1,
                    zs2,detrend,roi_name]
                self.df.loc[idx,'clf_acc'] = self.cross_sess_accuracy
            self.df.to_csv(self.subject_id+'_'+roi_name+'_cross_sess.csv')

    def test_cross_session_decoder_out(self, detrend=True):
        # roi_names = ['ba6','ba4','ba3']
        roi_names = ['ba4','ba3']
        for roi_name in roi_names:
            self.df = self.create_empty_cross_session_out_df(conditions=3,
                trials=self.num_runs*self.trials_per_run)
            for idx, zs in enumerate([[True,True],[True,False],[False,False]]):
                zs1 = zs[0]; zs2 = zs[1]
                self.test_cross_session_decoding(roi_name,zs1,zs2)
                self.df.loc[idx,['zs_all_sess1','zs_all_sess2','detrend','roi_name']]=[zs1,
                    zs2,detrend,roi_name]
                self.df.loc[idx,'target'] = self.fmri_data.sa.targets
                self.df.loc[idx,'clf_out_0'] = self.clf.ca.estimates[:,0]
                self.df.loc[idx,'clf_out_1'] = self.clf.ca.estimates[:,1]
                self.df.loc[idx,'clf_out_2'] = self.clf.ca.estimates[:,2]
                self.df.loc[idx,'clf_out_3'] = self.clf.ca.estimates[:,3]
            self.df.to_csv(self.subject_id+'_'+roi_name+'_cross_sess_out.csv')

    def test_all_filtering_multi_roi(self):
        roi_names = ['ba6','ba4','ba3']
        hemi = 'lh'
        for roi_name in roi_names:
            self.test_all_filtering(roi_name,hemi)

    def test_time_points_multi_roi(self):
        roi_names = ['ba3','ba3_normal','ba3ba4','ba4_normal','ba4']
        hemi = 'lh'
        self.bold_dir = self.bold_dir_sess2
        self.trial_targets = self.trial_targets_sess2
        for roi_name in roi_names:
            # self.test_filtering_time_points(roi_name,hemi,trial_feature_trs_array=[[4,6]],multi_session=True)
            self.test_filtering_time_points(roi_name,hemi,trial_feature_trs_array=[[4,6]])

    def test_all_filtering(self, roi_name='ba4', hemi='lh'):
        print 'starting time points...'
        self.test_filtering_time_points(roi_name,hemi)
        print 'starting zs proportion...'
        self.test_filtering_zs_proportion(roi_name,hemi)
        print 'starting zs all...'
        self.test_filtering_zs_all(roi_name,hemi)
        print 'starting detrend...'
        self.test_filtering_detrend(roi_name,hemi)
        print 'done!'

    def test_filtering_time_points(self, roi_name='ba4a', hemi='lh',
            trial_feature_trs_array=[[1,3],[2,4],[3,5],[4,6],[5,7],[6,8]],
            zs_proportion=1, zs_all=False, detrend=True, multi_session=False):
        if multi_session:
            runs = 2*self.num_runs
        else:
            runs = self.num_runs
        self.df = self.create_empty_filtering_df(cv_folds=runs,
            conditions_per_cv=len(trial_feature_trs_array))
        for idx,feature_trs in enumerate(trial_feature_trs_array):
            print 'starting feature TRs: '+str(feature_trs)
            self.trial_feature_trs = trial_feature_trs_array[idx]
            if multi_session:
                self.extract_multi_session_features(roi_name=roi_name, hemi=hemi, zs_proportion=zs_proportion, zs_all=zs_all, detrend=detrend)
            else:
                self.extract_features(roi_name=roi_name, hemi=hemi, zs_proportion=zs_proportion, zs_all=zs_all, detrend=detrend)
            self.cross_validate()
            self.df.loc[idx,['feature_trs','zs_prop','zs_all','detrend','roi_name']]=[feature_trs,
                zs_proportion,zs_all,detrend,roi_name]
            self.df.loc[idx,'clf_acc']=np.array(self.cv_results).T
        self.df.to_csv(self.subject_id+'_'+roi_name+'_time.csv')

    def test_filtering_zs_proportion(self, roi_name='ba4a', hemi='lh',
            zs_proportion_array=[.25, .5, .75, 1], trial_feature_trs=[4,6],
            zs_all=False, detrend=True):
        self.trial_feature_trs = trial_feature_trs
        self.df = self.create_empty_filtering_df(cv_folds=self.num_runs,
            conditions_per_cv=len(zs_proportion_array))
        for idx,zs_proportion in enumerate(zs_proportion_array):
            print 'starting zscore proportion: '+str(zs_proportion)
            self.extract_features(roi_name=roi_name, hemi=hemi, zs_proportion=zs_proportion, zs_all=zs_all, detrend=detrend)
            self.cross_validate()
            self.df.loc[idx,['feature_trs','zs_prop','zs_all','detrend','roi_name']]=[trial_feature_trs,
                zs_proportion,zs_all,detrend,roi_name]
            self.df.loc[idx,'clf_acc']=np.array(self.cv_results).T
        self.df.to_csv(self.subject_id+'_'+roi_name+'_zs_proportion.csv')

    def test_filtering_zs_all(self, roi_name='ba4a', hemi='lh',
            trial_feature_trs=[4,6],detrend=True,zs_proportion=1,
            zs_all_array=[True,False]):
        self.trial_feature_trs = trial_feature_trs
        self.df = self.create_empty_filtering_df(cv_folds=self.num_runs,
            conditions_per_cv=len(zs_all_array))
        for idx,zs_all in enumerate(zs_all_array):
            print 'starting zscore proportion: '+str(zs_proportion)
            self.extract_features(roi_name=roi_name, hemi=hemi, zs_proportion=zs_proportion, zs_all=zs_all, detrend=detrend)
            self.cross_validate()
            self.df.loc[idx,['feature_trs','zs_prop','zs_all','detrend','roi_name']]=[trial_feature_trs,
                zs_proportion,zs_all,detrend,roi_name]
            self.df.loc[idx,'clf_acc']=np.array(self.cv_results).T
        self.df.to_csv(self.subject_id+'_'+roi_name+'_zs_all.csv')

    def test_filtering_detrend(self, roi_name='ba4a', hemi='lh',
            trial_feature_trs=[4,6],zs_all=False,zs_proportion=1,
            detrend_array=[True,False]):
        self.trial_feature_trs = trial_feature_trs
        self.df = self.create_empty_filtering_df(cv_folds=self.num_runs,
            conditions_per_cv=len(detrend_array))
        for idx,detrend in enumerate(detrend_array):
            print 'starting zscore proportion: '+str(zs_proportion)
            self.extract_features(roi_name=roi_name, hemi=hemi, zs_proportion=zs_proportion, zs_all=zs_all, detrend=detrend)
            self.cross_validate()
            self.df.loc[idx,['feature_trs','zs_prop','zs_all','detrend','roi_name']]=[trial_feature_trs,
                zs_proportion,zs_all,detrend,roi_name]
            self.df.loc[idx,'clf_acc']=np.array(self.cv_results).T
        self.df.to_csv(self.subject_id+'_'+roi_name+'_detrend.csv')

    def create_empty_clf_param_df(self, conditions):
        import pandas as pd
        return pd.DataFrame(
            columns=['clf_acc','zs_all_sess1','zs_all_sess2','detrend','roi_name','lm'],
            index=range(conditions))

    def create_empty_filtering_df(self, cv_folds=8, conditions_per_cv=6):
        import pandas as pd
        return pd.DataFrame(
            columns=['clf_acc','feature_trs','zs_prop','zs_all','detrend','roi_name'],
            index=np.repeat(range(conditions_per_cv),cv_folds))

    def create_empty_cross_session_df(self, conditions):
        import pandas as pd
        return pd.DataFrame(
            columns=['clf_acc','zs_all_sess1','zs_all_sess2','detrend','roi_name'],
            index=range(conditions))

    def create_empty_cross_session_out_df(self, conditions, trials):
        import pandas as pd
        return pd.DataFrame(
            columns=['target','clf_out_0','clf_out_1','clf_out_2','clf_out_3',
            'zs_all_sess1','zs_all_sess2','detrend','roi_name'],
            index=np.repeat(range(conditions),trials))

    def extract_betas(self, roi_name=None, hemi='rh',
            onset_offset_trs=0, duration_offset_trs=0,
            betas_over='chunks'):
        condition_attr = ('targets',betas_over)
        datasets = []
        self.tr_targets = np.tile(self.trial_targets,(self.trs_per_trial,1)).flatten('F')
        self.tr_chunks = np.tile(self.trial_chunks,(self.trs_per_trial,1)).flatten('F')
        self.trial_regressor = np.tile(range(int(self.trials_per_run)),(self.trs_per_trial,1)).flatten('F')
        if roi_name == None:
            mask = None
        else:
            mask = self.ref_dir+'/mask_'+hemi+'_'+roi_name+'.nii'
        for run in range(self.num_runs):
            run_dataset = fmri_dataset(self.bold_dir+'/rrun-'+str(run+1).zfill(3)+'.nii',
                            mask=mask,
                            targets=self.tr_targets[self.tr_chunks==run],
                            chunks=self.tr_chunks[self.tr_chunks==run])
            run_dataset.sa['trials'] = self.trial_regressor
            poly_detrend(run_dataset, polyord=1, chunks_attr='chunks')
            zscore(run_dataset, chunks_attr='chunks')
            run_events = find_events(targets=run_dataset.sa.targets,
                chunks=run_dataset.sa.chunks,
                trials=run_dataset.sa['trials'])
            for event in run_events:
                event['onset'] = onset_offset_trs+event['onset']*self.tr
                event['duration'] = (event['duration']+duration_offset_trs)*self.tr
            run_event_dataset = fit_event_hrf_model(run_dataset,
                run_events,
                time_attr='time_coords',
                condition_attr=condition_attr)
            datasets.append(run_event_dataset)
        self.fmri_data = vstack(datasets, a=0)
        if betas_over == 'trials':
            self.fmri_data.sa['chunks'] = np.tile(range(self.num_runs),(self.trials_per_run,1)).flatten('F')

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
        out_name = 'clf_out'
        rfi_nii = nib.load(self.rfi_img)
        rfi_data = rfi_nii.get_data()
        ref_affine = rfi_nii.get_qform()
        ref_header = rfi_nii.header
        out_img = np.zeros(rfi_data.shape)
        max_abs_clf_weights = [voxel[np.where(np.abs(voxel)==np.max(np.abs(voxel)))][0] for voxel in self.clf.weights]
        for roi_idx, voxel in enumerate(self.fmri_data.fa.voxel_indices):
            out_img[voxel[0],voxel[1],voxel[2]] = max_abs_clf_weights[roi_idx]
        nib.save(nib.Nifti1Image(out_img, ref_affine, header=ref_header), out_name)

    def save_finger_as_nifti(self, finger=0):
        import nibabel as nib
        out_name = 'clf_out_'+str(finger)
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

    def motion_correct(self):
        # add mv ./* ./archive
        for run in range(self.num_runs):
            print 'starting mc run '+str(run+1)
            in_bold = glob.glob(self.bold_dir+'/*run-'+str(run+1).zfill(3)+'*.nii')[0]
            out_bold = self.bold_dir+'/rrun-'+str(run+1).zfill(3)+'.nii'
            ref_bold = self.rfi_img
            cmd = 'mcflirt -in '+in_bold+' -o '+out_bold+' -r '+ref_bold
            os.system(cmd)
        gunzip_cmd = 'gunzip '+self.bold_dir+'/*.gz'
        os.system(gunzip_cmd)

    def register_2_rfi(self):
        cmd = ('bbregister --s '+self.fs_subject_id+' --mov '+self.rfi_img
            +' --init-fsl --bold --reg '+self.reg_mat)
        os.system(cmd)

    def generate_exclusive_sensorimotor_masks(self, hemi='lh'):
        roi_names = ['ba4a','ba4p','ba3a','ba3b']
        self.generate_exclusive_mask(roi_names,hemi)
        cmd_ba4_expanded = ('fslmaths '+self.ref_dir+'/mask_'+hemi+'_multi_roi -uthr 2.5 -kernel sphere 2.3 -dilM -bin '
            +self.ref_dir+'/mask_'+hemi+'_ba4_expanded')
        os.system(cmd_ba4_expanded)
        cmd_ba4_normal = 'fslmaths '+self.ref_dir+'/mask_'+hemi+'_multi_roi -uthr 2.5 -bin '+self.ref_dir+'/mask_'+hemi+'_ba4_normal'
        os.system(cmd_ba4_normal)
        cmd_ba3_expanded = ('fslmaths '+self.ref_dir+'/mask_'+hemi+'_multi_roi -thr 2.5 -kernel sphere 2.3 -dilM -bin '
            +self.ref_dir+'/mask_'+hemi+'_ba3_expanded')
        os.system(cmd_ba3_expanded)
        cmd_ba3_normal = 'fslmaths '+self.ref_dir+'/mask_'+hemi+'_multi_roi -thr 2.5 -bin '+self.ref_dir+'/mask_'+hemi+'_ba3_normal'
        os.system(cmd_ba3_normal)

        cmd_reduce_ba4 = ('fslmaths '+self.ref_dir+'/mask_'+hemi+'_ba4_normal -sub '
            +self.ref_dir+'/mask_'+hemi+'_ba3_expanded -bin '+self.ref_dir+'/mask_'+hemi+'_ba4')
        os.system(cmd_reduce_ba4)
        cmd_reduce_ba3 = ('fslmaths '+self.ref_dir+'/mask_'+hemi+'_ba3_normal -sub '
            +self.ref_dir+'/mask_'+hemi+'_ba4_expanded -bin '+self.ref_dir+'/mask_'+hemi+'_ba3')
        os.system(cmd_reduce_ba3)
        gunzip_cmd = 'gunzip '+self.ref_dir+'/*.gz'
        os.system(gunzip_cmd)

    def combine_sensorimotor_masks(self, hemi='lh'):
        motor_mask = self.ref_dir+'/mask_'+hemi+'_ba3_normal.nii'
        sensory_mask = self.ref_dir+'/mask_'+hemi+'_ba4_normal.nii'
        out_mask = self.ref_dir+'/mask_'+hemi+'_ba3ba4.nii'
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

    def generate_sensorimotor_masks(self):
        self.generate_motor_masks()
        self.generate_sensory_masks()

    def generate_motor_masks(self):
        for roi_name in ['ba4p','ba4a','ba6']:
            for hemi in ['lh','rh']:
                self.generate_mask(roi_name,hemi)
            self.combine_hemis(roi_name)
        for hemi in ['lh','rh']:
            a_mask = self.ref_dir+'/mask_'+hemi+'_ba4a.nii'
            p_mask = self.ref_dir+'/mask_'+hemi+'_ba4p.nii'
            out_mask = self.ref_dir+'/mask_'+hemi+'_ba4.nii'
            cmd = 'fslmaths '+a_mask+' -add '+p_mask+' -bin '+out_mask
            os.system(cmd)
        self.combine_hemis('ba4')
        gunzip_cmd = 'gunzip '+self.ref_dir+'/*.gz'
        os.system(gunzip_cmd)

    def generate_sensory_masks(self):
        for hemi in ['lh','rh']:
            for roi_name in ['ba3a', 'ba3b']:
                self.generate_mask(roi_name,hemi)
            a_mask = self.ref_dir+'/mask_'+hemi+'_ba3a.nii'
            b_mask = self.ref_dir+'/mask_'+hemi+'_ba3b.nii'
            out_mask = self.ref_dir+'/mask_'+hemi+'_ba3.nii'
            cmd = 'fslmaths '+a_mask+' -add '+b_mask+' -bin '+out_mask
            os.system(cmd)
        self.combine_hemis('ba3')
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
