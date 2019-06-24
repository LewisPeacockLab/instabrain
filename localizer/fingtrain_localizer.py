import yaml,os,glob,pickle,copy,subprocess,shlex
import numpy as np
import pandas as pd
from mvpa2.clfs.smlr import SMLR
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.base.dataset import vstack
from mvpa2.mappers.zscore import zscore
from mvpa2.mappers.detrend import poly_detrend
from mvpa2.mappers.fx import mean_group_sample
from mvpa2.measures.base import CrossValidation
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.clfs.smlr import SMLR
from scipy.signal import detrend, savgol_filter

class FingtrainLocalizer(object):
    def __init__(self, subject_id):
        with open('fingtrain_localizer_config.yml') as f:
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
        self.bold_dir = self.base_dir+'/bold/localizer'

        # timing
        self.vols_per_run = self.CONFIG['VOLS_PER_RUN']
        self.num_runs = self.CONFIG['NUM_RUNS']
        self.trs_per_trial = self.CONFIG['TRS_PER_TRIAL']
        self.trial_feature_trs = self.CONFIG['TRIAL_FEATURE_TRS']
        self.zscore_trs = self.CONFIG['ZSCORE_TRS']
        self.trials_per_run = (self.vols_per_run-self.zscore_trs)/self.trs_per_trial

        # labels
        self.behav_data = pd.read_csv(self.ref_dir+'/localizer_press.csv')
        self.trial_data = self.behav_data.groupby('trial',sort=False).first()
        self.trial_targets = self.trial_data.target_finger
        self.trial_chunks = self.trial_data.run
        self.n_class = np.unique(self.trial_targets).size

        # classifier
        self.clf = SMLR()

    def preprocessing(self):
        self.create_rfi()
        self.register_2_rfi()
        self.generate_sensorimotor_masks()
        self.motion_correct()

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
            elif detrend == 'sg1':
                run_dataset = self.realtime_sg_filter(run_dataset,order=1)
            elif detrend == 'sg2':
                run_dataset = self.realtime_sg_filter(run_dataset,order=2)
            elif detrend == 'sg3':
                run_dataset = self.realtime_sg_filter(run_dataset,order=3)
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

    def realtime_sg_filter(self, run_dataset, max_frame_length=121, order=1):
        run_data = copy.deepcopy(run_dataset.samples)
        out_data = np.zeros(run_data.shape)
        out_data[:self.zscore_trs] = detrend(run_data[:self.zscore_trs],0)
        for tr in range(self.zscore_trs,self.vols_per_run):
            if tr<max_frame_length:
                frame_length = tr-int(not(tr%2))
            else:
                frame_length = max_frame_length
            out_data[tr] = (run_data[tr]-savgol_filter(run_data[:tr+1], frame_length, order, axis=0)[-1])
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

    def train_classifier(self):
        self.clf.train(self.fmri_data)

    def apply_classifier(self, data):
        self.clf.predict(data)
        return self.clf.ca.estimates

    def get_cv_clf_outs(self):
        self.cv_clf_outs = np.zeros((self.num_runs*self.trials_per_run,self.n_class))
        for run in range(self.num_runs):
            self.clf.train(self.fmri_data[self.fmri_data.chunks!=run])
            self.cv_clf_outs[run*self.trials_per_run:(run+1)*self.trials_per_run,:] = (
            self.apply_classifier(self.fmri_data[self.fmri_data.chunks==run]))

    def calculate_neurofeedback_score(self, score_scale_exponent=0.5):
        self.nfb_scores_1 = []
        self.nfb_scores_2 = []
        for idx,target_finger in enumerate(self.trial_targets):
            if target_finger == 1:
                positive_finger = 0
                negative_finger = 2
                append_to = self.nfb_scores_1
            elif target_finger == 2:
                positive_finger = 3
                negative_finger = 1
                append_to = self.nfb_scores_2
            if target_finger in [1,2]:
                clf_data = self.cv_clf_outs[idx,:]
                raw_score = clf_data[positive_finger]-clf_data[negative_finger]
                append_to.append(0.5+0.5*np.copysign(np.power(np.abs(raw_score),
                    score_scale_exponent),raw_score))

    def plot_neurofeedback_score(self, score_scale_exponent=0.5):
        import matplotlib.pyplot as plt
        import seaborn as sea
        self.calculate_neurofeedback_score(score_scale_exponent)
        plt.ion()
        sea.distplot(self.nfb_scores_1)
        sea.distplot(self.nfb_scores_2)
        plt.xlim((0,1))
        plt.title(self.subject_id+' calculated bias scores')
        plt.legend(['middle','ring'])

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

    def create_rfi(self):
        in_bold = glob.glob(self.bold_dir+'/*run-'+str(1).zfill(3)+'*.nii')[0]
        out_bold = self.rfi_img
        os.system('fslmaths ' + in_bold + ' -Tmean ' + out_bold)
        gunzip_cmd = 'gunzip '+self.rfi_img+'.gz'
        os.system(gunzip_cmd)

    def motion_correct(self, mode='afni'):
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
        cmd_m1 = 'fslmaths '+self.ref_dir+'/mask_'+hemi+'_multi_roi -uthr 2.5 -bin '+self.ref_dir+'/mask_'+hemi+'_m1'
        os.system(cmd_m1)
        cmd_s1 = 'fslmaths '+self.ref_dir+'/mask_'+hemi+'_multi_roi -thr 2.5 -bin '+self.ref_dir+'/mask_'+hemi+'_s1'
        os.system(cmd_s1)
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
