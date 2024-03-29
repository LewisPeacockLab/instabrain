import numpy as np
import yaml,os,glob,pickle
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

class InstaLocalizer(object):
    def __init__(self, subject_id):
        with open('localizer_config.yml') as f:
            self.CONFIG = yaml.load(f)
        
        # directories
        self.subject_id = subject_id
        self.fs_subject_id = self.subject_id+'fs'
        self.base_dir = self.CONFIG['SUBJECT_DIR']+'/'+self.subject_id
        self.ref_dir = self.base_dir+'/ref'
        self.rfi_img = self.ref_dir+'/rfi.nii'
        self.reg_mat = self.ref_dir+'/rai2rfi.dat'
        self.bold_dir = self.base_dir+'/bold'

        # timing
        self.tr = self.CONFIG['TR']
        self.vols_per_run = self.CONFIG['VOLS_PER_RUN']
        self.num_runs = self.CONFIG['NUM_RUNS']
        self.trs_per_trial = self.CONFIG['TRS_PER_TRIAL']
        self.trial_feature_trs = self.CONFIG['TRIAL_FEATURE_TRS']
        self.trials_per_run = self.vols_per_run/self.trs_per_trial
        self.presses_per_trial = self.CONFIG['PRESSES_PER_TRIAL']

        # labels
        self.behav_data = np.loadtxt(self.ref_dir+'/ft-data.txt',delimiter=',',skiprows=1)
        self.trial_data = self.behav_data[::self.presses_per_trial,:]
        self.trial_targets = self.trial_data[:,0]
        self.trial_chunks = self.trial_data[:,-1]
        self.n_class = np.unique(self.trial_targets).size

        # classifier
        self.clf = SMLR()

    def preprocessing(self):
        self.create_rfi()
        self.register_2_rfi()
        self.generate_motor_masks()
        self.motion_correct()

    def extract_features(self, roi_name=None, hemi='rh'):
        datasets = []
        self.tr_targets = np.tile(self.trial_targets,(self.trs_per_trial,1)).flatten('F')
        self.tr_chunks = np.tile(self.trial_chunks,(self.trs_per_trial,1)).flatten('F')
        if roi_name == None:
            mask = None
        else:
            mask = self.ref_dir+'/mask_'+hemi+'_'+roi_name+'.nii'
        for run in range(self.num_runs):
            datasets.append(fmri_dataset(self.bold_dir+'/rrun-'+str(run+1).zfill(3)+'.nii',
                mask=mask,
                targets=self.tr_targets[self.tr_chunks==run],
                chunks=self.tr_chunks[self.tr_chunks==run]))
        self.fmri_data = vstack(datasets, a=0)

        self.active_trs = np.zeros(self.trs_per_trial)
        self.active_trs[self.trial_feature_trs[0]-1:self.trial_feature_trs[1]] = 1
        self.active_trs = np.tile(self.active_trs,int(self.num_runs*self.trials_per_run))
        self.trial_regressor = np.tile(range(int(self.num_runs*self.trials_per_run)),(self.trs_per_trial,1)).flatten('F')
        self.fmri_data.sa['active_trs'] = self.active_trs
        self.fmri_data.sa['trial_regressor'] = self.trial_regressor
        poly_detrend(self.fmri_data, polyord=1, chunks_attr='chunks')
        zscore(self.fmri_data, chunks_attr='chunks')
        self.fmri_data = self.fmri_data[self.fmri_data.sa.active_trs==1]
        trial_mapper = mean_group_sample(['targets','trial_regressor'])
        self.fmri_data = self.fmri_data.get_mapped(trial_mapper)

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

    def generate_motor_masks(self):
        for roi_name in ['BA4p','BA4a','BA6']:
            for hemi in ['lh','rh']:
                self.generate_mask(roi_name,hemi)
            self.combine_hemis(roi_name)
        gunzip_cmd = 'gunzip '+self.ref_dir+'/*.gz'
        os.system(gunzip_cmd)

    def generate_mask(self, roi_name, hemi):
        cmd = ('mri_label2vol --subject '+self.fs_subject_id
            +' --label $SUBJECTS_DIR/'+self.fs_subject_id+'/label/'
            +hemi+'.'+roi_name+'_exvivo.label --temp '+self.rfi_img
            +' --reg '+self.reg_mat+' --proj frac 0 1 .1 --fillthresh .3'
            +' --hemi '+hemi+' --o '+self.ref_dir+'/mask_'+hemi+'_'+roi_name+'.nii')
        os.system(cmd)

    def combine_hemis(self, roi_name):
        rh_mask = self.ref_dir+'/mask_rh_'+roi_name+'.nii'
        lh_mask = self.ref_dir+'/mask_lh_'+roi_name+'.nii'
        bi_mask = self.ref_dir+'/mask_bi_'+roi_name+'.nii'
        cmd = 'fslmaths '+rh_mask+' -add '+lh_mask+' -bin '+bi_mask
        os.system(cmd)
