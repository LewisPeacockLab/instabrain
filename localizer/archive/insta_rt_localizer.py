import numpy as np
import yaml,os,glob,pickle
from mvpa2.clfs.smlr import SMLR
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.base.dataset import vstack
from mvpa2.mappers.zscore import zscore
from mvpa2.mappers.detrend import poly_detrend
from mvpa2.mappers.fx import mean_group_sample
from mvpa2.clfs.smlr import SMLR

class InstaRtLocalizer(object):
    def __init__(self, subject_id):
        with open('rt_localizer_config.yml') as f:
            self.CONFIG = yaml.load(f)
        
        # directories
        self.subject_id = subject_id
        self.fs_subject_id = self.subject_id+'fs'
        self.base_dir = self.CONFIG['SUBJECT_DIR']+'/'+self.subject_id
        self.ref_dir = self.base_dir+'/ref'
        self.rfi_img = self.ref_dir+'/rfi.nii'
        self.reg_mat = self.ref_dir+'/rai2rfi.dat'
        self.bold_dir = self.base_dir+'/bold-ff'
        self.proc_dir = self.base_dir+'/proc'

        # timing
        self.tr = self.CONFIG['TR']
        self.vols_per_run = self.CONFIG['VOLS_PER_RUN']
        self.num_runs = self.CONFIG['NUM_RUNS']
        self.trs_per_trial = self.CONFIG['TRS_PER_TRIAL']
        self.trial_feature_trs = self.CONFIG['TRIAL_FEATURE_TRS']
        self.trials_per_run = self.vols_per_run/self.trs_per_trial
        self.presses_per_trial = self.CONFIG['PRESSES_PER_TRIAL']
        self.zscore_trs = self.CONFIG['ZSCORE_TRS']

        # labels
        self.behav_data = np.loadtxt(self.ref_dir+'/ff-data.txt',delimiter=',',skiprows=1)
        self.trial_data = self.behav_data[::self.presses_per_trial,:]
        self.trial_targets = self.trial_data[:,7]
        self.trial_chunks = self.trial_data[:,0]
        self.n_class = np.unique(self.trial_targets).size

        # classifier
        self.clf = SMLR()

    def preprocessing(self):
        self.create_rfi()
        self.register_2_rfi()
        self.generate_motor_masks()
        self.motion_correct()

    def concat_rt_data(self):
        for run in range(1,self.num_runs+1):
            run_data = self.proc_dir+'/run_'+str(run).zfill(2)+'/*mc*'
            cmd = 'fslmerge -t ' + self.bold_dir + '/rrun-' + str(run).zfill(3) + ' ' + run_data
            os.system(cmd)
        gunzip_cmd = 'gunzip '+self.bold_dir+'/*.gz'
        os.system(gunzip_cmd)

    def extract_features(self, roi_name=None, hemi='rh', zs_all=True, detrend=True):
        datasets = []
        self.tr_targets = np.tile(self.trial_targets,(self.trs_per_trial,1)).flatten('F')
        self.tr_chunks = np.tile(self.trial_chunks,(self.trs_per_trial,1)).flatten('F')
        if roi_name == None:
            mask = None
        else:
            mask = self.ref_dir+'/mask_'+hemi+'_'+roi_name+'.nii'
        for run in range(self.num_runs):
            run_tr_targets = np.append(-1*np.ones(self.zscore_trs), self.tr_targets[self.tr_chunks==run])
            run_tr_chunks = np.append(run*np.ones(self.zscore_trs), self.tr_chunks[self.tr_chunks==run])
            datasets.append(fmri_dataset(self.bold_dir+'/rrun-'+str(run+1).zfill(3)+'.nii',
                mask=mask,
                targets=run_tr_targets,
                chunks=run_tr_chunks))
        self.fmri_data = vstack(datasets, a=0)

        self.active_trs = np.zeros(self.trs_per_trial)
        self.active_trs[self.trial_feature_trs[0]-1:self.trial_feature_trs[1]] = 1
        self.active_trs = np.tile(self.active_trs,int(self.num_runs*self.trials_per_run))
        self.trial_regressor = np.tile(range(int(self.num_runs*self.trials_per_run)),(self.trs_per_trial,1)).flatten('F')
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
        self.fmri_data = self.fmri_data[self.fmri_data.targets!=-1]


    def train_classifier(self):
        self.clf.train(self.fmri_data)

    def apply_classifier(self, data):
        self.clf.predict(data)
        return self.clf.ca.estimates

    def cross_validate(self, n_folds):
        train_ratio = 1-1/float(n_folds)
        n_samples = self.fmri_data.nsamples
        train_range = np.arange(n_samples)
        np.random.shuffle(train_range)
        self.out_accs = []
        for fold in range(n_folds):
            test_ex = train_range[int(np.floor(fold*n_samples/float(n_folds))+1):int(np.ceil((fold+1)*n_samples/float(n_folds)))]
            train_ex = np.setdiff1d(train_range,test_ex)
            self.clf.train(self.fmri_data[train_ex])
            out_acc = np.mean(self.clf.predict(self.fmri_data[test_ex])==self.fmri_data[test_ex].targets)
            self.out_accs.append(out_acc)

    def test_time_shift(self, clf, roi_name='ba4a'):
        self.time_shift_means = []
        self.loc_time_shift_out = []
        self.time_shift_ses = []
        for tr in range(1,9):
            self.trial_feature_trs = [tr,tr+2]
            self.extract_features(roi_name)
            self.cross_validate(5)
            self.time_shift_means.append(np.mean(self.out_accs))
            self.time_shift_ses.append(np.std(self.out_accs)/2.)
            self.loc_time_shift_out.append(np.mean(clf.predict(self.fmri_data)==self.fmri_data.targets))

    def save_classifier(self):
        self.clf.voxel_indices = self.fmri_data.fa.voxel_indices
        pickle.dump(self.clf,open(self.ref_dir+'/clf.p','wb'))

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
