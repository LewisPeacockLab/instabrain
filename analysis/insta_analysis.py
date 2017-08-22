import os, yaml
import nibabel as nib
import numpy as np
from scipy.signal import detrend

with open('analysis_config.yml') as f:
    CONFIG = yaml.load(f)
data_dir = CONFIG['subjs-dir']+CONFIG['nfb-subdir']
trs_per_run = CONFIG['trs-per-run']
num_runs = CONFIG['num-runs']
rfi_img = CONFIG['subjs-dir']+'/ref/'+CONFIG['subject-id']+'/rfi.nii'
ref_affine = nib.load(rfi_file).get_qform
run_data_raw = np.zeros((np.shape(rfi_img)[0],np.shape(rfi_img)[1],np.shape(rfi_img)[2],num_runs))
run_data_proc = np.zeros(run_data_raw.shape)
# load mask and create masked ROI array?
# to avoid divide by zero; maybe just do brain mask?

for run in range(num_runs):
    run_file = data_dir+'/run-'+str(run+1).zfill(3)+'.nii'
    out_file = data_dir+'/rt-run-'+str(run+1).zfill(3)+'.nii'
    run_header = nib.load(run_file).header
    os.system('fslsplit '+run_file)
    for tr in range(trs_per_run):
        vol_img = data_dir+'/vol'+str(tr).zfill(4)+'.nii.gz'
        mc_img = proc_dir+'/mc-vol-'+str(tr).zfill(3)+'.nii.gz'
        os.system('mcflirt -in '+vol_img+' -dof 6 -reffile '+rfi_img+' -out '+mc_img)
        run_data_raw[:,:,:,tr] = nib.load(mc_img).get_data()
        # broadcast to ROI and reshape?
        if tr == (CONFIG['zscore-trs']-1):
            # process only ROI?
            # broadcast arrays here [:,:,:,np.newaxis]
            voxel_sigmas = np.sqrt(np.var(self.run_data_raw[:,:,:,:tr+1],3))
            run_data_proc[:,:,:,:tr+1] = detrend(self.run_data_raw[:,:,:,:tr+1],3)/voxel_sigmas
        if tr >= CONFIG['zscore-trs']:
            detrend_voxel_array = detrend(self.run_data_raw[:,:,:,:tr+1],3)
            # process only ROI?
            # broadcast arrays here [:,:,:,np.newaxis]
            # remove mean here, let matlab do temporal filtering
            # map from ROI back to run_data_proc
            run_data_proc[:,:,:,tr] = np.mean(detrend_voxel_array[:,:,:,-moving_avg_trs:],3)/voxel_sigmas
    # save run_data_proc
    nib.save(nib.Nifti1Image(run_data_proc, ref_affine, header=run_header), out_file)
    # rm vol*.nii

