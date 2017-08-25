from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer
import multiprocessing as mp
import numpy as np
from scipy.signal import detrend
import nibabel as nib
import os, yaml, time, subprocess
import requests as r

class SmokerWatcher(PatternMatchingEventHandler):
    def __init__(self, config):
        PatternMatchingEventHandler.__init__(self, 
            patterns=['*SB*.imgdat.tmp'],
            ignore_patterns=[],
            ignore_directories=True)
        # multiprocessing workers
        self.pool = mp.Pool()

        # timings
        self.trials = config['trials-per-run']
        self.zscore_trs = config['zscore-trs']
        self.cue_trs = config['cue-trs']
        self.wait_trs = config['wait-trs']
        self.feedback_trs = config['feedback-trs']
        self.iti_trs = config['iti-trs']
        self.trial_trs = self.cue_trs+self.wait_trs+self.feedback_trs+self.iti_trs
        self.run_trs = self.zscore_trs+self.trials*self.trial_trs
        self.run_count = 0
        self.moving_avg_trs = config['moving-avg-trs']
        self.trs_to_score_calc = self.cue_trs+self.wait_trs-1
        self.feedback_calc_trs = (self.zscore_trs+self.trs_to_score_calc
                                  +np.arange(self.trials)*self.trial_trs-1)

        # files and directories
        self.subject_id = config['subject-id']
        self.ref_dir = os.getcwd()+'/ref/'+self.subject_id
        self.rfi_file = self.ref_dir+'/rfi.nii'
        self.rfi_img = nib.load(self.rfi_file)
        self.ref_affine = self.rfi_img.get_qform()
        self.ref_header = self.rfi_img.header
        self.clf_file = self.ref_dir+'/clf.nii'
        self.target_class = int(np.loadtxt(self.ref_dir+'/class.txt'))
        self.proc_dir = os.getcwd()+'/proc'
        self.watch_dir = config['watch-dir']

        # logic and initialization
        self.load_clf(self.clf_file)
        self.archive_bool = config['archive-data']
        self.reset_img_arrays()

        # networking
        self.post_url = config['post-url']

    def load_clf(self, filename):
        self.clf_img = nib.load(filename).get_data()
        self.slice_dims = (self.clf_img.shape[0],self.clf_img.shape[1])
        self.num_slices = self.clf_img.shape[2]
        self.clf_voxels = np.where(self.clf_img!=0)
        self.clf_voxels = np.ascontiguousarray(self.clf_voxels)
        self.clf_voxels = np.ascontiguousarray(self.clf_voxels[0:3,:].T)
        self.roi_voxels = np.unique(self.clf_voxels.view([('', self.clf_voxels.dtype)]*self.clf_voxels.shape[1]))
        self.roi_voxels = self.roi_voxels.view(self.clf_voxels.dtype).reshape((self.roi_voxels.shape[0], self.clf_voxels.shape[1]))
        self.num_roi_voxels = np.shape(self.roi_voxels)[0]
        self.classifier = np.zeros((self.num_roi_voxels,np.shape(self.clf_img)[3]))
        for out_class in range(np.shape(self.clf_img)[3]):
            self.classifier[:,out_class] = map_voxels_to_roi(self.clf_img[:,:,:,out_class], self.roi_voxels)

    def reset_img_arrays(self):
        self.img_status_array = np.zeros(self.run_trs)
        self.raw_img_array = np.zeros((self.slice_dims[0],self.slice_dims[1],
            self.num_slices,self.run_trs),dtype=np.uint16)
        self.raw_roi_array = np.zeros((self.num_roi_voxels,self.run_trs))
        self.trial_count = 0
        self.zscore_calc = False
        self.voxel_sigmas = np.zeros(self.num_roi_voxels)

    def on_moved(self, event):
        # is triggered when full .imgdat file received
        img_file = event.src_path.rsplit('/')[-1].rsplit('.tmp')[0]
        rep = int(img_file.split('-R')[1].split('-')[0])-1
        slc = int(img_file.split('-S')[1].split('.')[0])-1
        with open(event.src_path.rsplit('.tmp')[0]) as f:
            self.raw_img_array[:,:,slc,rep] = np.fromfile(f,dtype=np.uint16).reshape(self.slice_dims)
        self.img_status_array[rep] += 1
        if self.img_status_array[rep] == self.num_slices:
            self.pool.apply_async(func = process_volume,
                args = (self.raw_img_array[:,:,:,rep],self.roi_voxels,
                    rep,self.rfi_file,self.proc_dir,self.ref_header,self.ref_affine),
                callback = self.save_processed_roi)

    def save_processed_roi(self, roi_and_rep_data):
        (roi_data,rep) = roi_and_rep_data 
        self.raw_roi_array[:,rep] = roi_data
        if rep == (self.zscore_trs-1):
            self.voxel_sigmas = np.sqrt(np.var(self.raw_roi_array[:,:rep+1],1))
        if rep in self.feedback_calc_trs:
            self.trial_count += 1
            detrend_roi_array = detrend(self.raw_roi_array[:,:rep+1],1)
            zscore_avg_roi = np.mean(detrend_roi_array[:,-self.moving_avg_trs:],1)/self.voxel_sigmas
            clf_out_raw = np.matmul(zscore_avg_roi,self.classifier)
            clf_out_softmax = np.exp(clf_out_raw)/sum(np.exp(clf_out_raw))
            out_data = np.append(clf_out_softmax, self.target_class)
            self.send_clf_outputs(out_data)
        if rep == (self.run_trs-1):
            self.reset_for_next_run()

    def send_clf_outputs(self, out_data):
        payload = {"clf_outs": out_data[:-1],
            "target_class": out_data[-1],
            "trial_num": self.trial_count}
        post_status = 404
        while post_status != 200:
            post_status = r.post(self.post_url, json=payload)

    def reset_for_next_run(self):
        self.run_count += 1
        for target_dir in [self.proc_dir, self.watch_dir]:
            if self.archive_bool:
                run_dir = target_dir+'/run_'+str(self.run_count).zfill(2)
                os.mkdir(run_dir)
                os.system('mv '+target_dir+'/*.* '+run_dir+' 2> /dev/null')
            os.system('rm '+target_dir+'/*.* 2> /dev/null')
        self.reset_img_arrays()

# standalone functions
def process_volume(raw_img, roi_voxels, rep, rfi_file, proc_dir, ref_header, ref_affine):
    temp_file = proc_dir + '/img_' + str(rep+1).zfill(3) + '.nii.gz'
    mc_file = proc_dir + '/img_mc_' + str(rep+1).zfill(3) + '.nii.gz'
    nib.save(nib.Nifti1Image(raw_img, ref_affine, header=ref_header), temp_file)
    os.system('mcflirt -in '+temp_file+' -dof 6 -reffile '+rfi_file+' -out '+mc_file)
    roi_data = map_voxels_to_roi(nib.load(mc_file).get_data(),roi_voxels)
    return (roi_data, rep)

def map_voxels_to_roi(img, roi_voxels):
    out_roi = np.zeros(roi_voxels.shape[0])
    for voxel in range(roi_voxels.shape[0]):
        out_roi[voxel] = img[roi_voxels[voxel,0],roi_voxels[voxel,1],roi_voxels[voxel,2]]
    return out_roi

def start_watcher(CONFIG, subject_id):
    CONFIG['subject-id'] = subject_id
    OBS_TIMEOUT = 0.01
    event_observer = Observer(OBS_TIMEOUT)
    event_handler = SmokerWatcher(CONFIG)
    event_observer.schedule(event_handler,
                            CONFIG['watch-dir'],
                            recursive=False)
    event_observer.start()

def start_remote_recon(CONFIG):
    RECON_SCRIPT = CONFIG['recon-script']
    os.chdir(CONFIG['watch-dir'])
    subprocess.Popen(RECON_SCRIPT, shell=True)

if __name__ == "__main__":
    # load subject ID from args
    import argparse
    parser = argparse.ArgumentParser(description='Function arguments')
    parser.add_argument('-s','--subjectid', help='Subject ID',default='demo')
    args = parser.parse_args()

    # load config
    with open('insta_config.yml') as f:
        CONFIG = yaml.load(f)

    # start remote recon server
    if not(CONFIG['debug-bool']):
        start_remote_recon(CONFIG)

    # start realtime watcher
    if CONFIG['debug-bool']:
        CONFIG['watch-dir'] = '../data/dump'
    start_watcher(CONFIG, args.subjectid)

    # dummy loop for ongoing processes
    while True:
        pass
