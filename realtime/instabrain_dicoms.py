from watchdog.events import PatternMatchingEventHandler
# from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
import multiprocessing as mp
import numpy as np
from scipy.signal import detrend
import nibabel as nib
from nipype.interfaces.dcm2nii import Dcm2niix
import os, sys, glob, yaml, time, pickle, subprocess
import requests as r
# from dashboard import post_dashboard_mc_params, post_dashboard_clf_outs

REALTIME_TIMEOUT = 0.1 # seconds to HTTP post timeout

class InstaWatcher(PatternMatchingEventHandler):
    def __init__(self, config):
        file_pattern = '*.' + config['subject-id'] + '_' + str(config['session-number']) + '/*.dcm'
        print('Looking for file pattern: ' + file_pattern)
        PatternMatchingEventHandler.__init__(self, 
            patterns=[file_pattern],
            ignore_patterns=[],
            ignore_directories=False)

        # multiprocessing workers
        self.pool = mp.Pool()

        # timings
        self.run_count = int(config['start-run'])-1
        self.baseline_trs = config['baseline-trs']
        try:
            feedback_mode = config['feedback-mode'].lower()
        except:
            feedback_mode = 'continuous'
        if feedback_mode == 'continuous':
            self.feedback_trs = config['feedback-trs']
            self.run_trs = self.baseline_trs+self.feedback_trs
            self.feedback_calc_trs = np.arange(self.baseline_trs,self.feedback_trs+self.baseline_trs)
        elif feedback_mode == 'intermittent':
            self.trials = config['trials-per-run']
            self.cue_trs = config['cue-trs']
            self.wait_trs = config['wait-trs']
            self.feedback_trs = config['feedback-trs']
            self.iti_trs = config['iti-trs']
            self.trial_trs = self.cue_trs+self.wait_trs+self.feedback_trs+self.iti_trs
            self.run_trs = self.baseline_trs+self.trials*self.trial_trs
            self.trs_to_score_calc = self.cue_trs+self.wait_trs-1
            self.feedback_calc_trs = (self.baseline_trs+self.trs_to_score_calc
                                      +np.arange(self.trials)*self.trial_trs-1)

        # data processing
        self.moving_avg_trs = config['moving-avg-trs']
        self.mc_mode = config['mc-mode'].lower()

        # files and directories
        self.subject_id = config['subject-id']
        self.ref_dir = os.getcwd()+'/ref/'+self.subject_id
        self.rfi_file = glob.glob(self.ref_dir+'/*rfi.nii')[0]
        self.rfi_img = nib.load(self.rfi_file)
        self.rfi_data = self.rfi_img.get_data()
        self.ref_affine = self.rfi_img.get_qform()
        self.ref_header = self.rfi_img.header
        self.clf_file = glob.glob(self.ref_dir+'/*clf.p')[0]
        # optional: target class can be specified in backend
        try:
            self.target_class = int(np.loadtxt(self.ref_dir+'/class.txt'))
        except:
            self.target_class = -1
        self.proc_dir = os.getcwd()+'/proc'
        self.watch_dir = config['watch-dir']

        # logic and initialization
        self.slice_dims = (self.rfi_data.shape[0],self.rfi_data.shape[1])
        self.num_slices = self.rfi_data.shape[2]
        self.clf = pickle.load(open(self.clf_file,'rb'))
        self.num_roi_voxels = np.shape(self.clf.voxel_indices)[0]
        self.archive_bool = config['archive-data']
        self.archive_dir = config['archive-dir']
        self.reset_img_arrays()

        # networking
        self.post_url = config['post-url']

        # event logging
        if config['logging_bool']:
            self.logging_bool = True
            self.log_file_time = config['log_file_time']
            log_file_name = os.getcwd()+'/log/'+str(self.log_file_time)+'_event.log'
            self.log_file = open(os.path.normpath(log_file_name),'w')
            write_log_header(self.log_file)
            self.log_file_time = 1.539e9 #hacky
            write_log(self.log_file, self.log_file_time, 'startup', 0)
        else:
            self.logging_bool = False

        # dashboard display
        if config['dashboard_bool']:
            try:
                self.dashboard_base_url = config['dashboard-base-url']
                self.dashboard_mc_url = self.dashboard_base_url+'/mc_data'
                self.dashboard_clf_url = self.dashboard_base_url+'/clf_data'
                self.dashboard_shutdown_url = self.dashboard_base_url+'/shutdown'
                self.try_dashboard_connection = True
                self.dashboard_bool = True
            except:
                self.try_dashboard_connection = False
                self.dashboard_bool = False
        else:
            self.try_dashboard_connection = False
            self.dashboard_bool = False

    def apply_classifier(self, data):
        if not(self.logging_bool):
            self.clf.predict(np.ndarray((1,self.num_roi_voxels),buffer=data))
        else:
            self.clf.predict(np.random.normal(0,1,(1,self.num_roi_voxels)))
        return self.clf.ca.estimates

    def reset_img_arrays(self):
        self.img_status_array = np.zeros(self.run_trs)
        self.raw_img_array = np.zeros((self.slice_dims[0],self.slice_dims[1],
            self.num_slices,self.run_trs),dtype=np.uint16)
        self.raw_roi_array = np.zeros((self.num_roi_voxels,self.run_trs))
        self.feedback_count = -1
        self.zscore_calc = False
        self.voxel_sigmas = np.zeros(self.num_roi_voxels)

    def on_created(self,event):
        # is triggered by PatternMatchingEventHandler when dicom is created
        print(event.src_path)
        img_dir = event.src_path.rsplit('/', 1)[0]
        # print('dicom directory: ' + img_dir)
        img_file = event.src_path.rsplit('/')[-1]
        # print('dicom to convert: ' + img_file)
        rep = int(img_file.split('_')[-1].split('.dcm')[0])-1
        self.raw_nii = convert_dicom_to_nii(img_file,self.proc_dir,img_dir,img_file.split('_')[-1].split('.dcm')[0])
        os.system('rm ' + self.proc_dir + '/*.dcm')

        # self.raw_nii = convert_dicom_to_nii(event.src_path,self.proc_dir,img_dir)

        if self.raw_nii.split('_')[0] == 'epfid2d1':
            if self.logging_bool: write_log(self.log_file, self.log_file_time, 'mc_start', rep)
            self.pool.apply_async(func = process_volume,
                args = (self.raw_nii,self.clf.voxel_indices,
                    rep,self.rfi_file,self.proc_dir,self.ref_header,
                    self.ref_affine, self.mc_mode),
                callback = self.save_processed_roi)

    # def on_moved(self, event):
    #     # is triggered when full .imgdat file received
    #     img_file = event.src_path.rsplit('/')[-1].rsplit('.tmp')[0]
    #     rep = int(img_file.split('-R')[1].split('-')[0])-1
    #     slc = int(img_file.split('-S')[1].split('.')[0])-1
    #     with open(event.src_path.rsplit('.tmp')[0]) as f:
    #         self.raw_img_array[:,:,slc,rep] = np.rot90(np.fromfile(f,dtype=np.uint16).reshape(self.slice_dims),k=-1)
    #     self.img_status_array[rep] += 1
    #     if self.logging_bool: write_log(self.log_file, self.log_file_time, 'slc_'+str(slc), rep)
    #     if self.img_status_array[rep] == self.num_slices:
    #         if self.logging_bool: write_log(self.log_file, self.log_file_time, 'mc_start', rep)
    #         self.pool.apply_async(func = process_volume,
    #             args = (self.raw_img_array[:,:,:,rep],self.clf.voxel_indices,
    #                 rep,self.rfi_file,self.proc_dir,self.ref_header,
    #                 self.ref_affine, self.mc_mode),
    #             callback = self.save_processed_roi)

    def save_processed_roi(self, roi_and_rep_data):
        (roi_data,rep) = roi_and_rep_data 
        if self.logging_bool: write_log(self.log_file, self.log_file_time, 'mc_end', rep)
        self.raw_roi_array[:,rep] = roi_data
        if rep == (self.baseline_trs-1):
            self.voxel_sigmas = np.sqrt(np.var(self.raw_roi_array[:,:rep+1],1))
            self.voxel_sigmas[np.where(self.voxel_sigmas==0)] = 1e6 # ignore 'zero variance' voxels
        if rep in self.feedback_calc_trs:
            self.feedback_count += 1
            detrend_roi_array = detrend(self.raw_roi_array[:,:rep+1],1)
            zscore_avg_roi = np.mean(detrend_roi_array[:,-self.moving_avg_trs:],1)/self.voxel_sigmas
            clf_out = self.apply_classifier(zscore_avg_roi)
            out_data = np.append(clf_out, self.target_class)
            self.send_clf_outputs(out_data)
            # print('fb sent')
            if self.logging_bool: write_log(self.log_file, self.log_file_time, 'fb_sent', rep)

        # dashboard outputs 
        if self.dashboard_bool:
            try:
                self.check_for_dashboard(rep)
            except:
                self.dashboard_bool = False

        if rep == (self.run_trs-1):
            self.reset_for_next_run()

    def send_clf_outputs(self, out_data):
        payload = {"clf_outs": list(out_data[:-1]),
            "target_class": out_data[-1],
            "feedback_num": self.feedback_count}
        try:
            r.post(self.post_url, json=payload, timeout=REALTIME_TIMEOUT)
        except:
            pass

    def check_for_dashboard(self, rep):
        if rep >= (self.baseline_trs-1):
            detrend_roi_array = detrend(self.raw_roi_array[:,:rep+1],1)
            zscore_avg_roi = np.mean(detrend_roi_array[:,-self.moving_avg_trs:],1)/self.voxel_sigmas
            clf_out = self.apply_classifier(zscore_avg_roi)
            post_dashboard_clf_outs(clf_out[0], rep, self.dashboard_clf_url)
        mc_params_file = self.proc_dir +'/mc_params_' + str(rep+1).zfill(3) + '.txt'            
        mc_params = np.loadtxt(mc_params_file)
        post_dashboard_mc_params(mc_params, rep, self.dashboard_mc_url)

    def reset_for_next_run(self):
        if self.try_dashboard_connection:
            self.dashboard_bool = True
        self.run_count += 1
        # for target_dir in [self.proc_dir, self.watch_dir]:
        if self.archive_bool:
            for target_dir in [self.proc_dir, self.watch_dir]:
                if self.archive_bool:
                    run_dir = self.archive_dir+'/run_'+str(self.run_count).zfill(2)
                    if not(os.path.exists(run_dir)):
                        os.mkdir(run_dir)
                    if target_dir == self.proc_dir:
                        os.system('cat '+target_dir+'/*.txt > '+run_dir+'/mc_params.txt 2>/dev/null')
                        os.system('rm '+target_dir+'/*.txt 2>/dev/null')
                        os.system('mv '+target_dir+'/*.* '+run_dir+' 2>/dev/null')
                    elif target_dir == self.watch_dir:
                        os.system('mv '+self.watch_dir+'/*/*.dcm '+run_dir+' 2>/dev/null')
                os.system('rm '+target_dir+'/*.* 2>/dev/null')

            # run_dir = self.proc_dir+'/run_'+str(self.run_count).zfill(2)
            # if not(os.path.exists(run_dir)):
            #     os.mkdir(run_dir)
            # os.system('cat '+self.proc_dir+'/*.txt > '+run_dir+'/mc_params.txt 2>/dev/null')
            # os.system('rm '+self.proc_dir+'/*.txt 2>/dev/null')
            # os.system('mv '+self.proc_dir+'/*.* '+run_dir+' 2>/dev/null')
            # os.system('mv '+self.watch_dir+'/*/*.dcm '+run_dir+' 2>/dev/null')
        self.reset_img_arrays()

# standalone functions
def process_volume(raw_nii, roi_voxels, rep, rfi_file,
        proc_dir, ref_header, ref_affine, mc_mode):
    temp_file = proc_dir +'/'+raw_nii #proc_dir + '/img_' + str(rep+1).zfill(3) + '.nii.gz'
    mc_file = proc_dir + '/img_mc_' + str(rep+1).zfill(3) + '.nii.gz'
    mc_params_file = proc_dir +'/mc_params_' + str(rep+1).zfill(3) + '.txt'
    # nib.save(nib.Nifti1Image(raw_img, ref_affine, header=ref_header), temp_file)
    if mc_mode == 'afni':
        os.system('3dvolreg -prefix '+mc_file+' -base '+rfi_file+' -1Dfile '+mc_params_file+' '+temp_file+' 2>/dev/null')
    elif mc_mode == 'fsl':
        os.system('mcflirt -in '+raw_nii+' -dof 6 -reffile '+rfi_file+' -out '+mc_file)
    elif mc_mode == 'none':
        os.system('cp '+raw_nii+' '+mc_file)
    roi_data = map_voxels_to_roi(nib.load(mc_file).get_data(),roi_voxels)
    return (roi_data, rep)

def map_voxels_to_roi(img, roi_voxels):
    out_roi = np.zeros(roi_voxels.shape[0])
    for voxel in range(roi_voxels.shape[0]):
        out_roi[voxel] = img[roi_voxels[voxel,0],roi_voxels[voxel,1],roi_voxels[voxel,2]]
    return out_roi

def start_watcher(CONFIG, subject_id, session, config_name, debug_bool=False, logging_bool=False,
        dashboard_bool=False, start_run=1):
    CONFIG['subject-id'] = subject_id
    CONFIG['session-number'] = session
    CONFIG['config-name'] = config_name
    CONFIG['debug-bool'] = debug_bool
    CONFIG['logging_bool'] = logging_bool
    CONFIG['dashboard_bool'] = dashboard_bool
    CONFIG['start-run'] = start_run
    OBS_TIMEOUT = 0.01
    event_observer = PollingObserver(OBS_TIMEOUT)
    # event_observer = Observer(OBS_TIMEOUT)
    event_handler = InstaWatcher(CONFIG)
    event_observer.schedule(event_handler,
                            CONFIG['watch-dir'],
                            recursive=True)
    event_observer.start()

# def start_remote_recon(CONFIG):
#     RECON_SCRIPT = CONFIG['recon-script']
#     os.chdir(CONFIG['watch-dir'])
#     subprocess.Popen(RECON_SCRIPT, shell=True)

def convert_dicom_to_nii(dcm_file,nii_outdir,dcm_dir,TR_num):
    os.system('cp ' + dcm_dir + '/' + dcm_file + ' ' + nii_outdir)
    os.chdir(nii_outdir)
    converter = Dcm2niix()
    converter.inputs.source_names = dcm_file
    converter.inputs.output_dir = nii_outdir
    converter.inputs.compress = 'n'
    converter.inputs.single_file = True
    converter.inputs.out_filename = '%z_%s_' + TR_num
    # print(converter.cmdline)
    converter.run()
    new_nii_file = converter.output_files[0].split('/')[-1]
    return new_nii_file

def write_log_header(log_file):
    log_file.write('time,event,tr\n')

def write_log(log_file, start_time, event_name, count):
    log_file.write(str(time.time()-start_time)+','+event_name+','+str(count)+'\n')

if __name__ == "__main__":
    # load initialization parameters from args
    import argparse
    parser = argparse.ArgumentParser(description='Function arguments')
    parser.add_argument('-s','--subjectid', help='Subject ID',default='demo')
    parser.add_argument('-sess','--session',help='Scan session number (2 or 3)',required=True)
    parser.add_argument('-c','--config', help='Configuration file',default='default')
    parser.add_argument('-d','--debug', help='Debugging boolean', action='store_true', default=False)
    parser.add_argument('-l','--logging', help='Logging boolean', action='store_true', default=False)
    parser.add_argument('-b','--dashboard', help='Dashboard display boolean', action='store_true', default=False)
    parser.add_argument('-r','--startrun', help='Starting run number',default='1')
    args = parser.parse_args()

    # load config
    with open('config/'+args.config+'.yml') as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)

    # set debug parameters
    if args.debug:
        CONFIG['watch-dir'] = '/Users/el27322/Documents/instabrain/data/dump' #'../data/dump'

    # set logging parameters and start logging 
    if args.logging:
        CONFIG['logging_bool'] = True
        CONFIG['log_file_time'] = int(np.floor(time.time()))

    # start realtime watcher
    start_watcher(CONFIG, args.subjectid, args.session, args.config, args.debug, args.logging, args.dashboard, args.startrun)

    # # start remote recon server
    # if not(args.debug):
    #     start_remote_recon(CONFIG)

    # dummy loop for ongoing processes (or logging)
    while True:
        pass
