import multiprocessing as mp
import numpy as np
import threading as t
import requests as r
import matplotlib.pyplot as plt
import yaml, time
from flask import Flask, request

POST_CLF_URL = 'http://127.0.0.1:5000/clf_data'
POST_MC_URL = 'http://127.0.0.1:5000/mc_data'
SHUTDOWN_URL = 'http://127.0.0.1:5000/shutdown'

class InstaDashboard(object):
    def __init__(self, config_name='fingtrain', num_classes=4, num_mc_params=6, max_mc_display=5, fd_scale_factor=2):
        self.post_clf_url = POST_CLF_URL
        self.post_mc_url = POST_MC_URL
        self.shutdown_url = SHUTDOWN_URL

        self.num_classes = num_classes
        self.num_mc_params = num_mc_params
        self.max_mc_display = max_mc_display # millimeters
        self.fd_scale_factor = fd_scale_factor # FD only displays positive numbers, increasing ylim
        self.set_constants()
        self.start_dashboard_server()
        self.reset_clf_indices()

        config = load_config(config_name)
        self.run_count = 0
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
        self.reset_clf_indices()
        self.init_plot()

    def set_constants(self):
        self.CLF_COLORS = ['gold','seagreen','dodgerblue','violet'] # add more colors for more classes lol
        self.MC_COLORS = ['red',np.repeat(.3,3),np.repeat(.4,3),np.repeat(.5,3),np.repeat(.6,3),np.repeat(.7,3),np.repeat(.8,3)]

    def plot_clf_tr(self, tr, clf_outs=None):
        if clf_outs == None:
            clf_outs = np.random.rand(self.num_classes)
        plt.sca(self.clf_ax)
        if tr<self.baseline_trs:
            pass
        if tr==self.baseline_trs:
            self.last_clf_outs = 0.5*np.ones(self.num_classes)
        else:
            for class_num in range(self.num_classes):
                plt.plot([tr-1,tr],[self.last_clf_outs[class_num],clf_outs[class_num]],color=self.CLF_COLORS[class_num])
        self.last_clf_outs[:] = clf_outs

    def plot_mc_tr(self, tr, mc_params=None):
        if mc_params == None:
            mc_params = np.random.rand(self.num_mc_params)
            mc_params = mc_params-np.mean(mc_params)
        mc_params_mm = estimate_framewise_displacement(mc_params) # this includes FD, mc_params_mm
        plt.sca(self.mc_ax)
        if tr==0:
            self.last_mc_params = np.zeros(self.num_mc_params+1) # +1 to include FD
        else:
            for mc_param in range(self.num_mc_params+1):
                plt.plot([tr-1,tr],[self.last_mc_params[mc_param], mc_params_mm[mc_param]],color=self.MC_COLORS[mc_param])
        self.last_mc_params[:] = mc_params_mm 

    def plot_feedback_tr(self, tr):
        # just plot a rectangle on top of decoder output plot, marking feedback TRs
        pass

    def init_plot(self):
        plt.ion()
        self.clf_ax = plt.subplot(2,1,1)
        self.mc_ax = plt.subplot(2,1,2)
        plt.subplots_adjust(hspace=.4, right=.8)
        for ax in (self.clf_ax, self.mc_ax):
            plt.sca(ax)
            plt.xlim((0,self.run_trs))
            if ax == self.clf_ax:
                plt.title('Decoder outputs')
                plt.ylabel('probability')
                plt.ylim((0,1))
                for dummy_plot in range(self.num_classes):
                    plt.plot(-1,-1,color=self.CLF_COLORS[dummy_plot])
                plt.legend([('class '+str(class_num)) for class_num in range(self.num_classes)],bbox_to_anchor=(1.27,1.0))
            elif ax == self.mc_ax:
                plt.title('Motion parameters')
                plt.ylabel('displacement (mm)')
                plt.ylim((-self.max_mc_display,self.fd_scale_factor*self.max_mc_display))
                for dummy_plot in range(self.num_mc_params+1):
                    plt.plot(-1,-1,color=self.MC_COLORS[dummy_plot])
                plt.legend(['FD','x','y','z','a','b','c'],bbox_to_anchor=(1.23,1.0))

    def demo_realtime(self, sleep_time=0.1):
        for tr in range(self.run_trs):
            mc_payload = {"mc_params": list(np.random.rand(self.num_mc_params).flatten()),
                          "tr_num": int(tr)}
            status_code = 404
            while status_code != 200:
               post_status = r.post(self.post_mc_url, json=mc_payload)
               status_code = post_status.status_code
            status_code = 404
            if tr >= self.baseline_trs:
                clf_payload = {"clf_outs": list(np.random.rand(self.num_classes).flatten()),
                               "tr_num": int(tr)}
                status_code = 404
                while status_code != 200:
                    post_status = r.post(self.post_clf_url, json=clf_payload)
                    status_code = post_status.status_code
            self.check_for_new_data()
            time.sleep(sleep_time)
        self.reset_for_next_run()
        self.shutdown_server()

    def post_mc_params(self, mc_params, rep):
        mc_payload = {"mc_params": list(clf_outs),
                       "tr_num": int(rep)}
        r.post(self.post_mc_url, json=mc_payload)

    def post_clf_outs(self, clf_outs, rep):
        clf_payload = {"clf_outs": list(clf_outs),
                       "tr_num": int(rep)}
        r.post(self.post_clf_url, json=clf_payload)

    def demo_realtime_offline(self, sleep_time=0.1):
        for tr in range(self.run_trs):
            self.plot_mc_tr(tr)
            if tr >= self.baseline_trs:
                self.plot_clf_tr(tr)
            if sleep_time is not None:
                plt.pause(sleep_time)

    def check_for_new_data(self):
        self.check_for_mc_data()
        self.check_for_clf_data()
        plt.pause(0.001)

    def check_for_clf_data(self):
        if self.clf_tr_num_shared.value > self.clf_tr_num:
            self.plot_clf_tr(tr=self.clf_tr_num_shared.value,clf_outs=self.clf_outs[:])
            self.clf_tr_num+=1

    def check_for_mc_data(self):
        if self.mc_tr_num_shared.value > self.mc_tr_num:
            self.plot_mc_tr(tr=self.mc_tr_num_shared.value,mc_params=self.mc_params[:])
            self.mc_tr_num+=1

    def reset_for_next_run(self):
        self.clear_plot()
        self.reset_clf_indices()

    def reset_clf_indices(self):
        self.clf_tr_num = -1
        self.mc_tr_num = -1
        self.clf_tr_num_shared.value = -1
        self.mc_tr_num_shared.value = -1

    def clear_plot(self):
        for ax in [self.clf_ax, self.mc_ax]:
            # when ax.lines is large, only addresses half of the lines (fix me pls?)
            while(len(ax.lines)>0):
                for line in ax.lines:
                    line.remove()

    def start_dashboard_server(self):
        self.clf_tr_num_shared = mp.Value('i', -1)
        self.mc_tr_num_shared = mp.Value('i', -1)
        self.clf_outs = mp.Array('d', self.num_classes)
        self.mc_params = mp.Array('d', self.num_mc_params)
        self.status_display_thread = t.Thread(target=start_server,
                                         args=(self.clf_tr_num_shared,self.mc_tr_num_shared,self.clf_outs,self.mc_params))
        self.status_display_thread.start()

    def shutdown_server(self):
        r.post(self.shutdown_url)

def load_config(config_name='default'):
    with open('../realtime/config/'+config_name+'.yml') as f:
        return yaml.load(f)

def estimate_framewise_displacement(mc_params, gm_radius=50):
    # Framewise displacement definition from Power et al. 2012:
        # Differentiating head realignment parameters across frames yields a six dimensional
        # timeseries that represents instantaneous head motion. To express instantaneous head motion
        # as a scalar quantity we used the empirical formula, sum(abs(mc_params)). Rotational displacements were
        # converted from degrees to millimeters by calculating displacement on the surface of a sphere
        # of radius 50 mm, which is approximately the mean distance from the cerebral cortex to the
        # center of the head.
    mc_params_mm = np.array(mc_params,dtype=float)
    mc_params_mm[3:] = gm_radius*np.tan(np.deg2rad(mc_params[3:]))
    return np.append(np.sum(np.abs(mc_params_mm)), mc_params_mm)

def start_server(clf_tr_num_shared, mc_tr_num_shared, clf_outs, mc_params):
    app = Flask(__name__)

    app.clf_tr_num_shared = clf_tr_num_shared 
    app.mc_tr_num_shared = mc_tr_num_shared 
    app.clf_outs = clf_outs
    app.mc_params = mc_params 

    @app.route('/clf_data', methods=['POST'])
    def clf_data():
        data = request.get_json(force=True)
        app.clf_outs[:] = data['clf_outs'][:]
        app.clf_tr_num_shared.value = data['tr_num']
        return 'clf_data_received'
    @app.route('/mc_data', methods=['POST'])
    def mc_data():
        data = request.get_json(force=True)
        app.mc_params[:] = data['mc_params'][:]
        app.mc_tr_num_shared.value = data['tr_num']
        return 'mc_data_received'
    @app.route('/shutdown', methods=['POST'])
    def shutdown():
        shutdown_func = request.environ.get('werkzeug.server.shutdown')
        shutdown_func()
        return 'Server shutting down...' 
    app.run(host='0.0.0.0')
