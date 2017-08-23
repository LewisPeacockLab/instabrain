from smoker_watcher import SmokerWatcher
from watchdog.observers import Observer
from smoker_watcher import start_watcher
import multiprocessing as mp
import os, yaml, time, argparse, subprocess

parser = argparse.ArgumentParser(description='Function arguments')
parser.add_argument('-s','--subjectid', help='Subject ID',default='demo')
args = parser.parse_args()

with open('smoker_config.yml') as f:
    CONFIG = yaml.load(f)

try:
    OBS_TIMEOUT = 0.01
    if CONFIG['debug-bool']:
        WATCH_DIR = '../data/dump'
    else:
        WATCH_DIR = CONFIG['watch-dir']
    SMOKER_DIR = os.getcwd()
    SERVE_DIR = SMOKER_DIR+'/serve' # not necessary in final version
    CONFIG['serve-dir'] = SERVE_DIR # not necessary in final version
    RECON_DIR = CONFIG['recon-server-path']
    RECON_SCRIPT = RECON_DIR+'/'+CONFIG['recon-script']
    CONFIG['subject-id'] = args.subjectid
except:
    print('Error: config file incomplete/missing')


if __name__ == "__main__":
    # start remote recon server
    if not(CONFIG['debug-bool']):
        os.chdir(WATCH_DIR)
        subprocess.Popen(RECON_SCRIPT, shell=True)

    # set up data structures
    feedback_values = mp.Array('d', CONFIG['trials-per-run'])

    # start realtime watcher
    os.chdir(SMOKER_DIR)
    event_observer = Observer(OBS_TIMEOUT)
    event_handler = SmokerWatcher(CONFIG, feedback_values)
    event_observer.schedule(event_handler,
                            WATCH_DIR,
                            recursive=False)
    event_observer.start()

    # dummy loop for ongoing processes
    while True:
        # print(feedback_values[0])
        pass
