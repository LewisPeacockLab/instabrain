from smoker_watcher import start_watcher
import os, yaml, time, argparse, subprocess

parser = argparse.ArgumentParser(description='Function arguments')
parser.add_argument('-s','--subjectid', help='Subject ID',default='demo')
args = parser.parse_args()

with open('smoker_config.yml') as f:
    CONFIG = yaml.load(f)

if CONFIG['debug-bool']:
    CONFIG['watch-dir'] = '../data/dump'
RECON_DIR = CONFIG['recon-server-path']
RECON_SCRIPT = RECON_DIR+'/'+CONFIG['recon-script']
CONFIG['subject-id'] = args.subjectid

if __name__ == "__main__":
    # start remote recon server
    if not(CONFIG['debug-bool']):
        os.chdir(CONFIG['watch-dir'])
        subprocess.Popen(RECON_SCRIPT, shell=True)

    # start realtime watcher
    start_watcher(CONFIG)

    # dummy loop for ongoing processes
    while True:
        pass
