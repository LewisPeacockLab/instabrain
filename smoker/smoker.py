import SimpleHTTPServer
import SocketServer
from smoker_watcher import SmokerWatcher
from watchdog.observers import Observer
import multiprocessing as mp
import os, yaml, time, argparse, subprocess

parser = argparse.ArgumentParser(description='Function arguments')
parser.add_argument('-s','--subjectid', help='Subject ID',default='demo')
args = parser.parse_args()

with open('smoker_config.yml') as f:
    CONFIG = yaml.load(f)

try:
    OBS_TIMEOUT = 0.01
    PORT = CONFIG['server-port']
    WATCH_DIR = CONFIG['watch-dir']
    SMOKER_DIR = os.getcwd()
    SERVE_DIR = SMOKER_DIR+'/serve'
    RECON_DIR = CONFIG['recon-server-path']
    RECON_SCRIPT = RECON_DIR+'/'+CONFIG['recon-script']
    CONFIG['serve-dir'] = SERVE_DIR
    CONFIG['subject-id'] = args.subjectid
except:
    print 'Error: config file incomplete/missing'

class SmokerHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        return # silences log messages

    def do_POST(self):
        # template for custom post requests
        try:
            length = int(self.headers.getheader('content-length'))
            data = self.rfile.read(length)
            if self.path == '/set_mode':
                pass
            self.send_response(200, "OK")
            self.finish()
        except:
            pass

def serve_async(httpd):
    while True:
        httpd.handle_request()


if __name__ == "__main__":
    # init and configure smoker server
    request_handler = SmokerHandler
    SocketServer.TCPServer.allow_reuse_address = True 
    SocketServer.TCPServer.timeout = 1.0 

    # start smoker server
    os.chdir(SERVE_DIR)
    httpd = SocketServer.TCPServer(("", PORT),
                                   request_handler)
    server_process = mp.Process(target = serve_async,
                                args = (httpd,))
    server_process.start()

    # start remote recon server
    os.chdir(WATCH_DIR)
    subprocess.Popen(RECON_SCRIPT, shell=True)

    # start realtime watcher
    os.chdir(SMOKER_DIR)
    event_observer = Observer(OBS_TIMEOUT)
    event_handler = SmokerWatcher(CONFIG)
    event_observer.schedule(event_handler,
                            WATCH_DIR,
                            recursive=False)
    event_observer.start()

    # dummy loop for ongoing processes
    while True:
        pass
