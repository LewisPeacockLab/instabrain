import SimpleHTTPServer
import SocketServer
from smoker_watcher import SmokerWatcher
from watchdog.observers import Observer
import multiprocessing as mp
import os, yaml, time, argparse, subprocess

# pip install git+https://github.com/Pithikos/python-websocket-server
from websocket_server import WebsocketServer


#
# Configuration
#

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
    RECON_SCRIPT = CONFIG['recon-server-path']+'/runLocal_CPU.sh'
    CONFIG['serve-dir'] = SERVE_DIR
    CONFIG['subject-id'] = args.subjectid
    WS_SERVER_HOST = CONFIG['ws-server-host']
    WS_SERVER_PORT = CONFIG['ws-server-port']
    WS_SERVER_NAME = CONFIG['ws-server-name']
except:
    print 'Error: config file incomplete/missing'

#
# SimpleHTTPServer Communication Implementation
#

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

#
# Websocket Commmunication Implementation
#

def on_new_client(client, server):
    server.send_message_to_all(SERVER_NAME + ":HUD client connected\n")

def on_client_left(client,server):
    server.send_message_to_all(SERVER_NAME + ":HUD client left\n")

def on_message_received(client,server,message):
    server.send_message_to_all(SERVER_NAME + ":I got this:" + message)

def websocket_server_start():

    server = WebsocketServer(WS_SERVER_PORT, host=WS_SERVER_HOST, loglevel=logging.INFO)

    # setup callbacks

    server.set_fn_new_client(on_new_client)
    server.set_fn_client_left(on_client_left)
    server.set_fn_message_received(on_message_received)

    server.run_forever()

    return server

#
#  Main
#

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

    # start smoker websocket server
    ws_server = websocket_server_start()

    # start remote recon server
    subprocess.Popen(RECON_SCRIPT, shell=True)

    # start realtime watcher
    os.chdir(SMOKER_DIR)
    event_observer = Observer(OBS_TIMEOUT)
    event_handler = SmokerWatcher(CONFIG, ws_server)
    event_observer.schedule(event_handler,
                            WATCH_DIR,
                            recursive=False)
    event_observer.start()

    # dummy loop for ongoing processes
    while True:
        pass
