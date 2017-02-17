import SimpleHTTPServer
import SocketServer
from smoker_watcher import SmokerWatcher
from watchdog.observers import Observer
import multiprocessing as mp
import os, yaml, json, time

with open('smoker_config.yaml') as f:
    CONFIG = yaml.load(f)

try:
    OBS_TIMEOUT = 0.01
    PORT = CONFIG['server-port']
    WATCH_DIR = CONFIG['watch-dir']
    SERVE_DIR = CONFIG['serve-dir']
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
    request_handler = SmokerHandler
    SocketServer.TCPServer.allow_reuse_address = True 
    SocketServer.TCPServer.timeout = 1.0 
    SMOKER_DIR = os.getcwd()
    os.chdir(SERVE_DIR)
    httpd = SocketServer.TCPServer(("", PORT),
                                   request_handler)

    server_process = mp.Process(target = serve_async,
                                args = (httpd,))
    server_process.start()

    os.chdir(SMOKER_DIR)
    event_observer = Observer(OBS_TIMEOUT)
    event_handler = SmokerWatcher(CONFIG)
    event_observer.schedule(event_handler,
                            WATCH_DIR,
                            recursive=False)
    event_observer.start()
    while True:
        pass
