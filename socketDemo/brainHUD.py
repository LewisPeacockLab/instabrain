import websocket
import thread

SERVER_NAME = "ws://127.0.0.1:8080"
CLIENT_NAME = 'HUD'

def on_message(ws, message):
    print CLIENT_NAME + ":" + message

def on_error(ws, error):
    print error

def on_close(ws):
    print CLIENT_NAME + ": closed"

def on_open(ws):
    def run(*args):
        print CLIENT_NAME + ": connected"
        print CLIENT_NAME + ": start"
        ws.send("start")
    thread.start_new_thread(run, ())

# def send_messsage(ws):
#     ws.send("start")

# setup connection
ws = websocket.WebSocketApp(SERVER_NAME,
                            on_message=on_message,
                            on_error=on_error,
                            on_close=on_close)
ws.on_open = on_open
ws.run_forever(sslopt={"check_hostname": False})
websocket.enableTrace(True)


