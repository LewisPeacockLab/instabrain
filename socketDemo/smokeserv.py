import logging
from websocket_server import WebsocketServer


SERVER_PORT = 8080

# remeber that for all clients to be able to connect this needs to be 0.0.0.0 not 127.0.0.1f
SERVER_HOST = '0.0.0.0'
SERVER_NAME = 'SMOKER'


def on_new_client(client, server):
    server.send_message_to_all(SERVER_NAME + ":HUD client connected\n")

def on_client_left(client,server):
    server.send_message_to_all(SERVER_NAME + ":HUD client left\n")

def on_message_received(client,server,message):
    server.send_message_to_all(SERVER_NAME + ":I got this:" + message)

def start_server():

    server = WebsocketServer(SERVER_PORT, host=SERVER_HOST, loglevel=logging.INFO)

    # setup callbacks

    server.set_fn_new_client(on_new_client)
    server.set_fn_client_left(on_client_left)
    server.set_fn_message_received(on_message_received)

    server.run_forever()


start_server()
