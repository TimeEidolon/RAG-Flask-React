from flask_socketio import SocketIO, emit
from flask import current_app

socketio = SocketIO(current_app, cors_allowed_origins="*")


@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('server_response', {'data': 'Connected!'})  # 向客户端发送消息


@socketio.on('client_message')
def handle_message(data):
    print(f"Received message: {data}")
    emit('server_response', {'data': f"Server received: {data}"})  # 回复客户端


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
