import socket
from Loggers.Logger import Logger


class RemoteLogger(Logger):
    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._connection = self._socket.connect((self._host, self._port))

    def log(self, message: str):
        self._socket.sendall(message.encode())

    def log(self, data: dict):
        self._socket.sendall(str(data).encode())

    def close(self):
        self._socket.close()
