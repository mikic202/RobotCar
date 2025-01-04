import socket
from Loggers.Logger import Logger
from typing import Any


class RemoteLogger(Logger):
    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._connection = self._socket.connect((self._host, self._port))

    def log(self, data: list[dict[str, Any]], sample_index: int) -> None:
        self._socket.sendall(str(data).encode())

    def close(self) -> None:
        self._socket.close()
