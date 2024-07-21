from typing import List
import socket

CONTROLL_IP = "192.168.0.167"
CONTROLL_PORT = 8080


class RemoteRegulator:
    def __init__(self) -> None:
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.bind((CONTROLL_IP, CONTROLL_PORT))
        self._server_socket.listen(1)
        self._client_socket, _ = self._server_socket.accept()
        self._client_reader = self._client_socket.makefile("r")
        self._client_reader.readline()

    def get_controll(self, input: List[float]) -> List[float]:
        request = self._client_reader.readline()  # convert bytes to string
        values = str(request).strip().split(";")
        return [float(val) / 100 for val in values]