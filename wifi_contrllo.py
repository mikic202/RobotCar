import sys
import socket
import threading
import time
import numpy as np
from gpiozero import PWMOutputDevice, DigitalOutputDevice
from HardwareClasses.Motor import Motor
from HardwareClasses.MotorDrive import MotorDrive

# first = DigitalOutputDevice(15)
# second = DigitalOutputDevice(18)


# led = PWMOutputDevice(14)

# motor = Motor(15, 18, 14)
# motor2 = Motor(24, 25, 23)


motor_controller = MotorDrive([Motor(14, 15), Motor(23, 24)])


def handle_client(client_socket: socket, server: socket):
    reader = client_socket.makefile("r")
    reader.readline()
    # sleep(1)
    try:
        while True:
            request = reader.readline()  # convert bytes to string
            print(request)
            values = str(request).strip().split(";")
            print(values)
            # if we receive "close" from the client, then we break
            # out of the loop and close the conneciton
            # if request.lower() == "close":
            #     # send response to the client which acknowledges that the
            #     # connection should be closed and break out of the loop
            #     client_socket.send("closed".encode("utf-8"))
            #     break
            motor_controller.set_pwms([float(val) / 100 for val in values])
            # convert and send accept response to the client
    finally:
        reader.close()
        client_socket.close()
        server.close()

    # close connection socket with the client

    print("Connection to client closed")
    # close server socket


def run_server():
    # create a socket object
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_ip = "192.168.0.167"
    port = 8080

    # bind the socket to a specific address and port
    server.bind((server_ip, port))
    # listen for incoming connections
    server.listen(2)
    print(f"Listening on {server_ip}:{port}")

    # accept incoming connections
    client_socket, client_address = server.accept()
    t1 = threading.Thread(target=handle_client, args=[client_socket, server])
    print(f"Accepted connection from {client_address[0]}:{client_address[1]}")
    t1.daemon = True
    try:
        t1.start()
        while True:
            print(f"processing client {threading.active_count()}")
            time.sleep(1)
    finally:
        server.shutdown(0)
        server.close()
        sys.exit()

    # receive data from the client


run_server()
