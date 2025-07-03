#!/usr/bin/env python

import numpy as np
from PIL import Image
import serial
import time

def save_tensor(filename, arr):
    fname = filename.replace('/', '_')
    np.save(fname, arr)

def setup_uart(port, baudrate=115200):
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return None

def read_uart(baudrate, expected_bytes):
    serial_port = '/dev/ttyUSB0'
    ser = setup_uart(serial_port, baudrate=baudrate)
    print(f"Baud rate {baudrate}, Expected bytes {expected_bytes}")
    if not ser:
        return -1
    print(f"Listening on {serial_port}...")
    try:
        while True:
            try:
                data = ser.read(expected_bytes)
                buf = np.frombuffer(data, dtype=np.int8)
                return buf
            except Exception as e:
                print(f"Error reading data: {e}")
                return None
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ser.close()
        print("Serial port closed")

def compare_npy(received_tensor, residing_tensor_path):
    t2 = np.load(residing_tensor_path)
    t1 = received_tensor.flatten()
    t2 = t2.flatten()
    #assert(len(t1) == len(t2))
    for i,j in zip(t1, t2):
        print(i,j)

def identity(arr):
    return arr
