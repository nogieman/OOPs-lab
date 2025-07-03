import socket, pyrah, struct, os, subprocess, multiprocessing, time

# Shared Constants
DWP00, BUFFER_SIZE = bytes.fromhex("ff ff ff ff 00 00 00 00 00 00 00 00"), 3 * 1024 * 1024
PORT_MAIN, PORT_BITSTREAM, CONTROL_PORT = 8080, 8081, 9090

# States for Main Server
CONNECTING, READ_CLIENT, WRITE_FPGA, READ_FPGA, WRITE_CLIENT = range(5)

def print_hex_bytes(data, label=""):
    print(f"{label}{' ' if label else ''}{' '.join(f'{b:02x}' for b in data[:min(50, len(data))])} (len={len(data)})")

def main_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('', PORT_MAIN))
    try:
        state = CONNECTING
        app_id = 1
        read_client_data = b""
        while True:
            if state == CONNECTING:
                print(f"State: CONNECTING {state}")
                server_socket.listen(1)
                print(f"Server listening on port {PORT_MAIN}...")
                print("Server IP: Use your local IP address (e.g., 192.168.x.x)")
                client_socket, client_address = server_socket.accept()
                print(f"Client connected from {client_address}")
                state = READ_CLIENT
            if state == READ_CLIENT:
                print(f"State: READ_CLIENT {state}")
                app_id_data = client_socket.recv(4)
                if not app_id_data:
                    print("Client disconnected")
                    state = CONNECTING
                    continue
                app_id = struct.unpack('>I', app_id_data)[0]
                length_bytes = client_socket.recv(4)
                if not length_bytes:
                    print("Client disconnected")
                    state = CONNECTING
                    continue
                length = struct.unpack('>I', length_bytes)[0]
                #print(f"Writing {length} bytes to {app_id}")
                data = b""
                while len(data) < length:
                    chunk = client_socket.recv(min(length - len(data), 3 * 1024 * 1024))
                    if not chunk:
                        print("Connection lost while receiving message")
                        state = CONNECTING
                        break
                    data += chunk
                read_client_data = data
                #print(f"Length bytes {length}")
                #print_hex_bytes(read_client_data)
                if len(read_client_data) == length:
                    state = WRITE_FPGA
            if state == WRITE_FPGA:
                #print(f"State: WRITE_FPGA {state}")
                pyrah.rah_write(app_id, read_client_data)
                if read_client_data[-12:] == DWP00:
                    state = READ_FPGA
                else:
                    state = READ_CLIENT
            if state == READ_FPGA:
                print(f"State: READ_FPGA {state}")
                length_bytes = client_socket.recv(4)
                if not length_bytes:
                    print("Client disconnected")
                    state = CONNECTING
                    continue
                length = struct.unpack('>I', length_bytes)[0]
                #print(f"Read FPGA length {length}")
                #length = 56
                #print(length)
                pyrah.rah_clear_buffer(1)
                rah_read_data = pyrah.rah_read(1, length)
                if len(rah_read_data) == length:
                    state = WRITE_CLIENT
                    continue
                else:
                    state = CONNECTING
                    continue
            if state == WRITE_CLIENT:
                print(f"State: WRITE_CLIENT {state}")
                #print_hex_bytes(rah_read_data)
                client_socket.send(struct.pack('>I', len(rah_read_data)))
                client_socket.send(rah_read_data)
                state = READ_CLIENT
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client_socket.close()
        server_socket.close()
        print("Server shut down")

def bitstream_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('', PORT_BITSTREAM)); s.listen(1); print(f"Bitstream server on {PORT_BITSTREAM}...")
    try:
        while True:
            c, addr = s.accept(); print(f"Bitstream - Client from {addr}")
            length = struct.unpack('>I', c.recv(4) or (print("Bitstream - Client disconnected"), c.close())[1])[0]
            print(f"Bitstream - Receiving {length} bytes")
            data = b""
            while len(data) < length:
                data += c.recv(min(length - len(data), BUFFER_SIZE))
            #print_hex_bytes(data, "bitstream")
            with open("bitstream.hex", "wb") as f: f.write(data)
            subprocess.run(["sudo", "bitman", "-f", "bitstream.hex"])
            c.send(struct.pack('>I', 7)); c.send(b"Flashed"); c.close()
    except Exception as e:
        print(f"Bitstream - Error: {e}")
    finally:
        s.close(); print("Bitstream server shut down")

def parent_server():
    def start_servers():
        p1 = multiprocessing.Process(target=main_server, name="MainServer")
        p2 = multiprocessing.Process(target=bitstream_server, name="BitstreamServer")
        p1.start(); p2.start()
        return p1, p2

    p1, p2 = start_servers()
    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ctrl_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ctrl_sock.bind(('', CONTROL_PORT)); ctrl_sock.listen(1)
    print(f"Parent server listening on {CONTROL_PORT}...")

    try:
        while True:
            c, _ = ctrl_sock.accept()
            cmd = c.recv(16).strip()
            if cmd == b"reset":
                print("Reset signal received")
                p1.terminate(); p2.terminate()
                p1.join(); p2.join()
                p1, p2 = start_servers()
                c.send(b"OK")
            c.close()
    except Exception as e:
        print(f"Parent - Error: {e}")
    finally:
        p1.terminate(); p2.terminate()
        ctrl_sock.close()

if __name__ == "__main__":
    parent_server()
