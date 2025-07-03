import socket, pyrah, struct, subprocess, multiprocessing, time, asyncio, serial_asyncio, json

DWP00, BUFFER_SIZE = bytes.fromhex("ff ff ff ff 00 00 00 00 00 00 00 00"), 3 * 1024 * 1024
PORT_MAIN, PORT_BITSTREAM, CONTROL_PORT = 8080, 8081, 9090
UART_PORT, UART_BAUD, UART_TIMEOUT = "/dev/ttyUSB0", 230400, 10
MAX_UART_BUFFER = 10 * 1024
PORT_UART_SERVER = 5001

# Main Server -> write input to FPGA
def main_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('', PORT_MAIN))
    try:
        state, app_id, data = 0, 1, b""
        while True:
            if state == 0:
                s.listen(1); print(f"[MainServer] Listening on port {PORT_MAIN}...")
                c, addr = s.accept(); print(f"[MainServer] Client connected from {addr}"); state = 1
            if state == 1:
                app_id_data = c.recv(4)
                if not app_id_data: print("[MainServer] Client disconnected"); state = 0; continue
                app_id = struct.unpack('>I', app_id_data)[0]
                length_bytes = c.recv(4)
                if not length_bytes: print("[MainServer] Client disconnected"); state = 0; continue
                length = struct.unpack('>I', length_bytes)[0]
                print(f"[MainServer] Receiving {length} bytes")
                data = b""
                while len(data) < length:
                    chunk = c.recv(min(length - len(data), BUFFER_SIZE))
                    if not chunk: print("[MainServer] Connection lost"); state = 0; break
                    data += chunk
                if len(data) == length: pyrah.rah_write(app_id, data); print(f"[MainServer] rah_write done"); state = 1
    except Exception as e:
        print(f"[MainServer] Error: {e}")
    finally:
        s.close(); print("[MainServer] Shutdown")

# Bitstream Server
def bitstream_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('', PORT_BITSTREAM)); s.listen(1)
    print(f"[BitstreamServer] Listening on {PORT_BITSTREAM}...")
    try:
        while True:
            c, addr = s.accept(); print(f"[Bitstream] Client from {addr}")
            length = int.from_bytes(c.recv(4), 'big')
            print(f"[Bitstream] Receiving {length} bytes")
            data = b""
            while len(data) < length: data += c.recv(min(length - len(data), BUFFER_SIZE))
            with open("bitstream.hex", "wb") as f: f.write(data)
            subprocess.run(["sudo", "bitman", "-f", "bitstream.hex"])
            c.send((7).to_bytes(4, 'big')); c.send(b"Flashed"); c.close()
    except Exception as e:
        print(f"[BitstreamServer] Error: {e}")
    finally:
        s.close(); print("[BitstreamServer] Shutdown")

# UART Protocol
class UARTProtocol(asyncio.Protocol):
    def __init__(self, shared_buffer, lock): self.shared_buffer, self.lock = shared_buffer, lock
    def connection_made(self, transport): print("[UART] Connection opened")
    def data_received(self, data): asyncio.create_task(self._handle_data(data))
    async def _handle_data(self, data):
        async with self.lock:
            self.shared_buffer.extend(data)
            if len(self.shared_buffer) > MAX_UART_BUFFER: self.shared_buffer[:] = self.shared_buffer[-MAX_UART_BUFFER:]
    def connection_lost(self, exc): print("[UART] Connection lost")

# UART Server
def uart_server_worker():
    async def run_uart_server():
        shared_buffer = bytearray(); lock = asyncio.Lock()
        loop = asyncio.get_running_loop()
        await serial_asyncio.create_serial_connection(loop, lambda: UARTProtocol(shared_buffer, lock), UART_PORT, baudrate=UART_BAUD)
        print(f"[UART] Listening on {UART_PORT} @ {UART_BAUD}")

        async def handle_client(reader, writer):
            addr = writer.get_extra_info('peername'); print(f"[UART-Client] Connected: {addr}")
            try:
                raw = await asyncio.wait_for(reader.readline(), timeout=5)
                req = json.loads(raw.decode().strip()); size = int(req.get("size", 0))
                print(f"[UART-Client] Requested {size} bytes")
                uart_data = b""; timeout = time.time() + UART_TIMEOUT
                while len(uart_data) < size:
                    await asyncio.sleep(0.001)
                    async with lock:
                        if len(shared_buffer) >= size:
                            uart_data = bytes(shared_buffer[:size]); del shared_buffer[:size]; break
                    if time.time() > timeout: raise asyncio.TimeoutError()
                print(f"[UART-Client] Sending {len(uart_data)} bytes")
                writer.write(uart_data); await writer.drain()
            except asyncio.TimeoutError:
                print("[UART-Client] Timeout waiting for UART data")
                writer.write(b"ERROR: Timeout\n"); await writer.drain()
            except Exception as e:
                print(f"[UART-Client] Error: {e}")
                writer.write(b"ERROR: Bad request\n"); await writer.drain()
            finally:
                writer.close(); await writer.wait_closed(); print(f"[UART-Client] Disconnected: {addr}")

        server = await asyncio.start_server(handle_client, host='0.0.0.0', port=PORT_UART_SERVER)
        print(f"[UARTServer] Listening on 0.0.0.0:{PORT_UART_SERVER}")
        async with server: await server.serve_forever()

    asyncio.run(run_uart_server())

# Parent Server
def parent_server():
    def start_servers():
        p1 = multiprocessing.Process(target=main_server, name="MainServer")
        p2 = multiprocessing.Process(target=bitstream_server, name="BitstreamServer")
        p3 = multiprocessing.Process(target=uart_server_worker, name="UARTServer")
        p1.start(); p2.start(); p3.start(); return p1, p2, p3

    p1, p2, p3 = start_servers()
    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); ctrl_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); ctrl_sock.bind(('', CONTROL_PORT)); ctrl_sock.listen(1)
    print(f"[ParentServer] Listening on {CONTROL_PORT}...")
    try:
        while True:
            c, _ = ctrl_sock.accept(); cmd = c.recv(16).strip()
            if cmd == b"reset":
                print("[ParentServer] Reset signal received")
                p1.terminate(); p2.terminate(); p3.terminate(); p1.join(); p2.join(); p3.join()
                p1, p2, p3 = start_servers(); c.send(b"OK")
            c.close()
    except Exception as e:
        print(f"[ParentServer] Error: {e}")
    finally:
        p1.terminate(); p2.terminate(); p3.terminate(); ctrl_sock.close()

if __name__ == "__main__":
    parent_server()
