import socket, time
msg = b'{"source":"stm32","mode":"word","model":"CNN","text":"hi","infer_us":0}\n'
with socket.create_connection(("127.0.0.1", 8000), timeout=5) as s:
    s.settimeout(8)
    s.sendall(msg)
    data = b""
    while not data.endswith(b"\n"):
        chunk = s.recv(4096)
        if not chunk:
            break
        data += chunk
    print(data.decode("utf-8", "replace").strip())
time.sleep(2)
