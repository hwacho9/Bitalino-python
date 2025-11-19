import socket
import threading
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from bitalino import BITalino

# -------------------------------------------------------------------
# BITalino / 측정 설정 (네 기존 코드 기반)
# -------------------------------------------------------------------
macAddress   = "98:D3:71:FE:50:80"
samplingRate = 1000
nSamples     = 100
acqChannels  = [0, 1, 2, 3, 4, 5]   # A1~A6

sensor_labels = ["A1 (EMG)", "A2 (Sensor 2)", "A3", "A4", "A5", "A6"]

view_window  = 1000   # 그래프에 보여줄 샘플 개수

# 데이터 버퍼 (그래프용)
data_buffers = [deque([0] * view_window, maxlen=view_window) for _ in acqChannels]

# -------------------------------------------------------------------
# 상태 변수들
# -------------------------------------------------------------------
device        = None
is_measuring  = False
acq_thread    = None
lock          = threading.Lock()

# -------------------------------------------------------------------
# BITalino 측정 루프 (Unity에서 START 오면 실행)
# -------------------------------------------------------------------
def acquisition_loop():
    global is_measuring, device
    print("[BITalino] acquisition loop 시작")

    while is_measuring:
        try:
            data = device.read(nSamples)   # (nSamples, 5+len(acqChannels))

            # data: [Seq, D0, D1, D2, D3, A0, A1, A2, A3, A4, A5]
            with lock:
                for i in range(len(acqChannels)):
                    channel_data = data[:, 5 + i]
                    data_buffers[i].extend(channel_data)

        except Exception as e:
            print(f"[BITalino] Read error: {e}")
            break

    print("[BITalino] acquisition loop 종료")


def start_bitalino():
    global device, is_measuring, acq_thread

    if is_measuring:
        print("[BITalino] 이미 측정 중입니다.")
        return

    try:
        print(f"[BITalino] {macAddress} 에 연결 시도...")
        device = BITalino(macAddress)
        device.start(samplingRate, acqChannels)
        print("[BITalino] 연결 및 측정 시작 성공")

        is_measuring = True
        acq_thread = threading.Thread(target=acquisition_loop, daemon=True)
        acq_thread.start()

    except Exception as e:
        print(f"[BITalino] start 실패: {e}")
        is_measuring = False
        device = None


def stop_bitalino():
    global device, is_measuring, acq_thread

    if not is_measuring:
        print("[BITalino] 측정 중이 아닙니다.")
        return

    print("[BITalino] 측정 종료 중...")
    is_measuring = False

    if acq_thread is not None and acq_thread.is_alive():
        acq_thread.join(timeout=1.0)

    try:
        if device is not None:
            device.stop()
            device.close()
            print("[BITalino] stop & close 완료")
    except Exception as e:
        print(f"[BITalino] stop 중 에러: {e}")

    device = None
    acq_thread = None

# -------------------------------------------------------------------
# Unity와 통신할 TCP 서버
# -------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 5000

def handle_client(conn, addr):
    print(f"[연결] {addr}")
    with conn:
        while True:
            data = conn.recv(1024)
            if not data:
                break

            cmd = data.decode("utf-8").strip().upper()
            print(f"[수신] {addr}: {cmd}")

            if cmd == "START":
                start_bitalino()
                conn.sendall(b"OK START\n")

            elif cmd == "STOP":
                stop_bitalino()
                conn.sendall(b"OK STOP\n")

            elif cmd == "PING":
                conn.sendall(b"PONG\n")

            else:
                conn.sendall(b"UNKNOWN CMD\n")

    print(f"[끊김] {addr}")


def server_loop():
    print(f"[서버] {HOST}:{PORT} 에서 대기 중...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()

        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

# -------------------------------------------------------------------
# matplotlib 실시간 플롯 설정
# -------------------------------------------------------------------
def start_plot():
    fig, ax = plt.subplots()
    lines = []
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    for i, label in enumerate(sensor_labels):
        line, = ax.plot([], [], label=label, color=colors[i % len(colors)], linewidth=1)
        lines.append(line)

    ax.set_ylim(0, 1024)  # 10-bit (0~1023)
    ax.set_xlim(0, view_window)
    ax.grid(True)
    ax.legend(loc='upper right')
    ax.set_title("Real-time BITalino Sensor Data (from Unity START/STOP)")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Raw Value (0-1023)")

    def update(frame):
        # data_buffers 내용을 이용해서 그래프 업데이트
        with lock:
            for i in range(len(acqChannels)):
                buf = list(data_buffers[i])
                lines[i].set_data(range(len(buf)), buf)
        return lines

    ani = animation.FuncAnimation(fig, update, interval=50, blit=True)
    plt.show()

# -------------------------------------------------------------------
# 메인 엔트리 포인트
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # 1) 서버는 백그라운드 쓰레드로 실행 (Unity에서 START/STOP 받기)
        t = threading.Thread(target=server_loop, daemon=True)
        t.start()

        # 2) 메인 스레드에서는 matplotlib 플롯
        start_plot()

    finally:
        # 창 닫히면 혹시 측정 중이면 정리
        if is_measuring:
            stop_bitalino()
