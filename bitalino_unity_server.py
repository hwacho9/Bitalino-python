import socket
import threading
from collections import deque
import time
import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from bitalino import BITalino

# -------------------------------------------------------------------
# BITalino / 측정 설정
# -------------------------------------------------------------------
macAddress   = "98:D3:71:FE:50:80"
samplingRate = 1000
nSamples     = 100
acqChannels  = [0, 1, 2, 3, 4, 5]   # A1~A6

sensor_labels = ["A1 (EMG)", "A2 (EDA)", "A3 (ECG)", "A4", "A5", "A6"]

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

# 샘플 카운터(지금까지 읽은 총 샘플 수)
sample_counter = 0

# 이벤트 마커 (그래프용): (sample_index, label, unix_time)
event_markers = deque(maxlen=200)

# 이벤트 로그 파일
event_log_file = "events_log.csv"

# 센서 데이터 로그 파일
data_log_file = None  # START 시점에 타임스탬프로 생성
data_file_handle = None  # 파일 핸들

# -------------------------------------------------------------------
# 이벤트 로그 기록
# -------------------------------------------------------------------
def log_event(label: str):
    """
    Unity에서 보낸 이벤트를 시간 + 샘플 인덱스와 함께 기록
    """
    global sample_counter
    t = time.time()
    with lock:
        s_index = sample_counter  # 현재까지 읽은 샘플 수 기준 (대략적인 위치)
        event_markers.append((s_index, label, t))

    # CSV로 기록 (타임스탬프, 샘플 인덱스, 라벨)
    line = f"{t:.3f},{s_index},{label}\n"
    try:
        new_file = not os.path.exists(event_log_file)
        with open(event_log_file, "a", encoding="utf-8") as f:
            if new_file:
                f.write("unix_time, sample_index, label\n")
            f.write(line)
        print("[EVENT]", line.strip())
    except Exception as e:
        print(f"[EVENT] 로그 기록 실패: {e}")


# -------------------------------------------------------------------
# 센서 데이터 로그 기록
# -------------------------------------------------------------------
def log_sensor_data(data, start_time):
    """
    BITalino에서 읽은 센서 데이터를 CSV 파일에 기록
    data: numpy array [nSamples, 5+len(acqChannels)]
    start_time: 측정 시작 시간 (타임스탬프 계산용)
    """
    global data_file_handle, sample_counter, samplingRate
    
    if data_file_handle is None:
        return
    
    try:
        # 각 샘플에 대해 기록
        for sample_idx, row in enumerate(data):
            # 현재 샘플 인덱스 (lock 안에서 가져오기)
            with lock:
                current_sample = sample_counter - len(data) + sample_idx
                timestamp = start_time + (current_sample / samplingRate)
            
            # Sequence, Digital channels, Analog channels
            seq = row[0]
            d0, d1, d2, d3 = row[1], row[2], row[3], row[4]
            analog_values = row[5:5+len(acqChannels)].tolist()
            
            # CSV 형식: timestamp, sample_index, seq, D0, D1, D2, D3, A0, A1, A2, A3, A4, A5
            line = f"{timestamp:.6f},{current_sample},{seq},{d0},{d1},{d2},{d3}"
            for a_val in analog_values:
                line += f",{a_val}"
            line += "\n"
            
            data_file_handle.write(line)
        
        # 버퍼를 즉시 파일에 기록 (데이터 손실 방지)
        data_file_handle.flush()
            
    except Exception as e:
        print(f"[DATA] 로그 기록 실패: {e}")

# -------------------------------------------------------------------
# BITalino 측정 루프 (Unity에서 START 오면 실행)
# -------------------------------------------------------------------
def acquisition_loop(start_time):
    global is_measuring, device, sample_counter
    print("[BITalino] acquisition loop 시작")
    read_count = 0

    while is_measuring:
        try:
            data = device.read(nSamples)   # (nSamples, 5+len(acqChannels))

            # data: [Seq, D0, D1, D2, D3, A0, A1, A2, A3, A4, A5]
            with lock:
                for i in range(len(acqChannels)):
                    channel_data = data[:, 5 + i]
                    data_buffers[i].extend(channel_data)

                # 전체 샘플 수 업데이트
                sample_counter += len(data)
            
            read_count += 1
            # 처음 몇 번만 출력해서 데이터가 들어오는지 확인
            if read_count <= 3:
                print(f"[BITalino] 데이터 읽기 {read_count}: {len(data)}개 샘플, 총 {sample_counter}개")

            # 센서 데이터를 파일에 저장 (lock 밖에서 실행 - 파일 I/O는 느릴 수 있음)
            log_sensor_data(data, start_time)

        except Exception as e:
            print(f"[BITalino] Read error: {e}")
            break

    print(f"[BITalino] acquisition loop 종료 (총 {read_count}번 읽기, {sample_counter}개 샘플)")


def start_bitalino():
    global device, is_measuring, acq_thread, sample_counter, data_log_file, data_file_handle

    if is_measuring:
        print("[BITalino] 이미 측정 중입니다.")
        return

    try:
        print(f"[BITalino] {macAddress} 에 연결 시도...")
        device = BITalino(macAddress)
        device.start(samplingRate, acqChannels)
        print("[BITalino] 연결 및 측정 시작 성공")

        # 새 측정 시작 시 샘플 카운터 리셋 및 데이터 버퍼 초기화
        with lock:
            sample_counter = 0
            event_markers.clear()
            # 데이터 버퍼 초기화 (그래프 표시를 위해)
            for i in range(len(acqChannels)):
                data_buffers[i].clear()
                data_buffers[i].extend([0] * view_window)

        # 센서 데이터 파일 초기화 (타임스탬프 포함 파일명)
        start_time = time.time()
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(start_time))
        data_log_file = f"sensor_data_{timestamp_str}.csv"
        
        # CSV 헤더 작성
        header = "unix_time, sample_index, sequence, D0, D1, D2, D3"
        for i, label in enumerate(sensor_labels):
            header += f", {label.replace(' ', '_')}"
        header += "\n"
        
        # buffering=1은 line buffering (줄 단위로 즉시 저장)
        data_file_handle = open(data_log_file, "w", encoding="utf-8", buffering=1)
        data_file_handle.write(header)
        print(f"[BITalino] 센서 데이터 저장 시작: {data_log_file}")

        is_measuring = True
        acq_thread = threading.Thread(target=acquisition_loop, args=(start_time,), daemon=True)
        acq_thread.start()

    except Exception as e:
        print(f"[BITalino] start 실패: {e}")
        is_measuring = False
        device = None
        if data_file_handle:
            data_file_handle.close()
            data_file_handle = None


def stop_bitalino():
    global device, is_measuring, acq_thread, data_file_handle, data_log_file

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

    # 센서 데이터 파일 닫기
    if data_file_handle is not None:
        data_file_handle.close()
        data_file_handle = None
        print(f"[BITalino] 센서 데이터 저장 완료: {data_log_file}")

    device = None
    acq_thread = None

# -------------------------------------------------------------------
# Unity와 통신할 TCP 서버
# -------------------------------------------------------------------
HOST = "0.0.0.0"
PORT = 5000

def handle_client(conn, addr):
    print(f"[연결] {addr}")
    client_started_measuring = False  # 이 클라이언트가 측정을 시작했는지 추적
    
    try:
        with conn:
            while True:
                data = conn.recv(1024)
                if not data:
                    break

                cmd_raw = data.decode("utf-8").strip()
                cmd = cmd_raw.upper()
                print(f"[수신] {addr}: {cmd_raw}")

                if cmd == "START":
                    start_bitalino()
                    client_started_measuring = True
                    log_event("START")  # START event log
                    conn.sendall(b"OK START\n")

                elif cmd == "STOP":
                    log_event("STOP")  # STOP event log
                    stop_bitalino()
                    client_started_measuring = False
                    conn.sendall(b"OK STOP\n")

                elif cmd == "PING":
                    conn.sendall(b"PONG\n")

                else:
                    # All other commands are logged as events (custom events from Unity)
                    log_event(cmd_raw)   # log as original case
                    conn.sendall(b"OK EVENT\n")
    finally:
        # 연결이 끊어졌을 때, 이 클라이언트가 측정을 시작했다면 자동으로 중단
        if client_started_measuring and is_measuring:
            print(f"[끊김] {addr} - Unity 연결 끊김, 측정 자동 중단")
            stop_bitalino()
        else:
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
# matplotlib 실시간 플롯 설정 (이벤트 라인 표시 포함)
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
    ax.set_title("Real-time BITalino Sensor Data (Unity START/STOP + Events)")
    ax.set_xlabel("Samples (last window)")
    ax.set_ylabel("Raw Value (0-1023)")

    # 이벤트 라인을 따로 관리하는 리스트
    event_artists = []

    def update(frame):
        nonlocal event_artists

        # 센서 데이터 + 이벤트 정보 스냅샷
        with lock:
            buffers_copy = [list(buf) for buf in data_buffers]
            current_samples = sample_counter
            markers_copy = list(event_markers)
            measuring = is_measuring

        # 상태에 따른 제목 업데이트
        status = "RECORDING" if measuring else "WAITING"
        ax.set_title(f"Real-time BITalino Sensor Data - {status} (Samples: {current_samples})")

        # 센서 라인 업데이트
        for i in range(len(acqChannels)):
            buf = buffers_copy[i]
            x = range(len(buf))
            lines[i].set_data(x, buf)

        # 이전 프레임에서 그렸던 이벤트 라인 제거
        for art in event_artists:
            art.remove()
        event_artists.clear()

        # 현재 윈도우(view_window)에 들어오는 이벤트만 세로선으로 표시
        ymin, ymax = ax.get_ylim()
        for s_index, label, t_event in markers_copy:
            # 이벤트가 발생한 위치가 현재 윈도우에서 몇 번째 샘플인지 계산
            offset = current_samples - s_index     # 이벤트가 몇 샘플 전에 발생했는지
            x_pos = view_window - offset           # 오른쪽 끝이 최신 샘플

            if 0 <= x_pos <= view_window:
                ln = ax.axvline(x=x_pos, color='k', linestyle='--', alpha=0.4)
                event_artists.append(ln)

        # 그래프 재그리기
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=False)  # y축은 0-1024로 고정, x축은 데이터에 맞춤
        
        # blit=False 이므로 반환값은 크게 중요하지 않지만 형식상 리턴
        return lines + event_artists

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False)
    plt.show()

# -------------------------------------------------------------------
# 메인 엔트리 포인트
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # 1) 서버는 백그라운드 쓰레드로 실행 (Unity에서 START/STOP/이벤트 받기)
        t = threading.Thread(target=server_loop, daemon=True)
        t.start()

        # 2) 메인 스레드에서는 matplotlib 플롯
        start_plot()

    finally:
        # 창 닫히면 혹시 측정 중이면 정리
        if is_measuring:
            stop_bitalino()
        # 파일 핸들이 열려있으면 닫기
        if data_file_handle is not None:
            data_file_handle.close()
            data_file_handle = None