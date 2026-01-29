import csv
from datetime import datetime
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
# macAddress   = "COM3"
macAddress   = "98:D3:71:FE:50:80"
samplingRate = 1000
nSamples     = 100
acqChannels  = [0, 1, 2]   # A1~A3

sensor_labels = ["A1 (EMG)", "A2 (EDA)", "A3 (ECG)"]

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
# 이벤트 로그 파일 (실행 시점 기준 타임스탬프)
start_ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
event_log_file = f"events_log_{start_ts_str}.csv"

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

    # CSV로 기록 (Unix Time, Readable Time, Sample Index, Label)
    dt_str = datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    line = f"{t:.6f},{dt_str},{s_index},{label}\n"
    try:
        new_file = not os.path.exists(event_log_file)
        with open(event_log_file, "a", encoding="utf-8") as f:
            if new_file:
                f.write("unix_time, readable_time, sample_index, label\n")
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
                dt_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # Sequence, Digital channels, Analog channels
            seq = row[0]
            d0, d1, d2, d3 = row[1], row[2], row[3], row[4]
            analog_values = row[5:5+len(acqChannels)].tolist()
            
            # CSV 형식: unix_time, readable_time, sample_index, seq, D0, D1, D2, D3, A0, A1, A2, A3, A4, A5
            line = f"{timestamp:.6f},{dt_str},{current_sample},{seq},{d0},{d1},{d2},{d3}"
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
    error_count = 0
    max_errors = 3  # 연속 에러 최대 횟수

    while is_measuring:
        try:
            data = device.read(nSamples)   # (nSamples, 5+len(acqChannels))

            # 에러 카운트 리셋 (성공했으므로)
            error_count = 0

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
            
            # 500번 마다 진행 상황 출력 (약 5초마다)
            if read_count % 500 == 0:
                print(f"[BITalino] 진행 중: {read_count}번 읽기, {sample_counter}개 샘플")

            # 센서 데이터를 파일에 저장 (lock 밖에서 실행 - 파일 I/O는 느릴 수 있음)
            log_sensor_data(data, start_time)

        except Exception as e:
            error_count += 1
            print(f"[BITalino] Read error ({error_count}/{max_errors}): {e}")
            
            if error_count >= max_errors:
                print("[BITalino] 연속 에러 한계 도달, 재연결 시도...")
                # 재연결 시도
                try:
                    device.stop()
                except:
                    pass
                try:
                    device.close()
                except:
                    pass
                
                time.sleep(1)  # 잠시 대기
                
                if connect_bitalino():
                    try:
                        device.start(samplingRate, acqChannels)
                        print("[BITalino] 재연결 성공! 측정 재개")
                        error_count = 0
                        continue
                    except Exception as e2:
                        print(f"[BITalino] 재연결 후 측정 시작 실패: {e2}")
                        break
                else:
                    print("[BITalino] 재연결 실패, 측정 중단")
                    break
            else:
                time.sleep(0.1)  # 짧은 대기 후 재시도
                continue

    print(f"[BITalino] acquisition loop 종료 (총 {read_count}번 읽기, {sample_counter}개 샘플)")


# -------------------------------------------------------------------
# BITalino 연결 (서버 시작 시 즉시 호출)
# -------------------------------------------------------------------
def connect_bitalino():
    """서버 시작 시 BITalino에 연결 (visualize_sensors.py처럼)"""
    global device
    
    import sys
    print(f"[BITalino] {macAddress} 에 연결 시도...", file=sys.stderr, flush=True)
    try:
        device = BITalino(macAddress, timeout=5)
        print("[BITalino] 연결 성공!", file=sys.stderr, flush=True)
        return True
    except Exception as e:
        print(f"[BITalino] 연결 실패: {e}", file=sys.stderr, flush=True)
        device = None
        return False


# -------------------------------------------------------------------
# 측정 시작 (Unity에서 START 오면 실행)
# -------------------------------------------------------------------
def start_acquisition():
    """데이터 수집 시작 (이미 연결된 상태에서)"""
    global device, is_measuring, acq_thread, sample_counter, data_log_file, data_file_handle

    import sys
    
    if device is None:
        print("[BITalino] 장치가 연결되어 있지 않습니다!", file=sys.stderr, flush=True)
        return False

    if is_measuring:
        print("[BITalino] 이미 측정 중입니다.", file=sys.stderr, flush=True)
        return True

    try:
        # 측정 시작
        device.start(samplingRate, acqChannels)
        print("[BITalino] 측정 시작!", file=sys.stderr, flush=True)

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
        header = "unix_time, readable_time, sample_index, sequence, D0, D1, D2, D3"
        for i, label in enumerate(sensor_labels):
            header += f", {label.replace(' ', '_')}"
        header += "\n"
        
        # buffering=1은 line buffering (줄 단위로 즉시 저장)
        data_file_handle = open(data_log_file, "w", encoding="utf-8", buffering=1)
        data_file_handle.write(header)
        
        abs_path = os.path.abspath(data_log_file)
        print(f"[BITalino] 센서 데이터 저장 시작: {abs_path}", flush=True)

        is_measuring = True
        acq_thread = threading.Thread(target=acquisition_loop, args=(start_time,), daemon=True)
        acq_thread.start()
        return True

    except Exception as e:
        print(f"[BITalino] 측정 시작 실패: {e}", flush=True)
        is_measuring = False
        if data_file_handle:
            data_file_handle.close()
            data_file_handle = None
        return False


# -------------------------------------------------------------------
# 측정 중지 (Unity에서 STOP 오면 실행)
# -------------------------------------------------------------------
def stop_acquisition():
    """데이터 수집 중지 (연결은 유지)"""
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
            device.stop()  # 측정만 중지, 연결은 유지
            print("[BITalino] 측정 중지 완료 (연결 유지)")
    except Exception as e:
        print(f"[BITalino] stop 중 에러: {e}")

    # 센서 데이터 파일 닫기
    if data_file_handle is not None:
        data_file_handle.close()
        data_file_handle = None
        print(f"[BITalino] 센서 데이터 저장 완료: {data_log_file}")

    acq_thread = None


# -------------------------------------------------------------------
# BITalino 연결 해제 (프로그램 종료 시)
# -------------------------------------------------------------------
def disconnect_bitalino():
    """프로그램 종료 시 BITalino 연결 해제"""
    global device
    
    if device is not None:
        try:
            if is_measuring:
                device.stop()
            device.close()
            print("[BITalino] 연결 해제 완료")
        except Exception as e:
            print(f"[BITalino] 연결 해제 중 에러: {e}")
        device = None

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
                    success = start_acquisition()
                    if success:
                        client_started_measuring = True
                        log_event("START")
                        conn.sendall(b"OK START\n")
                    else:
                        conn.sendall(b"ERR START FAILED\n")

                elif cmd == "STOP":
                    log_event("STOP")
                    stop_acquisition()
                    client_started_measuring = False
                    conn.sendall(b"OK STOP\n")

                elif cmd == "PING":
                    conn.sendall(b"PONG\n")

                else:
                    # All other commands are logged as events (custom events from Unity)
                    log_event(cmd_raw)
                    conn.sendall(b"OK EVENT\n")
    finally:
        # 연결이 끊어졌을 때, 이 클라이언트가 측정을 시작했다면 자동으로 중단
        if client_started_measuring and is_measuring:
            print(f"[끊김] {addr} - Unity 연결 끊김, 측정 자동 중단")
            stop_acquisition()
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
    # A1, A2, A3 (3개 채널)만 표시
    visible_channels_count = 3
    fig, axes = plt.subplots(visible_channels_count, 1, sharex=True, figsize=(8, 8))
    # fig.tight_layout() # 레이아웃 조정
    
    lines = []
    colors = ['r', 'g', 'b'] # 3개 색상만 필요

    # 각 서브플롯 초기화 (처음 3개 채널만)
    for i in range(visible_channels_count):
        ax = axes[i]
        label = sensor_labels[i]
        line, = ax.plot([], [], label=label, color=colors[i % len(colors)], linewidth=1)
        lines.append(line)
        
        ax.set_ylim(0, 1024)
        ax.set_xlim(0, view_window)
        ax.grid(True)
        ax.legend(loc='upper right')
        ax.set_ylabel("Raw")

    axes[0].set_title("Real-time BITalino Sensor Data (Unity START/STOP + Events)")
    axes[-1].set_xlabel("Samples (last window)")

    # 이벤트 라인을 따로 관리하는 리스트 (각 서브플롯별로 관리)
    event_artists = []

    def update(frame):
        nonlocal event_artists

        # 센서 데이터 + 이벤트 정보 스냅샷
        with lock:
            buffers_copy = [list(buf) for buf in data_buffers]
            current_samples = sample_counter
            markers_copy = list(event_markers)
            measuring = is_measuring

        # 상태에 따른 제목 업데이트 (첫 번째 서브플롯에만)
        status = "RECORDING" if measuring else "WAITING"
        axes[0].set_title(f"Real-time BITalino Sensor Data - {status} (Samples: {current_samples})")

        # 센서 라인 업데이트 (화면에 보이는 3개 채널만 업데이트)
        for i in range(visible_channels_count):
            buf = buffers_copy[i]
            x = range(len(buf))
            lines[i].set_data(x, buf)

        # 이전 프레임에서 그렸던 이벤트 라인 제거
        for art in event_artists:
            art.remove()
        event_artists.clear()

        # 현재 윈도우(view_window)에 들어오는 이벤트만 세로선으로 표시
        for s_index, label, t_event in markers_copy:
            # 이벤트가 발생한 위치가 현재 윈도우에서 몇 번째 샘플인지 계산
            offset = current_samples - s_index     # 이벤트가 몇 샘플 전에 발생했는지
            x_pos = view_window - offset           # 오른쪽 끝이 최신 샘플

            if 0 <= x_pos <= view_window:
                # 모든 서브플롯에 세로선 그리기
                for ax in axes:
                    ln = ax.axvline(x=x_pos, color='k', linestyle='--', alpha=0.4)
                    event_artists.append(ln)

        return lines + event_artists

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()

# -------------------------------------------------------------------
# 메인 엔트리 포인트
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # 1) BITalino에 먼저 연결 (visualize_sensors.py처럼 즉시 연결)
        print("=" * 50)
        print("BITalino Unity Server 시작")
        print("=" * 50)
        
        if not connect_bitalino():
            print("[오류] BITalino 연결 실패! 프로그램을 종료합니다.")
            print("       장치 전원을 확인하고 다시 시도해주세요.")
            exit(1)
        
        print("[준비 완료] Unity에서 START 명령을 기다리는 중...")
        print("=" * 50)
        
        # 2) 서버는 백그라운드 쓰레드로 실행 (Unity에서 START/STOP/이벤트 받기)
        t = threading.Thread(target=server_loop, daemon=True)
        t.start()

        # 3) 메인 스레드에서는 matplotlib 플롯
        start_plot()

    finally:
        # 창 닫히면 정리
        print("\n[종료] 정리 중...")
        
        # 측정 중이면 중지
        if is_measuring:
            stop_acquisition()
        
        # BITalino 연결 해제
        disconnect_bitalino()
        
        # 파일 핸들이 열려있으면 닫기
        if data_file_handle is not None:
            data_file_handle.close()
        
        print("[종료] 완료")