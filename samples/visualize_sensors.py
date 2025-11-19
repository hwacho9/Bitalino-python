import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from bitalino import BITalino

# -----------------------------------------------------------------------------
# 사용자 설정 (User Configuration)
# -----------------------------------------------------------------------------
macAddress = "98:D3:71:FE:50:80"  # 기존 sample.py에 있던 주소
samplingRate = 1000
nSamples = 100  # 한 번에 읽을 샘플 수

# BITalino 보드의 채널 매핑 (Board Channel Mapping)
# 코드의 0, 1, 2... 는 보드의 A1, A2, A3... 에 대응됩니다.
# 실제 연결된 센서 이름을 아래 리스트에 순서대로 적어주세요.
# 예: ["EMG", "EDA", "ECG", "EEG", "ACC", "LUX"]
sensor_labels = ["A1 (EMG)", "A2 (Sensor 2)", "A3", "A4", "A5", "A6"]
acqChannels = [0, 1, 2, 3, 4, 5]  # 모든 아날로그 채널 읽기

# 시각화 설정
view_window = 1000  # 그래프에 보여줄 데이터 포인트 수 (예: 1000개)
# -----------------------------------------------------------------------------

# 데이터 저장을 위한 큐 생성 (오래된 데이터는 자동으로 삭제됨)
data_buffers = [deque([0] * view_window, maxlen=view_window) for _ in acqChannels]

# BITalino 연결
print(f"Connecting to {macAddress}...")
device = BITalino(macAddress)
device.start(samplingRate, acqChannels)
print("Connection successful. Reading data...")

# 그래프 초기화
fig, ax = plt.subplots()
lines = []
colors = ['r', 'g', 'b', 'c', 'm', 'y']

# 각 채널별 라인 생성
for i, label in enumerate(sensor_labels):
    line, = ax.plot([], [], label=label, color=colors[i % len(colors)], linewidth=1)
    lines.append(line)

ax.set_ylim(0, 1024)  # BITalino 데이터 범위 (10-bit: 0-1023)
ax.set_xlim(0, view_window)
ax.grid(True)
ax.legend(loc='upper right')
ax.set_title("Real-time BITalino Sensor Data")
ax.set_xlabel("Samples")
ax.set_ylabel("Raw Value (0-1023)")

def update(frame):
    try:
        # BITalino에서 데이터 읽기
        data = device.read(nSamples)
        
        # 데이터 파싱 및 버퍼 업데이트
        # data 구조: [Seq, D0, D1, D2, D3, A0, A1, A2, A3, A4, A5]
        # 아날로그 데이터는 인덱스 5부터 시작
        for i in range(len(acqChannels)):
            channel_data = data[:, 5 + i] # 5번째 컬럼부터 아날로그 데이터
            data_buffers[i].extend(channel_data)
            
            # 그래프 데이터 업데이트
            lines[i].set_data(range(view_window), data_buffers[i])
            
    except Exception as e:
        print(f"Error reading data: {e}")
    
    return lines

# 애니메이션 실행
ani = animation.FuncAnimation(fig, update, interval=50, blit=True)

try:
    plt.show()
finally:
    # 창이 닫히면 연결 종료
    print("Stopping acquisition...")
    device.stop()
    device.close()
    print("Connection closed.")

