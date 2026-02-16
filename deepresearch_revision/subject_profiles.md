# 피험자별 주요 반응 프로파일 (個人差分析용)

## 피험자 매핑

| 폴더명    | 피험자    | 코드 |
| --------- | --------- | ---- |
| masaya    | Noguchi   | A    |
| hase1     | Hase      | B1   |
| hase2     | Hase      | B2   |
| matsumoto | Matsumoto | C    |
| ishikawa  | Ishikawa  | D    |
| takamiya  | Takamiya  | E1   |
| takamiya2 | Takamiya  | E2   |
| sensei    | Sensei    | F    |

---

## 전체 피험자 CHASE Phase 요약

| Code | Subject   | Duration | HR (bpm) | RMSSD (ms) | EMG RMS (μV) | 특이사항            |
| ---- | --------- | -------- | -------- | ---------- | ------------ | ------------------- |
| A    | Noguchi   | 0.0s     | -        | -          | -            | CHASE 없음          |
| B1   | Hase      | 36.0s    | 71.21    | 228.75     | 32.11        | ✓ Verified          |
| B2   | Hase      | 21.1s    | 73.37    | 46.70      | 26.94        | ✓ Verified          |
| C    | Matsumoto | 49.2s    | 86.10    | 16.47      | 23.04        | ✓ Verified          |
| D    | Ishikawa  | 6.3s     | 78.65    | 28.35      | 12.65        | ✓ Verified          |
| E1   | Takamiya  | 16.0s    | 76.45    | 99.90      | 20.14        | ✓ Verified          |
| E2   | Takamiya  | 25.8s    | 28.96    | 4766.04    | 102.00       | **특이반응 (徐脈)** |
| F    | Sensei    | 9.8s     | 90.39    | 10.36      | 35.17        | 높은 각성           |

---

## 설문 응답 요약 (実験後アンケート)

| Code | Subject   | Q3-1 추적공포 | Q3-2 근접공포 | Q3-3 해방감 | 비고            |
| ---- | --------- | ------------- | ------------- | ----------- | --------------- |
| A    | Noguchi   | 6             | 3             | 5           |                 |
| B1   | Hase      | 4             | なかった      | 2           | 추적 미인지     |
| C    | Matsumoto | 7             | なかった      | 6           | 추적 미인지     |
| D    | Ishikawa  | 1             | 1             | 4           | 추적 미인지     |
| E2   | Takamiya  | 7             | 7             | 1           | **최고 공포도** |
| F    | Sensei    | 1             | なかった      | 6           | 추적 미인지     |

---

## 피험자별 상세 프로파일

### 🔴 Subject E2 (Takamiya Session 2) - 高没入・特異反応型

- **설문**: 추적공포 7점, 근접공포 7점, 해방감 7점 (최고점)
- **생체반응**:
    - HR: **28.96 bpm** (극단적 서맥)
    - RMSSD: **4766.04 ms** (비정상적 증가)
    - EMG RMS: **102.00 μV** (전체 최고)
- **해석**: 강렬한 공포 자극에 의한 **Freezing Response** (동결 반응) 및 부교감신경의 과도한 활성화.

### 🔴 Subject E1 (Takamiya Session 1) - 中反応型

- **생체반응**:
    - HR: 76.45 bpm
    - RMSSD: 99.90 ms
    - EMG RMS: 20.14 μV

### 🟡 Subject A (Noguchi) - 認知成功・回避型

- **설문**: 추적공포 6점
- **생체반응**: CHASE 데이터 없음 (0초) - 조기 종료 또는 회피 성공

### 🟢 Subject F (Sensei) - 無意識緊張유지형

- **설문**: 추적공포 1점 (인지 못함)
- **생체반응**: HR **90.39 bpm** (높음), EMG RMS **35.17 μV** (매우 높음)
- **해석**: 인지 실패 보고와 달리 신체는 높은 각성 상태를 유지함.

### 🔵 Subject D (Ishikawa) - 低反応型

- **설문**: 추적공포 1점, 근접공포 1점
- **생체반응**: HR 78.65 bpm, RMSSD 28.35 ms
- **해석**: 안정적인 생체 신호, 낮은 주관적 공포.

### 🔵 Subject D (Ishikawa) - 低反応型

- **설문**: 추적공포 1점, 근접공포 1점
- **생체반응**: HR 101.1 bpm (6.3초)

---

## 수정 필요 사항

### 삭제 (오류)

```
心拍数は29.00 bpmと異常に低く測定された...
RMSSD 4766.04 msの極端な上昇...
「すくみ反応 (Freezing Response)」
```

### 수정 (Subject E2)

```latex
Subject E2 (Takamiya) のPhase 3における心拍数は100.4 bpmで
高い覚醒状態を維持し,EMG RMS値も31.65 μVと最高の筋緊張を示した.
アンケートでも追跡時の恐怖度7点,解放感7点と最も高い数値を記録しており,
積極的な逃避反応 (Active Flight Response) として解釈される.
```
