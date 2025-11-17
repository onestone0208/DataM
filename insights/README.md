# 인사이트 분석 스크립트

이 폴더에는 학습된 모델을 사용하여 다양한 인사이트를 추출하는 스크립트들이 포함되어 있습니다.

## 📋 인사이트 목록

### 1. 공간 전파 분석 (`1_spatial_propagation.py`)
**특정 역의 혼잡 변화가 인접 역에 어떻게 영향을 주는지 파악**

- 환승역 중심으로 혼잡이 전파되는 패턴 분석
- 특정 구간 폐쇄/지연 시 영향도 시뮬레이션

**사용법:**
```bash
python insights/1_spatial_propagation.py --station 정부청사역 --impact 50.0 --samples 20
python insights/1_spatial_propagation.py --station 대전역 --closure  # 역 폐쇄 시뮬레이션
```

### 2. 시간대별 패턴 분석 (`2_temporal_patterns.py`)
**출근·퇴근·점심 시간대의 패턴을 자동으로 학습**

- Attention이 특정 시간 구간을 높은 가중치로 잡아냄 → 패턴 해석 가능
- 시간대별 혼잡 패턴 분석

**사용법:**
```bash
python insights/2_temporal_patterns.py --samples 50
python insights/2_temporal_patterns.py --station 정부청사역  # 특정 역 분석
```

### 3. 노드 특성 영향 분석 (`3_node_feature_impact.py`)
**노드 특성(역세권 인구, 환승 수, 상권 규모 등)이 혼잡도에 미치는 영향**

- 강남역 = 상권·유동 인구 반영으로 높은 baseline
- 정부청사 = 출근 시간 피크 강함
→ 정책적 인사이트 제공

**사용법:**
```bash
python insights/3_node_feature_impact.py --samples 30 --station 정부청사역
python insights/3_node_feature_impact.py --feature 관공서_밀도  # 특정 특성 비교
```

### 4. 시간/날짜 효과 분석 (`4_temporal_effects.py`)
**공휴일/날씨/계절 변화에 따른 혼잡 예측**

- 날짜 feature(21개)가 시계열과 결합되면서 특정 날짜 효과도 predict 가능

**사용법:**
```bash
python insights/4_temporal_effects.py
```

### 5. 승하차 흐름 분석 (`5_passenger_flow.py`)
**승하차 데이터를 직접 feature로 반영하므로 역별 편중된 흐름 분석**

- A역에서 탑승 증가 → B역 하차 증가 → C역 혼잡 변화
→ 실제 passenger flow chain 분석 가능

**사용법:**
```bash
python insights/5_passenger_flow.py --station 대전역 --boarding 50.0 --samples 20
```

### 6. 정책 시뮬레이션 (`../scripts/simulate_poi_impact.py`)
**정책 시뮬레이션 가능 (가장 중요한 실용적 의미)**

- "강남역 출구 3번 공사 시 혼잡도 변화?"
- "평일 7시 운행 편성 2대 증가 시 효과?"
- "기차 지연 시 환승역 혼잡도 예측?"

**사용법:**
```bash
python scripts/simulate_poi_impact.py --station 정부청사역 --education 0.5 --government 0.6
```

## 📊 시각화 결과

모든 스크립트는 `insights/visualizations/` 폴더에 시각화 결과를 저장합니다.

## 🔧 공통 옵션

모든 스크립트는 다음 옵션을 지원합니다:

- `--checkpoint`: 모델 체크포인트 경로 (기본값: `checkpoints/occupancy_model/best_occupancy_model.pth`)
- `--samples`: 분석할 샘플 수 (기본값: 스크립트별 상이)

## 📝 예시 실행

전체 인사이트 분석을 한 번에 실행하려면:

```bash
# 1. 공간 전파 분석
python insights/1_spatial_propagation.py --station 정부청사역 --samples 20

# 2. 시간대별 패턴 분석
python insights/2_temporal_patterns.py --samples 50

# 3. 노드 특성 영향 분석
python insights/3_node_feature_impact.py --station 정부청사역 --samples 30

# 4. 시간/날짜 효과 분석
python insights/4_temporal_effects.py

# 5. 승하차 흐름 분석
python insights/5_passenger_flow.py --station 대전역 --samples 20

# 6. POI 변화 시뮬레이션
python scripts/simulate_poi_impact.py --station 정부청사역 --education 0.5 --government 0.6
```

## 📈 결과 해석

각 스크립트는 콘솔에 통계 결과를 출력하고, 시각화 파일을 생성합니다. 결과를 해석할 때는:

1. **통계적 유의성**: 평균값과 표준편차를 함께 확인
2. **시각화**: 그래프를 통해 패턴 파악
3. **실용성**: 실제 정책 결정에 활용 가능한지 검토

## ⚠️ 주의사항

- 모든 시뮬레이션은 모델의 예측을 기반으로 하므로, 실제 현실과 다를 수 있습니다
- 샘플 수가 적을 경우 통계적 신뢰도가 낮을 수 있습니다
- 모델이 학습하지 못한 패턴은 반영되지 않을 수 있습니다

