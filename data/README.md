# 데이터 파일 설명

## 핵심 데이터
- `혼잡도.csv`: 역별 시간대별 혼잡도 타깃 데이터 (128K+ 레코드)
- `승하차.csv`: 원본 승하차 데이터
- `node_features.csv`: 역별 특징 벡터 (22개 역 × 16차원)
- `date_features.csv`: 날짜별 특징 벡터 (243일 × 24차원)
- `time_features.csv`: 시간대별 특징 벡터 (24시간 × 9차원)

## 분할 데이터
- `승하차_train.csv`: 훈련용 승하차 데이터
- `승하차_val.csv`: 검증용 승하차 데이터
- `승하차_test.csv`: 테스트용 승하차 데이터
- `날짜정보_split.csv`: 날짜별 분할 정보

## 원본 메타데이터
- `지도.csv`: 역 기본 정보 (좌표, 연면적 등)
- `주요장소.csv`: POI 데이터 (태깅된 버전, 387개)
- `공휴일.csv`: 공휴일 정보
- `기온.csv`: 기상 데이터 (일별 기온)
- `날짜정보.csv`: 날짜 메타데이터

## 분석 결과
- `los_calculated.csv`: LOS 등급 계산 결과
- `통계.csv`: 통계 분석 데이터

## 참고 자료
- `poi_category_samples.txt`: POI 카테고리 분류 샘플
- `node_features_info.json`: 노드 특징 통계 정보

## 데이터 플로우
```
원본 데이터 → 전처리 → 특징 생성 → 모델 입력
├── 승하차.csv → 혼잡도.csv (타깃)
├── 지도.csv + 주요장소.csv → node_features.csv
├── 공휴일.csv + 기온.csv → date_features.csv
└── 시간대 정보 → time_features.csv
```
