#!/usr/bin/env python3
"""
학습 중 역별 고정값 패턴 확인

평가 때처럼 역별로 고정된 값이 나오는지 확인
"""

import sys
import torch
import numpy as np
from pathlib import Path
import yaml
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.train_occupancy_model import OccupancyTrainer


def debug_training_fixed_values():
    """학습 중 역별 고정값 패턴 확인"""
    print("🔍 학습 중 역별 고정값 패턴 확인")
    print("=" * 60)
    
    # Config 로드
    with open('config/model_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 트레이너 생성
    trainer = OccupancyTrainer(config)
    trainer.load_data()
    trainer.build_model()
    trainer.setup_training()
    
    trainer.model.eval()  # 평가 모드로 확인
    
    print("\n1. 여러 배치에서 역별 예측값 수집:")
    print("-" * 60)
    
    # 각 역별로 예측값 수집
    station_predictions = defaultdict(list)
    num_batches = 20  # 20개 배치 확인
    
    batch_iter = iter(trainer.train_loader)
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(trainer.train_loader)
                batch = next(batch_iter)
            
            X = batch['X'][:1].to(trainer.device)  # 첫 배치만
            node_features = batch['node_features'][:1].to(trainer.device)
            batch_size = 1
            adjacency = trainer.adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
            
            predictions, _ = trainer.model(
                X, adjacency, target_length=1, teacher_forcing_ratio=0.0,
                node_features=node_features, date_features=None, time_features=None
            )
            predictions = predictions.squeeze(1)  # [1, N, 1]
            
            # 각 역별 예측값 저장
            for station_idx in range(predictions.size(1)):
                pred_value = predictions[0, station_idx, 0].item()
                station_predictions[station_idx].append(pred_value)
    
    print(f"총 {num_batches}개 배치에서 예측값 수집 완료")
    print(f"역 수: {len(station_predictions)}개")
    
    print("\n2. 역별 예측값 통계:")
    print("-" * 60)
    
    fixed_stations = []
    varied_stations = []
    
    for station_idx in sorted(station_predictions.keys()):
        preds = station_predictions[station_idx]
        mean_pred = np.mean(preds)
        std_pred = np.std(preds)
        min_pred = np.min(preds)
        max_pred = np.max(preds)
        range_pred = max_pred - min_pred
        
        # 표준편차가 매우 작으면 고정값으로 간주
        if std_pred < 0.01:  # 표준편차가 0.01 미만
            fixed_stations.append((station_idx, mean_pred, std_pred, range_pred))
        else:
            varied_stations.append((station_idx, mean_pred, std_pred, range_pred))
    
    print(f"\n⚠️ 고정값 패턴 역 (표준편차 < 0.01): {len(fixed_stations)}개")
    if fixed_stations:
        print("  역번호 | 평균 예측값 | 표준편차 | 범위")
        print("  " + "-" * 50)
        for station_idx, mean_pred, std_pred, range_pred in fixed_stations[:10]:  # 최대 10개만 표시
            print(f"  역 {station_idx:2d}  | {mean_pred:8.4f}  | {std_pred:8.6f} | {range_pred:8.4f}")
        if len(fixed_stations) > 10:
            print(f"  ... 외 {len(fixed_stations) - 10}개")
    
    print(f"\n✅ 다양한 예측값 역 (표준편차 >= 0.01): {len(varied_stations)}개")
    if varied_stations:
        print("  역번호 | 평균 예측값 | 표준편차 | 범위")
        print("  " + "-" * 50)
        for station_idx, mean_pred, std_pred, range_pred in varied_stations[:10]:  # 최대 10개만 표시
            print(f"  역 {station_idx:2d}  | {mean_pred:8.4f}  | {std_pred:8.6f} | {range_pred:8.4f}")
        if len(varied_stations) > 10:
            print(f"  ... 외 {len(varied_stations) - 10}개")
    
    print("\n3. 전체 통계:")
    print("-" * 60)
    all_stds = [np.std(preds) for preds in station_predictions.values()]
    print(f"역별 예측 표준편차 평균: {np.mean(all_stds):.6f}")
    print(f"역별 예측 표준편차 최소: {np.min(all_stds):.6f}")
    print(f"역별 예측 표준편차 최대: {np.max(all_stds):.6f}")
    print(f"표준편차 < 0.01인 역 비율: {len(fixed_stations) / len(station_predictions) * 100:.1f}%")
    
    if len(fixed_stations) > len(station_predictions) * 0.5:
        print("\n⚠️ 경고: 절반 이상의 역이 고정값 패턴을 보입니다!")
    elif len(fixed_stations) > 0:
        print(f"\n⚠️ 주의: {len(fixed_stations)}개 역이 고정값 패턴을 보입니다.")
    else:
        print("\n✅ 모든 역이 다양한 예측값을 생성합니다!")
    
    print("\n4. 구체적인 예측값 샘플 (첫 5개 역):")
    print("-" * 60)
    for station_idx in range(min(5, len(station_predictions))):
        preds = station_predictions[station_idx]
        print(f"역 {station_idx}: {preds[:10]}")  # 처음 10개만 표시


if __name__ == '__main__':
    debug_training_fixed_values()

