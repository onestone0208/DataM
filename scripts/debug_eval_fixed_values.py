#!/usr/bin/env python3
"""
평가 중 역별 고정값 패턴 확인

평가 때 역별로 고정된 값이 나오는지 확인
"""

import sys
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.evaluate_occupancy_seasonal import SeasonalOccupancyEvaluator


def debug_eval_fixed_values():
    """평가 중 역별 고정값 패턴 확인"""
    print("🔍 평가 중 역별 고정값 패턴 확인")
    print("=" * 60)
    
    evaluator = SeasonalOccupancyEvaluator('config/model_config.yaml')
    test_loader, _ = evaluator.load_test_dataset()
    model = evaluator.load_model(checkpoint_path='checkpoints/occupancy_model/checkpoint_epoch_1.pth')
    model.eval()
    
    adjacency_matrix = np.load('data/adjacency_matrix.npy')
    adjacency_tensor = torch.FloatTensor(adjacency_matrix).to(evaluator.device)
    
    print("\n1. 여러 배치에서 역별 예측값 수집:")
    print("-" * 60)
    
    # 각 역별로 예측값 수집
    station_predictions = defaultdict(list)
    num_batches = 0
    max_batches = 30  # 최대 30개 배치 확인
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= max_batches:
                break
            
            X = batch['X'].to(evaluator.device)
            node_features = batch['node_features'].to(evaluator.device)
            batch_size = X.size(0)
            adjacency = adjacency_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            
            predictions, _ = model(
                X, adjacency, target_length=1, teacher_forcing_ratio=0.0,
                node_features=node_features, date_features=None, time_features=None
            )
            predictions = predictions.squeeze(1)  # [B, N, 1]
            
            # 각 역별 예측값 저장 (모든 배치 샘플 포함)
            for b in range(predictions.size(0)):
                for station_idx in range(predictions.size(1)):
                    pred_value = predictions[b, station_idx, 0].item()
                    station_predictions[station_idx].append(pred_value)
            
            num_batches += 1
    
    print(f"총 {num_batches}개 배치에서 예측값 수집 완료")
    print(f"역 수: {len(station_predictions)}개")
    print(f"역당 평균 샘플 수: {np.mean([len(preds) for preds in station_predictions.values()]):.1f}개")
    
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
        for station_idx, mean_pred, std_pred, range_pred in fixed_stations:
            print(f"  역 {station_idx:2d}  | {mean_pred:8.4f}  | {std_pred:8.6f} | {range_pred:8.4f}")
    
    print(f"\n✅ 다양한 예측값 역 (표준편차 >= 0.01): {len(varied_stations)}개")
    if varied_stations:
        print("  역번호 | 평균 예측값 | 표준편차 | 범위")
        print("  " + "-" * 50)
        for station_idx, mean_pred, std_pred, range_pred in varied_stations[:15]:  # 최대 15개만 표시
            print(f"  역 {station_idx:2d}  | {mean_pred:8.4f}  | {std_pred:8.6f} | {range_pred:8.4f}")
        if len(varied_stations) > 15:
            print(f"  ... 외 {len(varied_stations) - 15}개")
    
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
    
    print("\n4. 구체적인 예측값 샘플 (첫 5개 역, 처음 10개만):")
    print("-" * 60)
    for station_idx in range(min(5, len(station_predictions))):
        preds = station_predictions[station_idx]
        print(f"역 {station_idx}: {preds[:10]}")


if __name__ == '__main__':
    debug_eval_fixed_values()

