#!/usr/bin/env python3
"""
평가 시 입력 데이터 디버깅

실제 평가 데이터가 제대로 전달되는지 확인
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.evaluate_occupancy_seasonal import SeasonalOccupancyEvaluator


def debug_evaluation_input():
    """평가 입력 데이터 확인"""
    print("🔍 평가 입력 데이터 디버깅")
    print("=" * 60)
    
    evaluator = SeasonalOccupancyEvaluator('config/model_config.yaml')
    
    # 테스트 데이터 로드
    test_loader, test_dataset = evaluator.load_test_dataset()
    
    # 첫 번째 배치 확인
    batch = next(iter(test_loader))
    
    print("\n1. 배치 데이터 구조:")
    print("-" * 60)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}, range=[{value.min().item():.3f}, {value.max().item():.3f}]")
        else:
            print(f"{key}: {type(value)}")
    
    print("\n2. X 데이터 상세 분석:")
    print("-" * 60)
    X = batch['X']  # [B, T, N, F]
    print(f"X shape: {X.shape}")
    print(f"X 전체 통계:")
    print(f"  Mean: {X.mean().item():.4f}, Std: {X.std().item():.4f}")
    print(f"  Min: {X.min().item():.4f}, Max: {X.max().item():.4f}")
    
    # 배치별로 다른지 확인
    print(f"\n배치별 X 통계:")
    for b in range(min(3, X.size(0))):
        x_b = X[b]
        print(f"  Batch {b}: mean={x_b.mean().item():.4f}, std={x_b.std().item():.4f}, "
              f"range=[{x_b.min().item():.4f}, {x_b.max().item():.4f}]")
    
    # 역별로 다른지 확인
    print(f"\n역별 X 통계 (첫 배치, 마지막 timestep):")
    x_last = X[0, -1]  # [N, F]
    for n in range(min(5, x_last.size(0))):
        x_n = x_last[n]
        print(f"  역 {n}: mean={x_n.mean().item():.4f}, std={x_n.std().item():.4f}, "
              f"range=[{x_n.min().item():.4f}, {x_n.max().item():.4f}]")
    
    # 시간별로 다른지 확인
    print(f"\n시간별 X 통계 (첫 배치, 첫 역):")
    x_time = X[0, :, 0]  # [T, F]
    for t in range(min(5, x_time.size(0))):
        x_t = x_time[t]
        print(f"  Time {t}: mean={x_t.mean().item():.4f}, std={x_t.std().item():.4f}")
    
    print("\n3. Node Features 확인:")
    print("-" * 60)
    node_features = batch['node_features']  # [B, N, node_dim]
    print(f"Node features shape: {node_features.shape}")
    print(f"Node features 통계:")
    print(f"  Mean: {node_features.mean().item():.4f}, Std: {node_features.std().item():.4f}")
    print(f"  Range: [{node_features.min().item():.4f}, {node_features.max().item():.4f}]")
    
    # 역별로 다른지 확인
    print(f"\n역별 Node features (첫 배치):")
    nf_first = node_features[0]  # [N, node_dim]
    for n in range(min(5, nf_first.size(0))):
        nf_n = nf_first[n]
        print(f"  역 {n}: mean={nf_n.mean().item():.4f}, range=[{nf_n.min().item():.4f}, {nf_n.max().item():.4f}]")
    
    # 모든 역이 같은 node_features를 가지는지 확인
    nf_diff = (nf_first.unsqueeze(0) - nf_first.unsqueeze(1)).abs().sum()
    print(f"\n역 간 Node features 차이 합: {nf_diff.item():.4f}")
    if nf_diff.item() < 1e-6:
        print("⚠️ 모든 역이 같은 node_features를 가짐! (문제!)")
    else:
        print("✅ 역별로 다른 node_features (정상)")
    
    print("\n4. 모델에 실제 입력 전달 테스트:")
    print("-" * 60)
    
    # 모델 로드
    model = evaluator.load_model(checkpoint_path='checkpoints/occupancy_model/checkpoint_epoch_1.pth')
    model.eval()
    
    adjacency_matrix = np.load('data/adjacency_matrix.npy')
    adjacency_tensor = torch.FloatTensor(adjacency_matrix)
    
    # 첫 번째 배치로 테스트
    X_test = batch['X'][:1].to(evaluator.device)  # [1, T, N, F]
    node_features_test = batch['node_features'][:1].to(evaluator.device)  # [1, N, node_dim]
    batch_size = X_test.size(0)
    adjacency_test = adjacency_tensor.unsqueeze(0).expand(batch_size, -1, -1).to(evaluator.device)
    
    with torch.no_grad():
        predictions, _ = model(
            X_test, adjacency_test,
            target_length=1,
            teacher_forcing_ratio=0.0,
            node_features=node_features_test,
            date_features=None,
            time_features=None
        )
        predictions = predictions.squeeze(1)  # [1, N, 1]
        
        print(f"예측 결과:")
        print(f"  Shape: {predictions.shape}")
        print(f"  Range: [{predictions.min().item():.3f}, {predictions.max().item():.3f}]")
        print(f"  역별 예측: {predictions[0, :, 0].tolist()}")
        
        # 역별 예측이 다른지 확인
        pred_std = predictions[0, :, 0].std().item()
        print(f"  역별 예측 표준편차: {pred_std:.6f}")
        
        if pred_std < 1e-6:
            print("⚠️ 모든 역이 같은 예측값! (문제!)")
        else:
            print("✅ 역별로 다른 예측값 (정상)")
    
    print("\n5. 다른 배치로 테스트:")
    print("-" * 60)
    
    # 두 번째 배치
    batch_iter = iter(test_loader)
    next(batch_iter)  # 첫 번째 스킵
    batch2 = next(batch_iter)
    
    X_test2 = batch2['X'][:1].to(evaluator.device)
    node_features_test2 = batch2['node_features'][:1].to(evaluator.device)
    adjacency_test2 = adjacency_tensor.unsqueeze(0).expand(1, -1, -1).to(evaluator.device)
    
    with torch.no_grad():
        predictions2, _ = model(
            X_test2, adjacency_test2,
            target_length=1,
            teacher_forcing_ratio=0.0,
            node_features=node_features_test2,
            date_features=None,
            time_features=None
        )
        predictions2 = predictions2.squeeze(1)
        
        print(f"배치 2 예측:")
        print(f"  역별 예측: {predictions2[0, :, 0].tolist()}")
        
        # 두 배치의 예측 차이
        pred_diff = (predictions - predictions2).abs().mean()
        print(f"\n배치 간 예측 차이: {pred_diff.item():.4f}")
        
        if pred_diff.item() < 1e-3:
            print("⚠️ 두 배치가 거의 같은 예측값! (문제!)")
        else:
            print("✅ 배치별로 다른 예측값 (정상)")


if __name__ == '__main__':
    debug_evaluation_input()

