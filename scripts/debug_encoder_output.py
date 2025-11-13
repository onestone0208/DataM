#!/usr/bin/env python3
"""
Encoder 출력 디버깅

Encoder가 입력에 따라 다른 출력을 생성하는지 확인
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.evaluate_occupancy_seasonal import SeasonalOccupancyEvaluator
from models.dcrnn_layers import DCRNN


def debug_encoder_output():
    """Encoder 출력 확인"""
    print("🔍 Encoder 출력 디버깅")
    print("=" * 60)
    
    evaluator = SeasonalOccupancyEvaluator('config/model_config.yaml')
    test_loader, _ = evaluator.load_test_dataset()
    model = evaluator.load_model(checkpoint_path='checkpoints/occupancy_model/checkpoint_epoch_1.pth')
    model.eval()
    
    adjacency_matrix = np.load('data/adjacency_matrix.npy')
    adjacency_tensor = torch.FloatTensor(adjacency_matrix).to(evaluator.device)
    
    # 두 개의 다른 배치 가져오기
    batch_iter = iter(test_loader)
    batch1 = next(batch_iter)
    batch2 = next(batch_iter)
    
    X1 = batch1['X'][:1].to(evaluator.device)
    X2 = batch2['X'][:1].to(evaluator.device)
    node_features1 = batch1['node_features'][:1].to(evaluator.device)
    node_features2 = batch2['node_features'][:1].to(evaluator.device)
    
    batch_size = 1
    adjacency = adjacency_tensor.unsqueeze(0).expand(batch_size, -1, -1).to(evaluator.device)
    
    print("\n1. 입력 데이터 비교:")
    print("-" * 60)
    print(f"배치 1 X 통계: mean={X1.mean().item():.4f}, std={X1.std().item():.4f}")
    print(f"배치 2 X 통계: mean={X2.mean().item():.4f}, std={X2.std().item():.4f}")
    x_diff = (X1 - X2).abs().mean()
    print(f"X 차이: {x_diff.item():.4f}")
    
    print(f"\n배치 1 Node features 통계: mean={node_features1.mean().item():.4f}")
    print(f"배치 2 Node features 통계: mean={node_features2.mean().item():.4f}")
    nf_diff = (node_features1 - node_features2).abs().mean()
    print(f"Node features 차이: {nf_diff.item():.4f}")
    
    print("\n2. 특성 결합 후 비교:")
    print("-" * 60)
    
    with torch.no_grad():
        # 특성 결합
        X1_aug = model._augment_features(X1, node_features1, None, None)
        X2_aug = model._augment_features(X2, node_features2, None, None)
        
        print(f"배치 1 Augmented X: mean={X1_aug.mean().item():.4f}, std={X1_aug.std().item():.4f}")
        print(f"배치 2 Augmented X: mean={X2_aug.mean().item():.4f}, std={X2_aug.std().item():.4f}")
        aug_diff = (X1_aug - X2_aug).abs().mean()
        print(f"Augmented X 차이: {aug_diff.item():.4f}")
        
        print("\n3. Encoder 출력 비교:")
        print("-" * 60)
        
        encoder_hidden1, encoder_outputs1 = model.encode(X1_aug, adjacency)
        encoder_hidden2, encoder_outputs2 = model.encode(X2_aug, adjacency)
        
        print(f"배치 1 Encoder hidden (마지막 레이어):")
        h1 = encoder_hidden1[-1]
        print(f"  mean={h1.mean().item():.4f}, std={h1.std().item():.4f}, "
              f"range=[{h1.min().item():.4f}, {h1.max().item():.4f}]")
        print(f"  역별 값 (첫 5개): {h1[0, :5, 0].tolist()}")
        
        print(f"\n배치 2 Encoder hidden (마지막 레이어):")
        h2 = encoder_hidden2[-1]
        print(f"  mean={h2.mean().item():.4f}, std={h2.std().item():.4f}, "
              f"range=[{h2.min().item():.4f}, {h2.max().item():.4f}]")
        print(f"  역별 값 (첫 5개): {h2[0, :5, 0].tolist()}")
        
        h_diff = (h1 - h2).abs().mean()
        print(f"\nEncoder hidden 차이: {h_diff.item():.4f}")
        
        if h_diff.item() < 1e-3:
            print("⚠️ 두 배치가 거의 같은 Encoder hidden state! (문제!)")
        else:
            print("✅ 배치별로 다른 Encoder hidden state (정상)")
        
        print("\n4. 디코더 초기 입력 비교:")
        print("-" * 60)
        
        initial_input1 = model.hidden_to_input_projection(h1)
        initial_input2 = model.hidden_to_input_projection(h2)
        
        print(f"배치 1 디코더 초기 입력:")
        print(f"  mean={initial_input1.mean().item():.4f}, std={initial_input1.std().item():.4f}")
        print(f"  역별 값 (첫 5개): {initial_input1[0, :5, 0].tolist()}")
        
        print(f"\n배치 2 디코더 초기 입력:")
        print(f"  mean={initial_input2.mean().item():.4f}, std={initial_input2.std().item():.4f}")
        print(f"  역별 값 (첫 5개): {initial_input2[0, :5, 0].tolist()}")
        
        init_diff = (initial_input1 - initial_input2).abs().mean()
        print(f"\n디코더 초기 입력 차이: {init_diff.item():.4f}")
        
        if init_diff.item() < 1e-3:
            print("⚠️ 두 배치가 거의 같은 디코더 초기 입력! (문제!)")
        else:
            print("✅ 배치별로 다른 디코더 초기 입력 (정상)")
        
        print("\n5. 최종 예측 비교:")
        print("-" * 60)
        
        pred1, _ = model(X1, adjacency, target_length=1, teacher_forcing_ratio=0.0,
                        node_features=node_features1, date_features=None, time_features=None)
        pred1 = pred1.squeeze(1)
        
        pred2, _ = model(X2, adjacency, target_length=1, teacher_forcing_ratio=0.0,
                        node_features=node_features2, date_features=None, time_features=None)
        pred2 = pred2.squeeze(1)
        
        print(f"배치 1 예측: {pred1[0, :5, 0].tolist()}")
        print(f"배치 2 예측: {pred2[0, :5, 0].tolist()}")
        
        pred_diff = (pred1 - pred2).abs().mean()
        print(f"\n예측 차이: {pred_diff.item():.4f}")
        
        if pred_diff.item() < 1e-3:
            print("⚠️ 두 배치가 거의 같은 예측! (문제!)")
            print("\n🔍 원인 분석:")
            if h_diff.item() < 1e-3:
                print("  - Encoder가 입력을 구분하지 못함")
            elif init_diff.item() < 1e-3:
                print("  - 디코더 초기 입력이 동일함")
            else:
                print("  - 디코더가 초기 입력을 무시함")
        else:
            print("✅ 배치별로 다른 예측 (정상)")


if __name__ == '__main__':
    debug_encoder_output()

