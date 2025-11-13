#!/usr/bin/env python3
"""
학습 과정 간단 디버깅

학습 중 모델이 입력을 제대로 구분하는지 간단히 확인
"""

import sys
import torch
import numpy as np
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.train_occupancy_model import OccupancyTrainer


def debug_training_simple():
    """학습 과정 간단 확인"""
    print("🔍 학습 과정 모델 출력 간단 확인")
    print("=" * 60)
    
    # Config 로드
    with open('config/model_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 트레이너 생성
    trainer = OccupancyTrainer(config)
    trainer.load_data()
    trainer.build_model()
    trainer.setup_training()
    
    print("\n1. 학습 데이터 배치 확인:")
    print("-" * 60)
    
    batch_iter = iter(trainer.train_loader)
    batch1 = next(batch_iter)
    batch2 = next(batch_iter)
    
    X1 = batch1['X'][:1].to(trainer.device)  # 첫 배치만
    X2 = batch2['X'][:1].to(trainer.device)
    node_features1 = batch1['node_features'][:1].to(trainer.device)
    node_features2 = batch2['node_features'][:1].to(trainer.device)
    
    batch_size = 1
    adjacency = trainer.adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
    
    print(f"배치 1 X: mean={X1.mean().item():.4f}, range=[{X1.min().item():.4f}, {X1.max().item():.4f}]")
    print(f"배치 2 X: mean={X2.mean().item():.4f}, range=[{X2.min().item():.4f}, {X2.max().item():.4f}]")
    x_diff = (X1 - X2).abs().mean()
    print(f"X 차이: {x_diff.item():.4f}")
    
    print("\n2. 모델 출력 비교 (학습 모드, Teacher Forcing 없음):")
    print("-" * 60)
    
    trainer.model.train()
    
    with torch.no_grad():
        pred1, _ = trainer.model(
            X1, adjacency, target_length=1, teacher_forcing_ratio=0.0,
            node_features=node_features1, date_features=None, time_features=None
        )
        pred1 = pred1.squeeze(1)
        
        pred2, _ = trainer.model(
            X2, adjacency, target_length=1, teacher_forcing_ratio=0.0,
            node_features=node_features2, date_features=None, time_features=None
        )
        pred2 = pred2.squeeze(1)
        
        print(f"배치 1 예측: mean={pred1.mean().item():.4f}, range=[{pred1.min().item():.4f}, {pred1.max().item():.4f}]")
        print(f"  역별 예측 (첫 5개): {pred1[0, :5, 0].tolist()}")
        
        print(f"\n배치 2 예측: mean={pred2.mean().item():.4f}, range=[{pred2.min().item():.4f}, {pred2.max().item():.4f}]")
        print(f"  역별 예측 (첫 5개): {pred2[0, :5, 0].tolist()}")
        
        pred_diff = (pred1 - pred2).abs().mean()
        print(f"\n예측 차이: {pred_diff.item():.4f}")
        
        if pred_diff.item() < 1e-3:
            print("⚠️ 두 배치가 거의 같은 예측! (문제!)")
        else:
            print("✅ 배치별로 다른 예측 (정상)")
    
    print("\n3. Encoder 출력 비교:")
    print("-" * 60)
    
    trainer.model.train()
    
    with torch.no_grad():
        X1_aug = trainer.model._augment_features(X1, node_features1, None, None)
        X2_aug = trainer.model._augment_features(X2, node_features2, None, None)
        
        encoder_hidden1, _ = trainer.model.encode(X1_aug, adjacency)
        encoder_hidden2, _ = trainer.model.encode(X2_aug, adjacency)
        
        h1 = encoder_hidden1[-1]
        h2 = encoder_hidden2[-1]
        
        h_diff = (h1 - h2).abs().mean()
        print(f"Encoder hidden 차이: {h_diff.item():.4f}")
        
        if h_diff.item() < 1e-3:
            print("⚠️ 두 배치가 거의 같은 Encoder hidden state! (문제!)")
        else:
            print("✅ 배치별로 다른 Encoder hidden state (정상)")
    
    print("\n4. 실제 학습 스텝 (1번만):")
    print("-" * 60)
    
    trainer.model.train()
    trainer.optimizer.zero_grad()
    
    Y1 = batch1['Y'][:1].to(trainer.device)
    
    predictions1, _ = trainer.model(
        X1, adjacency, target_length=1, teacher_forcing_ratio=0.7,
        targets=Y1.unsqueeze(1), node_features=node_features1,
        date_features=None, time_features=None
    )
    predictions1 = predictions1.squeeze(1)
    
    loss1 = trainer.loss_fn(predictions1, Y1, batch1['Y_raw'][:1].to(trainer.device))
    
    print(f"학습 전 예측 (첫 5개 역): {predictions1[0, :5, 0].detach().tolist()}")
    print(f"Loss: {loss1.item():.4f}")
    
    loss1.backward()
    trainer.optimizer.step()
    
    # 학습 후 다시 예측
    with torch.no_grad():
        predictions1_after, _ = trainer.model(
            X1, adjacency, target_length=1, teacher_forcing_ratio=0.0,
            node_features=node_features1, date_features=None, time_features=None
        )
        predictions1_after = predictions1_after.squeeze(1)
        
        print(f"\n학습 후 예측 (첫 5개 역): {predictions1_after[0, :5, 0].tolist()}")
        
        change = (predictions1 - predictions1_after).abs().mean()
        print(f"학습 전후 예측 변화: {change.item():.4f}")
        
        if change.item() < 1e-6:
            print("⚠️ 학습해도 예측이 거의 변하지 않음! (문제!)")
        else:
            print("✅ 학습 후 예측이 변화함 (정상)")


if __name__ == '__main__':
    debug_training_simple()

