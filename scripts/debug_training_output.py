#!/usr/bin/env python3
"""
학습 과정에서 모델 출력 디버깅

학습 중에도 모델이 입력을 제대로 구분하는지 확인
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.train_occupancy_model import OccupancyTrainer, OccupancyDataset
from models.dcrnn_layers import DCRNN


def debug_training_output():
    """학습 과정 출력 확인"""
    print("🔍 학습 과정 모델 출력 디버깅")
    print("=" * 60)
    
    # Config 로드
    with open('config/model_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 트레이너 생성 (데이터만 로드)
    trainer = OccupancyTrainer(config)
    trainer.load_data()
    trainer.build_model()
    trainer.setup_training()
    
    print("\n1. 학습 데이터 배치 확인:")
    print("-" * 60)
    
    # 첫 번째 배치
    batch1 = next(iter(trainer.train_loader))
    
    # 두 번째 배치
    batch_iter = iter(trainer.train_loader)
    next(batch_iter)
    batch2 = next(batch_iter)
    
    X1 = batch1['X'].to(trainer.device)
    X2 = batch2['X'].to(trainer.device)
    Y1 = batch1['Y'].to(trainer.device)
    Y2 = batch2['Y'].to(trainer.device)
    node_features1 = batch1['node_features'].to(trainer.device)
    node_features2 = batch2['node_features'].to(trainer.device)
    
    batch_size = X1.size(0)
    adjacency = trainer.adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
    
    print(f"배치 1 X 통계: mean={X1.mean().item():.4f}, std={X1.std().item():.4f}, "
          f"range=[{X1.min().item():.4f}, {X1.max().item():.4f}]")
    print(f"배치 2 X 통계: mean={X2.mean().item():.4f}, std={X2.std().item():.4f}, "
          f"range=[{X2.min().item():.4f}, {X2.max().item():.4f}]")
    x_diff = (X1 - X2).abs().mean()
    print(f"X 차이: {x_diff.item():.4f}")
    
    print(f"\n배치 1 Y 통계: mean={Y1.mean().item():.4f}, std={Y1.std().item():.4f}, "
          f"range=[{Y1.min().item():.4f}, {Y1.max().item():.4f}]")
    print(f"배치 2 Y 통계: mean={Y2.mean().item():.4f}, std={Y2.std().item():.4f}, "
          f"range=[{Y2.min().item():.4f}, {Y2.max().item():.4f}]")
    y_diff = (Y1 - Y2).abs().mean()
    print(f"Y 차이: {y_diff.item():.4f}")
    
    print("\n2. 모델 출력 비교 (학습 모드):")
    print("-" * 60)
    
    trainer.model.train()
    
    with torch.no_grad():
        # 배치 1 예측
        pred1, _ = trainer.model(
            X1, adjacency,
            target_length=1,
            teacher_forcing_ratio=0.0,  # Teacher forcing 없이
            node_features=node_features1,
            date_features=None,
            time_features=None
        )
        pred1 = pred1.squeeze(1)  # [B, N, 1]
        
        # 배치 2 예측
        pred2, _ = trainer.model(
            X2, adjacency,
            target_length=1,
            teacher_forcing_ratio=0.0,
            node_features=node_features2,
            date_features=None,
            time_features=None
        )
        pred2 = pred2.squeeze(1)
        
        print(f"배치 1 예측 통계:")
        print(f"  mean={pred1.mean().item():.4f}, std={pred1.std().item():.4f}")
        print(f"  range=[{pred1.min().item():.4f}, {pred1.max().item():.4f}]")
        print(f"  역별 예측 (첫 배치, 첫 5개 역): {pred1[0, :5, 0].tolist()}")
        
        print(f"\n배치 2 예측 통계:")
        print(f"  mean={pred2.mean().item():.4f}, std={pred2.std().item():.4f}")
        print(f"  range=[{pred2.min().item():.4f}, {pred2.max().item():.4f}]")
        print(f"  역별 예측 (첫 배치, 첫 5개 역): {pred2[0, :5, 0].tolist()}")
        
        pred_diff = (pred1 - pred2).abs().mean()
        print(f"\n예측 차이: {pred_diff.item():.4f}")
        
        if pred_diff.item() < 1e-3:
            print("⚠️ 두 배치가 거의 같은 예측! (문제!)")
        else:
            print("✅ 배치별로 다른 예측 (정상)")
    
    print("\n3. Encoder 출력 비교 (학습 모드):")
    print("-" * 60)
    
    trainer.model.train()
    
    with torch.no_grad():
        # 특성 결합
        X1_aug = trainer.model._augment_features(X1, node_features1, None, None)
        X2_aug = trainer.model._augment_features(X2, node_features2, None, None)
        
        print(f"배치 1 Augmented X: mean={X1_aug.mean().item():.4f}, std={X1_aug.std().item():.4f}")
        print(f"배치 2 Augmented X: mean={X2_aug.mean().item():.4f}, std={X2_aug.std().item():.4f}")
        aug_diff = (X1_aug - X2_aug).abs().mean()
        print(f"Augmented X 차이: {aug_diff.item():.4f}")
        
        # Encoder 출력
        encoder_hidden1, encoder_outputs1 = trainer.model.encode(X1_aug, adjacency)
        encoder_hidden2, encoder_outputs2 = trainer.model.encode(X2_aug, adjacency)
        
        h1 = encoder_hidden1[-1]
        h2 = encoder_hidden2[-1]
        
        print(f"\n배치 1 Encoder hidden: mean={h1.mean().item():.4f}, std={h1.std().item():.4f}")
        print(f"배치 2 Encoder hidden: mean={h2.mean().item():.4f}, std={h2.std().item():.4f}")
        h_diff = (h1 - h2).abs().mean()
        print(f"Encoder hidden 차이: {h_diff.item():.4f}")
        
        if h_diff.item() < 1e-3:
            print("⚠️ 두 배치가 거의 같은 Encoder hidden state! (문제!)")
        else:
            print("✅ 배치별로 다른 Encoder hidden state (정상)")
    
    print("\n4. 실제 학습 스텝 테스트:")
    print("-" * 60)
    
    trainer.model.train()
    trainer.optimizer.zero_grad()
    
    # 배치 1로 학습
    predictions1, _ = trainer.model(
        X1, adjacency,
        target_length=1,
        teacher_forcing_ratio=0.7,
        targets=Y1.unsqueeze(1),
        node_features=node_features1,
        date_features=None,
        time_features=None
    )
    predictions1 = predictions1.squeeze(1)
    
    loss1 = trainer.loss_fn(predictions1, Y1, batch1['Y_raw'].to(trainer.device))
    loss1.backward()
    trainer.optimizer.step()
    
    print(f"배치 1 학습 후 예측:")
    print(f"  역별 예측 (첫 배치, 첫 5개 역): {predictions1[0, :5, 0].detach().tolist()}")
    print(f"  Loss: {loss1.item():.4f}")
    
    # 배치 2로 학습
    trainer.optimizer.zero_grad()
    
    predictions2, _ = trainer.model(
        X2, adjacency,
        target_length=1,
        teacher_forcing_ratio=0.7,
        targets=Y2.unsqueeze(1),
        node_features=node_features2,
        date_features=None,
        time_features=None
    )
    predictions2 = predictions2.squeeze(1)
    
    loss2 = trainer.loss_fn(predictions2, Y2, batch2['Y_raw'].to(trainer.device))
    loss2.backward()
    trainer.optimizer.step()
    
    print(f"\n배치 2 학습 후 예측:")
    print(f"  역별 예측 (첫 배치, 첫 5개 역): {predictions2[0, :5, 0].detach().tolist()}")
    print(f"  Loss: {loss2.item():.4f}")
    
    # 두 배치의 예측 차이
    pred_diff_after = (predictions1 - predictions2).abs().mean()
    print(f"\n학습 후 예측 차이: {pred_diff_after.item():.4f}")
    
    if pred_diff_after.item() < 1e-3:
        print("⚠️ 학습 후에도 두 배치가 거의 같은 예측! (문제!)")
    else:
        print("✅ 학습 후에도 배치별로 다른 예측 (정상)")
    
    print("\n5. 여러 배치 연속 학습 테스트:")
    print("-" * 60)
    
    # 5개 배치 연속 학습
    batch_iter = iter(trainer.train_loader)
    predictions_list = []
    
    for i in range(5):
        batch = next(batch_iter)
        X = batch['X'].to(trainer.device)
        Y = batch['Y'].to(trainer.device)
        node_features = batch['node_features'].to(trainer.device)
        batch_size = X.size(0)
        adjacency = trainer.adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        
        trainer.optimizer.zero_grad()
        
        predictions, _ = trainer.model(
            X, adjacency,
            target_length=1,
            teacher_forcing_ratio=0.7,
            targets=Y.unsqueeze(1),
            node_features=node_features,
            date_features=None,
            time_features=None
        )
        predictions = predictions.squeeze(1)
        
        loss = trainer.loss_fn(predictions, Y, batch['Y_raw'].to(trainer.device))
        loss.backward()
        trainer.optimizer.step()
        
        # 첫 배치, 첫 역의 예측 저장
        pred_value = predictions[0, 0, 0].item()
        predictions_list.append(pred_value)
        
        print(f"배치 {i+1}: Loss={loss.item():.4f}, 첫 역 예측={pred_value:.4f}")
    
    # 예측 다양성 확인
    pred_std = np.std(predictions_list)
    print(f"\n5개 배치 연속 학습 후 첫 역 예측 표준편차: {pred_std:.6f}")
    
    if pred_std < 1e-3:
        print("⚠️ 모든 배치가 거의 같은 예측! (문제!)")
    else:
        print("✅ 배치별로 다른 예측 생성 (정상)")


if __name__ == '__main__':
    debug_training_output()

