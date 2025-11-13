#!/usr/bin/env python3
"""
최소한의 학습 테스트

첫 번째 배치에서 멈추는 문제를 디버깅
"""

import sys
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN
from models.losses import MultiTaskLoss
from models.dataset import SubwayGraphDataset
from torch.utils.data import DataLoader


def main():
    print("🔍 최소한의 학습 테스트")
    print("=" * 50)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"디바이스: {device}")
    
    try:
        # 1. 데이터 로드 (작은 데이터셋)
        print("\n1. 데이터 로드...")
        
        congestion_data = pd.read_csv('data/혼잡도.csv')
        node_features = pd.read_csv('data/node_features.csv')
        date_features = pd.read_csv('data/date_features.csv')
        time_features = pd.read_csv('data/time_features.csv')
        adjacency_matrix = np.load('data/adjacency_matrix.npy')
        
        # 매우 작은 데이터셋 (첫 3일만)
        dates = sorted(congestion_data['date'].unique())[:3]
        small_data = congestion_data[congestion_data['date'].isin(dates)]
        
        stations = sorted(congestion_data['station'].unique())
        station_to_idx = {station: idx for idx, station in enumerate(stations)}
        
        print(f"데이터 크기: {len(small_data)} 레코드")
        
        # 2. 데이터셋 생성
        print("\n2. 데이터셋 생성...")
        
        dataset = SubwayGraphDataset(
            small_data, node_features, date_features, time_features,
            adjacency_matrix, station_to_idx, sequence_length=12, prediction_length=1
        )
        
        dataloader = DataLoader(
            dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=False
        )
        
        print(f"배치 수: {len(dataloader)}")
        
        # 3. 모델 생성 (작은 모델)
        print("\n3. 모델 생성...")
        
        model = DCRNN(
            input_size=2,
            hidden_size=16,  # 매우 작게
            output_size=2,
            num_layers=1,
            diffusion_steps=2,
            use_attention=True,
            dropout=0.0
        ).to(device)
        
        print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")
        
        # 4. 손실 함수 및 옵티마이저
        print("\n4. 학습 설정...")
        
        loss_fn = MultiTaskLoss(loss_type='mae')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 5. 첫 번째 배치 테스트
        print("\n5. 첫 번째 배치 테스트...")
        
        model.train()
        
        batch_start = time.time()
        batch = next(iter(dataloader))
        print(f"배치 로드 시간: {time.time() - batch_start:.4f}초")
        
        # 데이터 준비
        X = batch['X'].to(device)
        Y = batch['Y'].to(device)
        adjacency_matrix_tensor = torch.FloatTensor(adjacency_matrix).to(device)
        
        batch_size = X.size(0)
        adjacency = adjacency_matrix_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        
        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}")
        print(f"adjacency shape: {adjacency.shape}")
        
        # 6. Forward pass 테스트
        print("\n6. Forward pass 테스트...")
        
        forward_start = time.time()
        
        optimizer.zero_grad()
        
        predictions, attention_weights = model(
            X, adjacency, 
            target_length=1,
            teacher_forcing_ratio=1.0,
            targets=Y.unsqueeze(1)
        )
        
        predictions = predictions.squeeze(1)
        
        print(f"Forward pass 시간: {time.time() - forward_start:.4f}초")
        print(f"Predictions shape: {predictions.shape}")
        
        # 7. 손실 계산 테스트
        print("\n7. 손실 계산 테스트...")
        
        loss_start = time.time()
        loss_dict = loss_fn(predictions, Y)
        loss = loss_dict['total_loss']
        print(f"손실 계산 시간: {time.time() - loss_start:.4f}초")
        print(f"Loss: {loss.item():.4f}")
        
        # 8. Backward pass 테스트
        print("\n8. Backward pass 테스트...")
        
        backward_start = time.time()
        loss.backward()
        print(f"Backward pass 시간: {time.time() - backward_start:.4f}초")
        
        # 9. Optimizer step 테스트
        print("\n9. Optimizer step 테스트...")
        
        optim_start = time.time()
        optimizer.step()
        print(f"Optimizer step 시간: {time.time() - optim_start:.4f}초")
        
        print(f"\n총 시간: {time.time() - batch_start:.4f}초")
        
        # 10. 두 번째 배치 테스트
        print("\n10. 두 번째 배치 테스트...")
        
        if len(dataloader) > 1:
            batch_iter = iter(dataloader)
            next(batch_iter)  # 첫 번째 스킵
            
            second_batch_start = time.time()
            batch2 = next(batch_iter)
            
            X2 = batch2['X'].to(device)
            Y2 = batch2['Y'].to(device)
            
            optimizer.zero_grad()
            predictions2, _ = model(X2, adjacency, target_length=1)
            predictions2 = predictions2.squeeze(1)
            
            loss_dict2 = loss_fn(predictions2, Y2)
            loss2 = loss_dict2['total_loss']
            loss2.backward()
            optimizer.step()
            
            print(f"두 번째 배치 시간: {time.time() - second_batch_start:.4f}초")
            print(f"Loss: {loss2.item():.4f}")
        
        print("\n✅ 모든 테스트 통과!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
