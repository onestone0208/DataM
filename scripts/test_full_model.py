#!/usr/bin/env python3
"""
전체 크기 모델 테스트

원래 설정의 모델로 첫 번째 배치 테스트
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
    print("🔍 전체 크기 모델 테스트")
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
        
        # 작은 데이터셋 (첫 5일만)
        dates = sorted(congestion_data['date'].unique())[:5]
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
            dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=False
        )
        
        print(f"배치 수: {len(dataloader)}")
        
        # 3. 원래 크기 모델 생성
        print("\n3. 원래 크기 모델 생성...")
        
        model = DCRNN(
            input_size=2,
            hidden_size=64,  # 원래 크기
            output_size=2,
            num_layers=2,    # 원래 크기
            diffusion_steps=3,  # 원래 크기
            use_attention=True,
            dropout=0.1
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"모델 파라미터: {total_params:,}")
        
        # 4. 손실 함수 및 옵티마이저
        print("\n4. 학습 설정...")
        
        loss_fn = MultiTaskLoss(loss_type='mae')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 5. 첫 번째 배치 테스트 (상세 타이밍)
        print("\n5. 첫 번째 배치 테스트...")
        
        model.train()
        
        total_start = time.time()
        
        # 배치 로드
        batch_load_start = time.time()
        batch = next(iter(dataloader))
        batch_load_time = time.time() - batch_load_start
        print(f"배치 로드 시간: {batch_load_time:.4f}초")
        
        # 데이터 준비
        data_prep_start = time.time()
        X = batch['X'].to(device)
        Y = batch['Y'].to(device)
        adjacency_matrix_tensor = torch.FloatTensor(adjacency_matrix).to(device)
        
        batch_size = X.size(0)
        adjacency = adjacency_matrix_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        data_prep_time = time.time() - data_prep_start
        print(f"데이터 준비 시간: {data_prep_time:.4f}초")
        
        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}")
        print(f"adjacency shape: {adjacency.shape}")
        
        # Forward pass
        print("\n6. Forward pass...")
        
        optimizer.zero_grad()
        
        forward_start = time.time()
        
        print("  - 모델 호출 시작...")
        predictions, attention_weights = model(
            X, adjacency, 
            target_length=1,
            teacher_forcing_ratio=1.0,
            targets=Y.unsqueeze(1)
        )
        
        predictions = predictions.squeeze(1)
        forward_time = time.time() - forward_start
        print(f"  - Forward pass 완료: {forward_time:.4f}초")
        print(f"  - Predictions shape: {predictions.shape}")
        
        # 손실 계산
        print("\n7. 손실 계산...")
        loss_start = time.time()
        loss_dict = loss_fn(predictions, Y)
        loss = loss_dict['total_loss']
        loss_time = time.time() - loss_start
        print(f"  - 손실 계산 완료: {loss_time:.4f}초")
        print(f"  - Loss: {loss.item():.4f}")
        
        # Backward pass
        print("\n8. Backward pass...")
        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start
        print(f"  - Backward pass 완료: {backward_time:.4f}초")
        
        # Optimizer step
        print("\n9. Optimizer step...")
        optim_start = time.time()
        optimizer.step()
        optim_time = time.time() - optim_start
        print(f"  - Optimizer step 완료: {optim_time:.4f}초")
        
        total_time = time.time() - total_start
        print(f"\n총 시간: {total_time:.4f}초")
        
        # 메모리 사용량 확인 (MPS)
        if device.type == 'mps':
            print(f"\nMPS 메모리 사용량: {torch.mps.current_allocated_memory() / 1024**2:.1f} MB")
        
        print("\n✅ 전체 크기 모델 테스트 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
