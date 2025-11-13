#!/usr/bin/env python3
"""
학습 성능 디버깅 스크립트

학습이 느린 원인을 찾기 위한 프로파일링
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


def profile_data_loading():
    """데이터 로딩 성능 측정"""
    print("=== 데이터 로딩 성능 측정 ===")
    
    start_time = time.time()
    
    # 원본 데이터 로드
    congestion_data = pd.read_csv('data/혼잡도.csv')
    node_features = pd.read_csv('data/node_features.csv')
    date_features = pd.read_csv('data/date_features.csv')
    time_features = pd.read_csv('data/time_features.csv')
    adjacency_matrix = np.load('data/adjacency_matrix.npy')
    
    print(f"원본 데이터 로드: {time.time() - start_time:.2f}초")
    
    # 역 인덱스 매핑
    stations = sorted(congestion_data['station'].unique())
    station_to_idx = {station: idx for idx, station in enumerate(stations)}
    
    # 작은 데이터셋으로 테스트 (첫 7일만)
    dates = sorted(congestion_data['date'].unique())[:7]
    small_data = congestion_data[congestion_data['date'].isin(dates)]
    
    dataset_start = time.time()
    
    # 데이터셋 생성
    dataset = SubwayGraphDataset(
        small_data, node_features, date_features, time_features,
        adjacency_matrix, station_to_idx, sequence_length=12, prediction_length=1
    )
    
    print(f"데이터셋 생성: {time.time() - dataset_start:.2f}초")
    print(f"데이터셋 크기: {len(dataset)}")
    
    # 데이터 로더 생성
    loader_start = time.time()
    
    dataloader = DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False
    )
    
    print(f"데이터로더 생성: {time.time() - loader_start:.2f}초")
    
    # 첫 번째 배치 로드 시간 측정
    batch_start = time.time()
    first_batch = next(iter(dataloader))
    print(f"첫 번째 배치 로드: {time.time() - batch_start:.2f}초")
    
    # 배치 정보
    for key, value in first_batch.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    return dataset, dataloader


def profile_model_forward():
    """모델 forward pass 성능 측정"""
    print("\n=== 모델 Forward Pass 성능 측정 ===")
    
    # 작은 모델로 테스트
    model = DCRNN(
        input_size=2,
        hidden_size=32,  # 작게 설정
        output_size=2,
        num_layers=1,    # 작게 설정
        diffusion_steps=2,  # 작게 설정
        use_attention=True,
        dropout=0.0
    )
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"디바이스: {device}")
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 테스트 데이터
    batch_size, seq_len, num_nodes = 4, 12, 22
    X = torch.randn(batch_size, seq_len, num_nodes, 2).to(device)
    adjacency = torch.rand(batch_size, num_nodes, num_nodes).to(device)
    
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = model(X, adjacency, target_length=1)
    
    # 실제 측정
    forward_times = []
    for i in range(10):
        start_time = time.time()
        
        with torch.no_grad():
            predictions, attention_weights = model(X, adjacency, target_length=1)
        
        forward_time = time.time() - start_time
        forward_times.append(forward_time)
        
        if i == 0:
            print(f"예측 결과 shape: {predictions.shape}")
            if attention_weights is not None:
                print(f"Attention weights shape: {attention_weights.shape}")
    
    print(f"Forward pass 평균 시간: {np.mean(forward_times):.4f}초 (±{np.std(forward_times):.4f})")
    print(f"Forward pass 범위: {min(forward_times):.4f} ~ {max(forward_times):.4f}초")


def profile_training_step():
    """학습 스텝 성능 측정"""
    print("\n=== 학습 스텝 성능 측정 ===")
    
    # 작은 모델
    model = DCRNN(
        input_size=2, hidden_size=32, output_size=2,
        num_layers=1, diffusion_steps=2, use_attention=True
    )
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    loss_fn = MultiTaskLoss(loss_type='mae')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 테스트 데이터
    batch_size, seq_len, num_nodes = 4, 12, 22
    X = torch.randn(batch_size, seq_len, num_nodes, 2).to(device)
    Y = torch.randn(batch_size, num_nodes, 2).to(device)
    adjacency = torch.rand(batch_size, num_nodes, num_nodes).to(device)
    
    model.train()
    
    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        predictions, _ = model(X, adjacency, target_length=1)
        predictions = predictions.squeeze(1)
        loss_dict = loss_fn(predictions, Y)
        loss_dict['total_loss'].backward()
        optimizer.step()
    
    # 실제 측정
    step_times = []
    for i in range(10):
        step_start = time.time()
        
        optimizer.zero_grad()
        
        # Forward pass
        forward_start = time.time()
        predictions, _ = model(X, adjacency, target_length=1)
        predictions = predictions.squeeze(1)
        forward_time = time.time() - forward_start
        
        # Loss computation
        loss_start = time.time()
        loss_dict = loss_fn(predictions, Y)
        loss = loss_dict['total_loss']
        loss_time = time.time() - loss_start
        
        # Backward pass
        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start
        
        # Optimizer step
        optim_start = time.time()
        optimizer.step()
        optim_time = time.time() - optim_start
        
        total_time = time.time() - step_start
        step_times.append(total_time)
        
        if i == 0:
            print(f"  Forward: {forward_time:.4f}초")
            print(f"  Loss: {loss_time:.4f}초")
            print(f"  Backward: {backward_time:.4f}초")
            print(f"  Optimizer: {optim_time:.4f}초")
            print(f"  Total: {total_time:.4f}초")
    
    print(f"학습 스텝 평균 시간: {np.mean(step_times):.4f}초 (±{np.std(step_times):.4f})")


def main():
    print("🔍 DCRNN 학습 성능 디버깅")
    print("=" * 50)
    
    try:
        # 데이터 로딩 성능
        dataset, dataloader = profile_data_loading()
        
        # 모델 forward pass 성능
        profile_model_forward()
        
        # 학습 스텝 성능
        profile_training_step()
        
        print("\n" + "=" * 50)
        print("✅ 성능 프로파일링 완료!")
        
        # 권장사항
        print("\n💡 성능 개선 권장사항:")
        print("1. 배치 크기를 줄여보세요 (32 → 8 또는 16)")
        print("2. 모델 크기를 줄여보세요 (hidden_size: 64 → 32)")
        print("3. diffusion_steps를 줄여보세요 (3 → 2)")
        print("4. num_workers=0으로 설정하세요 (MPS 호환성)")
        print("5. 데이터셋 크기를 줄여서 테스트해보세요")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
