#!/usr/bin/env python3
"""
학습 루프 테스트

실제 학습 루프와 유사한 환경에서 여러 배치 처리 테스트
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
    print("🔍 학습 루프 테스트")
    print("=" * 50)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"디바이스: {device}")
    
    try:
        # 1. 데이터 로드
        print("\n1. 데이터 로드...")
        
        congestion_data = pd.read_csv('data/혼잡도.csv')
        node_features = pd.read_csv('data/node_features.csv')
        date_features = pd.read_csv('data/date_features.csv')
        time_features = pd.read_csv('data/time_features.csv')
        adjacency_matrix = np.load('data/adjacency_matrix.npy')
        
        # 작은 데이터셋 (첫 10일)
        dates = sorted(congestion_data['date'].unique())[:10]
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
            dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False
        )
        
        print(f"배치 수: {len(dataloader)}")
        
        # 3. 모델 생성
        print("\n3. 모델 생성...")
        
        model = DCRNN(
            input_size=2,
            hidden_size=64,
            output_size=2,
            num_layers=2,
            diffusion_steps=3,
            use_attention=True,
            dropout=0.1
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"모델 파라미터: {total_params:,}")
        
        # 4. 학습 설정
        print("\n4. 학습 설정...")
        
        loss_fn = MultiTaskLoss(loss_type='mae')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 인접행렬을 미리 GPU로 이동
        adjacency_matrix_tensor = torch.FloatTensor(adjacency_matrix).to(device)
        
        # 5. 학습 루프 테스트 (첫 10개 배치)
        print("\n5. 학습 루프 테스트...")
        
        model.train()
        
        batch_times = []
        losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 10:  # 첫 10개 배치만
                break
                
            batch_start = time.time()
            
            print(f"\n--- 배치 {batch_idx + 1} ---")
            
            # 데이터 준비
            X = batch['X'].to(device)
            Y = batch['Y'].to(device)
            
            batch_size = X.size(0)
            adjacency = adjacency_matrix_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Forward pass
            optimizer.zero_grad()
            
            predictions, attention_weights = model(
                X, adjacency, 
                target_length=1,
                teacher_forcing_ratio=1.0,
                targets=Y.unsqueeze(1)
            )
            
            predictions = predictions.squeeze(1)
            
            # 손실 계산
            loss_dict = loss_fn(predictions, Y)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step
            optimizer.step()
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            losses.append(loss.item())
            
            print(f"  시간: {batch_time:.4f}초")
            print(f"  Loss: {loss.item():.4f}")
            
            # 메모리 사용량 (MPS)
            if device.type == 'mps':
                memory_mb = torch.mps.current_allocated_memory() / 1024**2
                print(f"  메모리: {memory_mb:.1f} MB")
            
            # 메모리 정리
            del X, Y, predictions, loss
            if device.type == 'mps':
                torch.mps.empty_cache()
        
        # 6. 결과 분석
        print(f"\n6. 결과 분석")
        print(f"평균 배치 시간: {np.mean(batch_times):.4f}초 (±{np.std(batch_times):.4f})")
        print(f"최대 배치 시간: {max(batch_times):.4f}초")
        print(f"최소 배치 시간: {min(batch_times):.4f}초")
        print(f"평균 Loss: {np.mean(losses):.4f}")
        
        # 시간 증가 패턴 확인
        if len(batch_times) > 5:
            first_half = np.mean(batch_times[:len(batch_times)//2])
            second_half = np.mean(batch_times[len(batch_times)//2:])
            print(f"전반부 평균: {first_half:.4f}초")
            print(f"후반부 평균: {second_half:.4f}초")
            
            if second_half > first_half * 1.5:
                print("⚠️  시간이 점진적으로 증가하고 있습니다 (메모리 누수 가능성)")
            else:
                print("✅ 시간이 안정적입니다")
        
        print("\n✅ 학습 루프 테스트 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
