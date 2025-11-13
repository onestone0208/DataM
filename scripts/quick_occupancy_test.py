#!/usr/bin/env python3
"""
현재 훈련 중인 혼잡도 모델 실시간 테스트
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN
from scripts.train_occupancy_model import OccupancyDataset

def quick_test():
    print("🔍 현재 혼잡도 모델 실시간 테스트")
    
    # 최신 체크포인트 로드
    checkpoint_path = "checkpoints/occupancy_model/best_occupancy_model.pth"
    device = torch.device('cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    epoch = checkpoint['epoch']
    
    print(f"📂 체크포인트 로드: Epoch {epoch}")
    
    # 모델 생성
    model = DCRNN(
        input_size=1,
        hidden_size=config['model']['hidden_size'],
        output_size=1,
        num_layers=config['model']['num_layers'],
        diffusion_steps=config['model']['diffusion_steps'],
        use_attention=config['model']['use_attention'],
        dropout=config['model']['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 🔧 훈련 모드와 평가 모드 둘 다 테스트
    print("\n" + "="*50)
    print("🧪 훈련 모드 vs 평가 모드 비교")
    print("="*50)
    
    # 간단한 테스트 데이터 생성
    batch_size, seq_len, num_nodes, features = 2, 12, 22, 1
    X = torch.randn(batch_size, seq_len, num_nodes, features) * 10 + 50  # 평균 50 정도
    adjacency = torch.eye(num_nodes).unsqueeze(0).expand(batch_size, -1, -1)
    
    print(f"입력 데이터 범위: [{X.min().item():.1f}, {X.max().item():.1f}]")
    
    # 훈련 모드
    model.train()
    with torch.no_grad():
        pred_train, _ = model(X, adjacency, target_length=1)
        pred_train = pred_train.squeeze()
    
    print(f"훈련 모드 예측 범위: [{pred_train.min().item():.1f}, {pred_train.max().item():.1f}]")
    
    # 평가 모드
    model.eval()
    with torch.no_grad():
        pred_eval, _ = model(X, adjacency, target_length=1)
        pred_eval = pred_eval.squeeze()
    
    print(f"평가 모드 예측 범위: [{pred_eval.min().item():.1f}, {pred_eval.max().item():.1f}]")
    
    # 실제 데이터로 테스트
    print("\n" + "="*50)
    print("🔍 실제 데이터로 테스트")
    print("="*50)
    
    # 실제 데이터셋 생성
    congestion_data = pd.read_csv('data/혼잡도.csv')
    node_features = pd.read_csv('data/node_features.csv')
    date_features = pd.read_csv('data/date_features.csv')
    time_features = pd.read_csv('data/time_features.csv')
    adjacency_matrix = np.load('data/adjacency_matrix.npy')
    
    stations = sorted(congestion_data['station'].unique())
    station_to_idx = {station: idx for idx, station in enumerate(stations)}
    
    # 테스트 데이터
    dates = sorted(congestion_data['date'].unique())
    test_start = int(len(dates) * 0.85)
    test_dates = dates[test_start:test_start+2]  # 처음 2일만
    test_data = congestion_data[congestion_data['date'].isin(test_dates)]
    
    test_dataset = OccupancyDataset(
        test_data, node_features, date_features, time_features,
        adjacency_matrix, station_to_idx, 
        sequence_length=12, prediction_length=1,
        target_columns=['occupancy']
    )
    
    print(f"테스트 데이터셋 크기: {len(test_dataset)}")
    
    # 첫 번째 샘플 테스트
    sample = test_dataset[0]
    X_real = sample['X'].unsqueeze(0).to(device)
    Y_real = sample['Y'].unsqueeze(0).to(device)
    
    print(f"실제 입력 X 범위: [{X_real.min().item():.1f}, {X_real.max().item():.1f}]")
    print(f"실제 타겟 Y 범위: [{Y_real.min().item():.1f}, {Y_real.max().item():.1f}]")
    
    adjacency_real = torch.FloatTensor(adjacency_matrix).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        pred_real, _ = model(X_real, adjacency_real, target_length=1)
        pred_real = pred_real.squeeze()
    
    print(f"실제 데이터 예측 범위: [{pred_real.min().item():.1f}, {pred_real.max().item():.1f}]")
    
    # 주요 역별 비교
    major_stations = ['대전역', '탄방역', '용문역']
    print(f"\n주요 역별 예측 vs 실제:")
    for station in major_stations:
        if station in station_to_idx:
            idx = station_to_idx[station]
            pred_val = pred_real[idx].item()
            true_val = Y_real.squeeze()[idx].item()
            print(f"  {station:8s}: 예측 {pred_val:6.1f}명, 실제 {true_val:6.1f}명")

if __name__ == '__main__':
    quick_test()
