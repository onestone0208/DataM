#!/usr/bin/env python3
"""
구체적인 예측 결과 확인 - 특정 역, 특정 시간 예측 정확도
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN
from models.dataset import SubwayGraphDataset

def load_latest_model():
    """최신 모델 로드"""
    print("=== 구체적인 예측 결과 확인 ===")
    
    device = torch.device('cpu')
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    
    print(f"✅ 모델 로드 완료 (Epoch {checkpoint['epoch']})")
    
    # 모델 구성
    model_config = checkpoint['config']['model']
    model = DCRNN(
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        output_size=model_config['output_size'],
        num_layers=model_config['num_layers'],
        diffusion_steps=model_config['diffusion_steps'],
        use_attention=model_config['use_attention'],
        dropout=model_config['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device, checkpoint

def create_test_data():
    """테스트 데이터 생성 (훈련과 동일한 방식)"""
    print("\n=== 테스트 데이터 준비 ===")
    
    # 원본 데이터 로드
    congestion_data = pd.read_csv('data/혼잡도.csv')
    node_features = pd.read_csv('data/node_features.csv')
    date_features = pd.read_csv('data/date_features.csv')
    time_features = pd.read_csv('data/time_features.csv')
    adjacency_matrix = np.load('data/adjacency_matrix.npy')
    
    # 역 인덱스 매핑
    stations = sorted(congestion_data['station'].unique())
    station_to_idx = {station: idx for idx, station in enumerate(stations)}
    
    print(f"📍 역 목록: {stations}")
    
    # 테스트용 데이터 (최근 데이터 사용)
    dates = sorted(congestion_data['date'].unique())
    test_dates = dates[-30:]  # 최근 30일
    
    print(f"📅 테스트 기간: {test_dates[0]} ~ {test_dates[-1]}")
    
    test_data = congestion_data[congestion_data['date'].isin(test_dates)]
    
    # 데이터셋 생성
    test_dataset = SubwayGraphDataset(
        test_data, node_features, date_features, time_features,
        adjacency_matrix, station_to_idx, 
        sequence_length=12, prediction_length=1,
        target_columns=['occupancy', 'total_flow']
    )
    
    print(f"✅ 테스트 데이터셋 생성: {len(test_dataset)} 샘플")
    
    return test_dataset, adjacency_matrix, station_to_idx, stations

def predict_specific_cases():
    """구체적인 예측 사례 확인"""
    model, device, checkpoint = load_latest_model()
    test_dataset, adjacency_matrix, station_to_idx, stations = create_test_data()
    
    # 인접행렬 준비
    adjacency = torch.FloatTensor(adjacency_matrix).to(device)
    
    print(f"\n=== 구체적인 예측 결과 (Epoch {checkpoint['epoch']}) ===")
    
    # 몇 개 샘플 선택해서 상세 분석
    sample_indices = [0, 10, 20, 30, 40] if len(test_dataset) > 40 else list(range(min(5, len(test_dataset))))
    
    with torch.no_grad():
        for i, sample_idx in enumerate(sample_indices):
            try:
                sample = test_dataset[sample_idx]
                
                X = sample['X'].unsqueeze(0).to(device)  # [1, 12, 22, 2]
                Y = sample['Y'].unsqueeze(0).to(device)  # [1, 22, 2]
                
                adj_batch = adjacency.unsqueeze(0)
                
                # 예측
                predictions, attention = model(X, adj_batch, target_length=1)
                predictions = predictions.squeeze()  # [22, 2]
                targets = Y.squeeze()  # [22, 2]
                
                # 메타데이터 확인
                metadata = sample.get('metadata', {})
                date = metadata.get('date', 'Unknown')
                hour = metadata.get('hour', 'Unknown')
                
                print(f"\n🎯 샘플 {i+1}: {date} {hour}시 예측")
                print("=" * 60)
                
                # 주요 역들 예측 결과
                major_stations = ['대전역', '갈마', '시청', '정부청사', '탄방']
                
                for station in major_stations:
                    if station in station_to_idx:
                        station_idx = station_to_idx[station]
                        
                        # 실제값
                        true_occupancy = targets[station_idx, 0].item()
                        true_flow = targets[station_idx, 1].item()
                        
                        # 예측값
                        pred_occupancy = predictions[station_idx, 0].item()
                        pred_flow = predictions[station_idx, 1].item()
                        
                        # 오차
                        occ_error = abs(true_occupancy - pred_occupancy)
                        flow_error = abs(true_flow - pred_flow)
                        
                        print(f"📍 {station}역:")
                        print(f"   혼잡도: 실제 {true_occupancy:6.1f}명 | 예측 {pred_occupancy:6.1f}명 | 오차 {occ_error:6.1f}명")
                        print(f"   통행량: 실제 {true_flow:6.1f}명 | 예측 {pred_flow:6.1f}명 | 오차 {flow_error:6.1f}명")
                        
                        if true_occupancy > 0:
                            occ_mape = (occ_error / true_occupancy) * 100
                            print(f"   혼잡도 MAPE: {occ_mape:.1f}%")
                        if true_flow > 0:
                            flow_mape = (flow_error / true_flow) * 100
                            print(f"   통행량 MAPE: {flow_mape:.1f}%")
                        print()
                
                # 전체 통계
                all_occ_true = targets[:, 0].cpu().numpy()
                all_occ_pred = predictions[:, 0].cpu().numpy()
                all_flow_true = targets[:, 1].cpu().numpy()
                all_flow_pred = predictions[:, 1].cpu().numpy()
                
                # 유효한 값만 (0이 아닌 값)
                valid_occ = all_occ_true != 0
                valid_flow = all_flow_true != 0
                
                if valid_occ.sum() > 0:
                    occ_mae = np.mean(np.abs(all_occ_true[valid_occ] - all_occ_pred[valid_occ]))
                    print(f"📊 이 시점 혼잡도 MAE: {occ_mae:.1f}명")
                
                if valid_flow.sum() > 0:
                    flow_mae = np.mean(np.abs(all_flow_true[valid_flow] - all_flow_pred[valid_flow]))
                    print(f"📊 이 시점 통행량 MAE: {flow_mae:.1f}명")
                
                print("-" * 60)
                
            except Exception as e:
                print(f"❌ 샘플 {sample_idx} 처리 중 오류: {e}")
                continue
    
    # 전체 성능 요약
    print(f"\n📈 모델 정보:")
    print(f"   현재 에포크: {checkpoint['epoch']}")
    print(f"   파라미터 수: {sum(p.numel() for p in model.parameters()):,}개")
    print(f"   테스트 샘플: {len(test_dataset)}개")

if __name__ == '__main__':
    try:
        predict_specific_cases()
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
