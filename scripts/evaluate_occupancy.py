#!/usr/bin/env python3
"""
혼잡도 전용 모델 실제 예측 정확도 평가

실제 validation 데이터에서:
- 특정 시간, 날짜, 역의 혼잡도를 얼마나 정확히 예측하는지
- MAE, RMSE, MAPE 등 정량적 지표 계산
- 시간대별, 역별, 요일별 예측 성능 분석
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN
from scripts.train_occupancy_model import OccupancyDataset


def load_occupancy_model(checkpoint_path: str, device: torch.device) -> DCRNN:
    """혼잡도 모델 로드"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = DCRNN(
        input_size=45,  # 🔧 혼잡도(1) + 추가특성(44) = 45차원
        hidden_size=config['model']['hidden_size'],
        output_size=1,  # 혼잡도만
        num_layers=config['model']['num_layers'],
        diffusion_steps=config['model']['diffusion_steps'],
        use_attention=config['model']['use_attention'],
        dropout=config['model']['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ 모델 로드 완료 (Epoch {checkpoint['epoch']})")
    return model, config


def create_test_dataset():
    """테스트 데이터셋 생성"""
    # 원본 데이터 로드
    congestion_data = pd.read_csv('data/혼잡도.csv')
    node_features = pd.read_csv('data/node_features.csv')
    date_features = pd.read_csv('data/date_features.csv')
    time_features = pd.read_csv('data/time_features.csv')
    adjacency_matrix = np.load('data/adjacency_matrix.npy')
    
    # 역 인덱스 매핑
    stations = sorted(congestion_data['station'].unique())
    station_to_idx = {station: idx for idx, station in enumerate(stations)}
    
    # 테스트 데이터 분할 (85% 이후)
    dates = sorted(congestion_data['date'].unique())
    n_dates = len(dates)
    test_start = int(n_dates * 0.85)
    test_dates = dates[test_start:]
    
    test_data = congestion_data[congestion_data['date'].isin(test_dates)]
    
    test_dataset = OccupancyDataset(
        test_data, node_features, date_features, time_features,
        adjacency_matrix, station_to_idx, 
        sequence_length=12, prediction_length=1,
        target_columns=['occupancy']
    )
    
    return test_dataset, adjacency_matrix, station_to_idx


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """예측 정확도 지표 계산"""
    # 유효한 값만 사용 (NaN 제거)
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    pred_valid = predictions[mask]
    target_valid = targets[mask]
    
    if len(pred_valid) == 0:
        return {'mae': float('nan'), 'rmse': float('nan'), 'mape': float('nan'), 'r2': float('nan')}
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(pred_valid - target_valid))
    
    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((pred_valid - target_valid) ** 2))
    
    # MAPE (Mean Absolute Percentage Error)
    # 0이 아닌 값만 사용
    nonzero_mask = target_valid != 0
    if np.sum(nonzero_mask) > 0:
        mape = np.mean(np.abs((pred_valid[nonzero_mask] - target_valid[nonzero_mask]) / target_valid[nonzero_mask])) * 100
    else:
        mape = float('nan')
    
    # R² (Coefficient of Determination)
    ss_res = np.sum((target_valid - pred_valid) ** 2)
    ss_tot = np.sum((target_valid - np.mean(target_valid)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'count': len(pred_valid)
    }


def evaluate_specific_cases(model: DCRNN, test_dataset, adjacency_matrix: np.ndarray, 
                          station_to_idx: Dict[str, int], device: torch.device):
    """구체적인 예측 사례 분석"""
    
    adjacency_tensor = torch.FloatTensor(adjacency_matrix).to(device)
    
    print("\n" + "="*60)
    print("🎯 구체적인 예측 사례 분석")
    print("="*60)
    
    # 주요 역들
    major_stations = ['대전역', '정부청사역', '시청역', '탄방역', '용문역']
    
    all_predictions = []
    all_targets = []
    station_results = {}
    
    with torch.no_grad():
        for i in range(min(100, len(test_dataset))):  # 처음 100개 샘플만
            try:
                sample = test_dataset[i]
                X = sample['X'].unsqueeze(0).to(device)  # [1, T, N, 1]
                Y = sample['Y'].unsqueeze(0).to(device)  # [1, N, 1]
                
                # 인접행렬 확장
                adjacency = adjacency_tensor.unsqueeze(0)  # [1, N, N]
                
                # 추가 특성들 추출
                node_features = sample['node_features'].unsqueeze(0).to(device)
                date_features = sample['date_features'].unsqueeze(0).to(device)
                time_features = sample['time_features'].unsqueeze(0).to(device)
                
                # 예측
                predictions, _ = model(
                    X, adjacency, 
                    target_length=1,
                    node_features=node_features,
                    date_features=date_features,
                    time_features=time_features
                )
                predictions = predictions.squeeze()  # [N, 1] -> [N]
                targets = Y.squeeze()  # [N, 1] -> [N]
                
                # 각 역별로 결과 저장
                for station, idx in station_to_idx.items():
                    pred_val = predictions[idx].cpu().numpy()
                    true_val = targets[idx].cpu().numpy()
                    
                    if station not in station_results:
                        station_results[station] = {'predictions': [], 'targets': []}
                    
                    station_results[station]['predictions'].append(pred_val)
                    station_results[station]['targets'].append(true_val)
                
                # 전체 결과에 추가
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # 처음 몇 개 샘플의 주요 역 결과 출력
                if i < 5:
                    print(f"\n📍 샘플 {i+1}:")
                    for station in major_stations:
                        if station in station_to_idx:
                            idx = station_to_idx[station]
                            pred = predictions[idx].item()
                            true = targets[idx].item()
                            error = abs(pred - true)
                            error_pct = (error / true * 100) if true != 0 else float('inf')
                            
                            print(f"  {station:8s}: 실제 {true:6.1f}명 | 예측 {pred:6.1f}명 | 오차 {error:6.1f}명 ({error_pct:5.1f}%)")
                            
            except Exception as e:
                print(f"샘플 {i} 처리 중 오류: {e}")
                continue
    
    # 전체 성능 지표
    print(f"\n📊 전체 성능 지표 (샘플 수: {len(all_predictions)})")
    overall_metrics = compute_metrics(np.array(all_predictions), np.array(all_targets))
    print(f"MAE:  {overall_metrics['mae']:.2f}명")
    print(f"RMSE: {overall_metrics['rmse']:.2f}명") 
    print(f"MAPE: {overall_metrics['mape']:.1f}%")
    print(f"R²:   {overall_metrics['r2']:.3f}")
    
    # 역별 성능 분석
    print(f"\n🚇 주요 역별 성능 분석")
    print("-" * 60)
    for station in major_stations:
        if station in station_results and len(station_results[station]['predictions']) > 0:
            preds = np.array(station_results[station]['predictions'])
            targets = np.array(station_results[station]['targets'])
            metrics = compute_metrics(preds, targets)
            
            print(f"{station:10s}: MAE {metrics['mae']:5.1f}명, RMSE {metrics['rmse']:5.1f}명, "
                  f"MAPE {metrics['mape']:5.1f}%, R² {metrics['r2']:5.3f} (샘플: {metrics['count']})")
    
    return overall_metrics, station_results


def main():
    print("🔍 혼잡도 전용 모델 실제 예측 정확도 평가")
    print("=" * 60)
    
    # 디바이스 설정
    device = torch.device('cpu')  # CPU 사용
    
    # 모델 로드
    checkpoint_path = "checkpoints/occupancy_model/best_occupancy_model.pth"
    if not Path(checkpoint_path).exists():
        print(f"❌ 체크포인트 파일이 없습니다: {checkpoint_path}")
        return
    
    model, config = load_occupancy_model(checkpoint_path, device)
    
    # 테스트 데이터 생성
    print("📂 테스트 데이터셋 생성 중...")
    test_dataset, adjacency_matrix, station_to_idx = create_test_dataset()
    print(f"✅ 테스트 데이터셋 준비 완료 (샘플 수: {len(test_dataset)})")
    
    # 구체적인 예측 사례 분석
    overall_metrics, station_results = evaluate_specific_cases(
        model, test_dataset, adjacency_matrix, station_to_idx, device
    )
    
    print(f"\n🎯 결론:")
    print(f"현재 혼잡도 전용 모델의 예측 정확도는 MAE {overall_metrics['mae']:.1f}명입니다.")
    if overall_metrics['mae'] < 20:
        print("✅ 매우 좋은 성능입니다!")
    elif overall_metrics['mae'] < 50:
        print("⚠️  괜찮은 성능이지만 개선 여지가 있습니다.")
    else:
        print("❌ 성능 개선이 필요합니다.")


if __name__ == '__main__':
    main()
