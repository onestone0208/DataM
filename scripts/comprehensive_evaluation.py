#!/usr/bin/env python3
"""
최종 모델 종합 평가 스크립트

시간대별, 역별, 요일별 성능 분석 포함
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN
from scripts.train_occupancy_model import OccupancyDataset
from torch.utils.data import DataLoader


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[DCRNN, dict]:
    """모델 로드"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    state_dict = checkpoint['model_state_dict']
    
    # 실제 State Dict에서 차원 확인
    encoder_key = 'encoder_layers.0.reset_gate_x.weight_forward'
    actual_input_size = state_dict[encoder_key].shape[1]
    output_key = 'output_projection.weight'
    actual_output_size = state_dict[output_key].shape[0]
    
    model = DCRNN(
        input_size=actual_input_size,
        hidden_size=config['model']['hidden_size'],
        output_size=actual_output_size,
        num_layers=config['model']['num_layers'],
        diffusion_steps=config['model']['diffusion_steps'],
        use_attention=config['model']['use_attention'],
        dropout=config['model']['dropout']
    ).to(device)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, config


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """성능 지표 계산"""
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    pred_valid = predictions[mask]
    target_valid = targets[mask]
    
    if len(pred_valid) == 0:
        return {}
    
    mae = np.mean(np.abs(pred_valid - target_valid))
    rmse = np.sqrt(np.mean((pred_valid - target_valid) ** 2))
    
    nonzero_mask = target_valid > 0
    if np.sum(nonzero_mask) > 0:
        mape = np.mean(np.abs((pred_valid[nonzero_mask] - target_valid[nonzero_mask]) / target_valid[nonzero_mask])) * 100
    else:
        mape = float('nan')
    
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


def comprehensive_evaluation():
    """종합 평가 실행"""
    print("="*70)
    print("🎯 최종 모델 종합 평가")
    print("="*70)
    
    device = torch.device('cpu')
    
    # 모델 로드
    checkpoint_path = "checkpoints/occupancy_model/best_occupancy_model.pth"
    print(f"\n📦 모델 로드: {checkpoint_path}")
    model, config = load_model(checkpoint_path, device)
    print(f"✅ 모델 로드 완료 (Epoch {torch.load(checkpoint_path, map_location=device)['epoch']})")
    
    # 데이터 로드
    print("\n📂 데이터 로드 중...")
    congestion_data = pd.read_csv('data/혼잡도.csv')
    node_features = pd.read_csv('data/node_features.csv')
    date_features = pd.read_csv('data/date_features.csv')
    time_features = pd.read_csv('data/time_features.csv')
    adjacency_matrix = np.load('data/adjacency_matrix.npy')
    
    # 계절별 분할 로드
    with open('data/seasonal_split.pkl', 'rb') as f:
        split_data = pickle.load(f)
    
    stations = sorted(congestion_data['station'].unique())
    station_to_idx = {station: idx for idx, station in enumerate(stations)}
    
    # Train 데이터로 정규화 통계 계산
    train_data = congestion_data[congestion_data['date'].isin(split_data['train_dates'])]
    train_dataset = OccupancyDataset(
        train_data, node_features, date_features, time_features,
        adjacency_matrix, station_to_idx,
        sequence_length=12, prediction_length=1,
        target_columns=['occupancy']
    )
    
    # Test 데이터셋
    test_data = congestion_data[congestion_data['date'].isin(split_data['test_dates'])]
    test_dataset = OccupancyDataset(
        test_data, node_features, date_features, time_features,
        adjacency_matrix, station_to_idx,
        sequence_length=12, prediction_length=1,
        target_columns=['occupancy'],
        normalization_stats=(train_dataset.log_mean, train_dataset.log_std)
    )
    
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    adjacency_tensor = torch.FloatTensor(adjacency_matrix).to(device)
    
    print(f"✅ Test 데이터셋 준비 완료 (샘플 수: {len(test_dataset)})")
    
    # 평가 실행
    print("\n📊 평가 진행 중...")
    
    all_predictions = []
    all_targets = []
    
    # 분석용 데이터 수집
    by_hour = defaultdict(lambda: {'pred': [], 'target': []})
    by_station = defaultdict(lambda: {'pred': [], 'target': []})
    by_weekday = defaultdict(lambda: {'pred': [], 'target': []})
    
    metadata_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            X = batch['X'].to(device)
            Y_raw = batch['Y_raw'].to(device)
            node_features_batch = batch['node_features'].to(device)
            
            batch_size = X.size(0)
            adjacency = adjacency_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            
            predictions, _ = model(
                X, adjacency,
                target_length=1,
                teacher_forcing_ratio=0.0,
                node_features=node_features_batch,
                date_features=None,
                time_features=None
            )
            predictions = predictions.squeeze(1)  # [B, N, 1]
            pred_raw = test_dataset.inverse_transform(predictions)
            
            # 전체 결과 저장
            all_predictions.extend(pred_raw.cpu().numpy().flatten())
            all_targets.extend(Y_raw.cpu().numpy().flatten())
            
            # 메타데이터 수집
            for sample_idx in range(batch_size):
                date = batch['metadata']['date'][sample_idx]
                hour = batch['metadata']['hour'][sample_idx]
                
                # 요일 계산
                date_obj = pd.to_datetime(date)
                weekday = date_obj.weekday()  # 0=월요일, 6=일요일
                
                for station_name, station_idx in station_to_idx.items():
                    pred_val = pred_raw[sample_idx, station_idx].item()
                    target_val = Y_raw[sample_idx, station_idx, 0].item()
                    
                    by_hour[hour]['pred'].append(pred_val)
                    by_hour[hour]['target'].append(target_val)
                    
                    by_station[station_name]['pred'].append(pred_val)
                    by_station[station_name]['target'].append(target_val)
                    
                    by_weekday[weekday]['pred'].append(pred_val)
                    by_weekday[weekday]['target'].append(target_val)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  진행: {batch_idx+1}/{len(test_loader)} 배치")
    
    # 전체 성능
    print("\n" + "="*70)
    print("📊 전체 성능 지표")
    print("="*70)
    overall_metrics = compute_metrics(np.array(all_predictions), np.array(all_targets))
    print(f"MAE:  {overall_metrics['mae']:.2f}명")
    print(f"RMSE: {overall_metrics['rmse']:.2f}명")
    print(f"MAPE: {overall_metrics['mape']:.1f}%")
    print(f"R²:   {overall_metrics['r2']:.3f}")
    print(f"총 샘플 수: {overall_metrics['count']:,}개")
    
    # 시간대별 성능
    print("\n" + "="*70)
    print("⏰ 시간대별 성능 분석")
    print("="*70)
    hour_metrics = []
    for hour in sorted(by_hour.keys()):
        metrics = compute_metrics(
            np.array(by_hour[hour]['pred']),
            np.array(by_hour[hour]['target'])
        )
        if metrics:
            hour_metrics.append((hour, metrics))
            print(f"{hour:2d}시: MAE {metrics['mae']:6.1f}명, RMSE {metrics['rmse']:6.1f}명, "
                  f"R² {metrics['r2']:5.3f} (샘플: {metrics['count']:4d}개)")
    
    # 역별 성능 (상위 10개 역)
    print("\n" + "="*70)
    print("🚇 역별 성능 분석 (상위 10개 역)")
    print("="*70)
    station_metrics = []
    for station_name in sorted(by_station.keys()):
        metrics = compute_metrics(
            np.array(by_station[station_name]['pred']),
            np.array(by_station[station_name]['target'])
        )
        if metrics:
            station_metrics.append((station_name, metrics))
    
    station_metrics.sort(key=lambda x: x[1]['mae'])  # MAE 기준 정렬
    for station_name, metrics in station_metrics[:10]:
        print(f"{station_name:12s}: MAE {metrics['mae']:6.1f}명, RMSE {metrics['rmse']:6.1f}명, "
              f"R² {metrics['r2']:5.3f} (샘플: {metrics['count']:4d}개)")
    
    # 요일별 성능
    print("\n" + "="*70)
    print("📅 요일별 성능 분석")
    print("="*70)
    weekday_names = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    for weekday in sorted(by_weekday.keys()):
        metrics = compute_metrics(
            np.array(by_weekday[weekday]['pred']),
            np.array(by_weekday[weekday]['target'])
        )
        if metrics:
            print(f"{weekday_names[weekday]:4s}: MAE {metrics['mae']:6.1f}명, RMSE {metrics['rmse']:6.1f}명, "
                  f"R² {metrics['r2']:5.3f} (샘플: {metrics['count']:4d}개)")
    
    # 혼잡도 구간별 성능
    print("\n" + "="*70)
    print("📈 혼잡도 구간별 성능 분석")
    print("="*70)
    targets_array = np.array(all_targets)
    predictions_array = np.array(all_predictions)
    
    ranges = [
        (0, 20, "0-20명 (저혼잡)"),
        (20, 50, "20-50명 (중저혼잡)"),
        (50, 100, "50-100명 (중혼잡)"),
        (100, 200, "100-200명 (고혼잡)"),
        (200, float('inf'), "200명 이상 (초고혼잡)")
    ]
    
    for min_val, max_val, label in ranges:
        if max_val == float('inf'):
            mask = targets_array >= min_val
        else:
            mask = (targets_array >= min_val) & (targets_array < max_val)
        
        if np.sum(mask) > 0:
            pred_range = predictions_array[mask]
            target_range = targets_array[mask]
            metrics = compute_metrics(pred_range, target_range)
            if metrics:
                print(f"{label:20s}: MAE {metrics['mae']:6.1f}명, RMSE {metrics['rmse']:6.1f}명, "
                      f"R² {metrics['r2']:5.3f} (샘플: {metrics['count']:5d}개)")
    
    print("\n" + "="*70)
    print("✅ 종합 평가 완료!")
    print("="*70)


if __name__ == '__main__':
    comprehensive_evaluation()

