#!/usr/bin/env python3
"""
인사이트 2: 출근·퇴근·점심 시간대의 패턴을 자동으로 학습

- Attention이 특정 시간 구간을 높은 가중치로 잡아냄 → 패턴 해석 가능
- 시간대별 혼잡 패턴 분석
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN
from scripts.train_occupancy_model import OccupancyDataset
from scripts.simulate_poi_impact import load_model


def analyze_attention_patterns(
    model: DCRNN,
    test_dataset: OccupancyDataset,
    num_samples: int = 50,
    device: torch.device = torch.device('cpu')
):
    """
    Attention weights를 분석하여 중요한 시간대 파악
    """
    print("="*70)
    print("🔍 Attention 패턴 분석: 중요한 시간대 파악")
    print("="*70)
    
    adjacency_tensor = torch.FloatTensor(test_dataset.adjacency_matrix).to(device)
    
    all_attention_weights = []
    all_hours = []
    all_weekdays = []
    
    print(f"\n📊 {num_samples}개 샘플로 Attention 분석 진행...")
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            sample = test_dataset[i]
            
            X = sample['X'].unsqueeze(0).to(device)
            node_features = sample['node_features'].unsqueeze(0).to(device)
            date_features = sample['date_features'].unsqueeze(0).to(device)
            time_features = sample['time_features'].unsqueeze(0).to(device)
            
            batch_size = X.size(0)
            adjacency = adjacency_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Attention weights 추출
            predictions, attention_weights = model(
                X, adjacency,
                target_length=1,
                teacher_forcing_ratio=0.0,
                node_features=node_features,
                date_features=date_features,
                time_features=time_features
            )
            
            if attention_weights is not None:
                # attention_weights: [1, 1, 12] (batch, target_length, seq_len)
                att_weights = attention_weights.squeeze().cpu().numpy()  # [12]
                all_attention_weights.append(att_weights)
                
                # 메타데이터 수집
                date = sample['metadata']['date']
                hour = sample['metadata']['hour']
                all_hours.append(hour)
                
                date_obj = pd.to_datetime(date)
                all_weekdays.append(date_obj.weekday())
            
            if (i + 1) % 10 == 0:
                print(f"  진행: {i+1}/{num_samples}")
    
    if not all_attention_weights:
        print("⚠️ Attention weights를 추출할 수 없습니다.")
        return
    
    # 시간대별 평균 Attention 분석
    all_attention_weights = np.array(all_attention_weights)  # [num_samples, 12]
    
    # 시간대별로 그룹화
    hour_groups = {
        '출근 (7-9시)': [],
        '오전 (9-12시)': [],
        '점심 (12-14시)': [],
        '오후 (14-17시)': [],
        '퇴근 (17-19시)': [],
        '저녁 (19-22시)': [],
        '심야 (22-24시)': []
    }
    
    for i, hour in enumerate(all_hours):
        if 7 <= hour < 9:
            hour_groups['출근 (7-9시)'].append(all_attention_weights[i])
        elif 9 <= hour < 12:
            hour_groups['오전 (9-12시)'].append(all_attention_weights[i])
        elif 12 <= hour < 14:
            hour_groups['점심 (12-14시)'].append(all_attention_weights[i])
        elif 14 <= hour < 17:
            hour_groups['오후 (14-17시)'].append(all_attention_weights[i])
        elif 17 <= hour < 19:
            hour_groups['퇴근 (17-19시)'].append(all_attention_weights[i])
        elif 19 <= hour < 22:
            hour_groups['저녁 (19-22시)'].append(all_attention_weights[i])
        else:
            hour_groups['심야 (22-24시)'].append(all_attention_weights[i])
    
    print("\n📈 시간대별 평균 Attention 가중치:")
    print("-"*70)
    print(f"{'시간대':<15} {'평균 가중치':<15} {'최대 가중치 위치':<20}")
    print("-"*70)
    
    for time_group, weights_list in hour_groups.items():
        if weights_list:
            weights_array = np.array(weights_list)
            mean_weights = np.mean(weights_array, axis=0)  # [12]
            max_idx = np.argmax(mean_weights)
            max_hour_back = 12 - max_idx - 1  # 몇 시간 전이 가장 중요한가
            
            print(f"{time_group:<15} "
                  f"{np.mean(mean_weights):>13.4f}  "
                  f"{max_hour_back}시간 전 (가중치: {mean_weights[max_idx]:.4f})")
    
    # 시각화
    visualize_attention_patterns(all_attention_weights, all_hours, all_weekdays)
    
    return all_attention_weights, all_hours


def visualize_attention_patterns(
    attention_weights: np.ndarray,
    hours: List[int],
    weekdays: List[int]
):
    """Attention 패턴 시각화"""
    os.makedirs('insights/visualizations', exist_ok=True)
    
    # 1. 시간대별 평균 Attention 히트맵
    plt.figure(figsize=(16, 10))
    
    # 시간대별로 그룹화
    hour_ranges = [
        (0, 6, '심야 (0-6시)'),
        (6, 9, '출근 (6-9시)'),
        (9, 12, '오전 (9-12시)'),
        (12, 14, '점심 (12-14시)'),
        (14, 17, '오후 (14-17시)'),
        (17, 19, '퇴근 (17-19시)'),
        (19, 22, '저녁 (19-22시)'),
        (22, 24, '심야 (22-24시)')
    ]
    
    heatmap_data = []
    hour_labels = []
    
    for start, end, label in hour_ranges:
        mask = np.array([start <= h < end for h in hours])
        if np.sum(mask) > 0:
            group_weights = attention_weights[mask]
            mean_weights = np.mean(group_weights, axis=0)
            heatmap_data.append(mean_weights)
            hour_labels.append(label)
    
    if heatmap_data:
        heatmap_array = np.array(heatmap_data)
        
        plt.subplot(2, 2, 1)
        sns.heatmap(heatmap_array,
                   xticklabels=[f't-{11-i}' for i in range(12)],
                   yticklabels=hour_labels,
                   cmap='YlOrRd', annot=False, fmt='.3f',
                   cbar_kws={'label': 'Attention 가중치'})
        plt.xlabel('과거 시간 (t-11 ~ t-0)')
        plt.ylabel('예측 시간대')
        plt.title('시간대별 Attention 패턴 (평균)')
    
    # 2. 요일별 Attention 패턴
    weekday_names = ['월', '화', '수', '목', '금', '토', '일']
    weekday_data = []
    weekday_labels = []
    
    for wd in range(7):
        mask = np.array([w == wd for w in weekdays])
        if np.sum(mask) > 0:
            group_weights = attention_weights[mask]
            mean_weights = np.mean(group_weights, axis=0)
            weekday_data.append(mean_weights)
            weekday_labels.append(weekday_names[wd])
    
    if weekday_data:
        weekday_array = np.array(weekday_data)
        
        plt.subplot(2, 2, 2)
        sns.heatmap(weekday_array,
                   xticklabels=[f't-{11-i}' for i in range(12)],
                   yticklabels=weekday_labels,
                   cmap='YlOrRd', annot=False, fmt='.3f',
                   cbar_kws={'label': 'Attention 가중치'})
        plt.xlabel('과거 시간 (t-11 ~ t-0)')
        plt.ylabel('요일')
        plt.title('요일별 Attention 패턴 (평균)')
    
    # 3. 전체 평균 Attention 가중치
    plt.subplot(2, 2, 3)
    mean_attention = np.mean(attention_weights, axis=0)
    std_attention = np.std(attention_weights, axis=0)
    
    time_steps = [f't-{11-i}' for i in range(12)]
    plt.plot(time_steps, mean_attention, marker='o', linewidth=2, markersize=8)
    plt.fill_between(time_steps,
                    mean_attention - std_attention,
                    mean_attention + std_attention,
                    alpha=0.3)
    plt.xlabel('과거 시간')
    plt.ylabel('Attention 가중치')
    plt.title('전체 평균 Attention 패턴')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    
    # 4. 최근 시간대에 대한 Attention (출근/퇴근 시간대)
    plt.subplot(2, 2, 4)
    rush_hour_mask = np.array([(7 <= h < 9) or (17 <= h < 19) for h in hours])
    
    if np.sum(rush_hour_mask) > 0:
        rush_weights = attention_weights[rush_hour_mask]
        normal_weights = attention_weights[~rush_hour_mask]
        
        rush_mean = np.mean(rush_weights, axis=0)
        normal_mean = np.mean(normal_weights, axis=0)
        
        x = np.arange(12)
        width = 0.35
        
        plt.bar(x - width/2, rush_mean, width, label='출근/퇴근 시간대', alpha=0.8)
        plt.bar(x + width/2, normal_mean, width, label='일반 시간대', alpha=0.8)
        plt.xlabel('과거 시간 (t-11 ~ t-0)')
        plt.ylabel('Attention 가중치')
        plt.title('출근/퇴근 vs 일반 시간대 Attention 비교')
        plt.xticks(x, [f't-{11-i}' for i in range(12)], rotation=45)
        plt.legend()
        plt.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = 'insights/visualizations/attention_patterns.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 시각화 저장: {output_path}")
    plt.close()


def analyze_hourly_congestion_patterns(
    model: DCRNN,
    test_dataset: OccupancyDataset,
    station_name: str = None,
    device: torch.device = torch.device('cpu')
):
    """
    시간대별 혼잡 패턴 분석
    """
    print("\n" + "="*70)
    print("📊 시간대별 혼잡 패턴 분석")
    print("="*70)
    
    adjacency_tensor = torch.FloatTensor(test_dataset.adjacency_matrix).to(device)
    station_to_idx = test_dataset.station_to_idx
    
    # 시간대별 데이터 수집
    hourly_data = {hour: {'pred': [], 'target': []} for hour in range(24)}
    
    print(f"\n📊 전체 테스트 데이터 분석 중...")
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            
            X = sample['X'].unsqueeze(0).to(device)
            node_features = sample['node_features'].unsqueeze(0).to(device)
            date_features = sample['date_features'].unsqueeze(0).to(device)
            time_features = sample['time_features'].unsqueeze(0).to(device)
            
            batch_size = X.size(0)
            adjacency = adjacency_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            
            predictions, _ = model(
                X, adjacency,
                target_length=1,
                teacher_forcing_ratio=0.0,
                node_features=node_features,
                date_features=date_features,
                time_features=time_features
            )
            pred_raw = test_dataset.inverse_transform(predictions.squeeze(1))
            
            hour = sample['metadata']['hour']
            
            if station_name:
                station_idx = station_to_idx.get(station_name)
                if station_idx is not None:
                    hourly_data[hour]['pred'].append(pred_raw[station_idx, 0].item())
                    if 'Y_raw' in sample:
                        hourly_data[hour]['target'].append(sample['Y_raw'][station_idx, 0].item())
            else:
                # 전체 역 평균
                hourly_data[hour]['pred'].append(pred_raw.mean().item())
                if 'Y_raw' in sample:
                    hourly_data[hour]['target'].append(sample['Y_raw'].mean().item())
    
    # 결과 출력
    print(f"\n📈 시간대별 혼잡 패턴 ({'전체 평균' if not station_name else station_name}):")
    print("-"*70)
    print(f"{'시간':<8} {'평균 예측':<12} {'평균 실제':<12} {'오차':<12}")
    print("-"*70)
    
    for hour in range(24):
        if hourly_data[hour]['pred']:
            mean_pred = np.mean(hourly_data[hour]['pred'])
            if hourly_data[hour]['target']:
                mean_target = np.mean(hourly_data[hour]['target'])
                error = abs(mean_pred - mean_target)
                print(f"{hour:2d}시    "
                      f"{mean_pred:>10.1f}명  "
                      f"{mean_target:>10.1f}명  "
                      f"{error:>10.1f}명")
            else:
                print(f"{hour:2d}시    {mean_pred:>10.1f}명")
    
    # 시각화
    visualize_hourly_patterns(hourly_data, station_name)
    
    return hourly_data


def visualize_hourly_patterns(
    hourly_data: Dict[int, Dict],
    station_name: str = None
):
    """시간대별 패턴 시각화"""
    hours = list(range(24))
    mean_pred = [np.mean(hourly_data[h]['pred']) if hourly_data[h]['pred'] else 0 
                 for h in hours]
    mean_target = [np.mean(hourly_data[h]['target']) if hourly_data[h]['target'] else 0 
                   for h in hours]
    
    plt.figure(figsize=(14, 6))
    plt.plot(hours, mean_pred, marker='o', label='예측값', linewidth=2, markersize=6)
    if any(mean_target):
        plt.plot(hours, mean_target, marker='s', label='실제값', linewidth=2, markersize=6)
    
    # 출근/퇴근 시간대 표시
    plt.axvspan(7, 9, alpha=0.2, color='red', label='출근 시간대')
    plt.axvspan(17, 19, alpha=0.2, color='blue', label='퇴근 시간대')
    plt.axvspan(12, 14, alpha=0.2, color='orange', label='점심 시간대')
    
    plt.xlabel('시간 (시)')
    plt.ylabel('혼잡도 (명)')
    plt.title(f'시간대별 혼잡 패턴 ({station_name if station_name else "전체 평균"})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(range(24))
    
    output_path = f'insights/visualizations/hourly_patterns_{station_name or "all"}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 시각화 저장: {output_path}")
    plt.close()


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="시간대별 패턴 분석")
    parser.add_argument('--samples', type=int, default=50,
                       help='Attention 분석 샘플 수')
    parser.add_argument('--station', type=str, default=None,
                       help='특정 역 분석 (없으면 전체 평균)')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/occupancy_model/best_occupancy_model.pth',
                       help='모델 체크포인트 경로')
    
    args = parser.parse_args()
    
    device = torch.device('cpu')
    
    # 모델 로드
    print(f"📦 모델 로드: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    
    # 데이터 로드
    print("\n📂 데이터 로드 중...")
    congestion_data = pd.read_csv('data/혼잡도.csv')
    node_features = pd.read_csv('data/node_features.csv')
    date_features = pd.read_csv('data/date_features.csv')
    time_features = pd.read_csv('data/time_features.csv')
    adjacency_matrix = np.load('data/adjacency_matrix.npy')
    
    import pickle
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
    
    # Attention 패턴 분석
    attention_weights, hours = analyze_attention_patterns(
        model, test_dataset,
        num_samples=args.samples,
        device=device
    )
    
    # 시간대별 혼잡 패턴 분석
    hourly_data = analyze_hourly_congestion_patterns(
        model, test_dataset,
        station_name=args.station,
        device=device
    )
    
    print("\n✅ 분석 완료!")


if __name__ == '__main__':
    main()

