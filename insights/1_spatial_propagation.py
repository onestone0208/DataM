#!/usr/bin/env python3
"""
인사이트 1: 특정 역의 혼잡 변화가 인접 역에 어떻게 영향을 주는지 파악

- 환승역 중심으로 혼잡이 전파되는 패턴 분석
- 특정 구간 폐쇄/지연 시 영향도 시뮬레이션
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

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN
from scripts.train_occupancy_model import OccupancyDataset
from scripts.simulate_poi_impact import load_model, load_node_features, convert_node_features_to_tensor


def analyze_spatial_propagation(
    model: DCRNN,
    test_dataset: OccupancyDataset,
    source_station: str,
    impact_value: float = 50.0,  # 원본 혼잡도에 추가할 값
    num_samples: int = 10,
    device: torch.device = torch.device('cpu')
):
    """
    특정 역의 혼잡도 변화가 다른 역들에 미치는 영향 분석
    
    Args:
        model: 학습된 모델
        test_dataset: 테스트 데이터셋
        source_station: 혼잡도 변화를 시뮬레이션할 역
        impact_value: 추가할 혼잡도 값
        num_samples: 테스트할 샘플 수
        device: 디바이스
    """
    print("="*70)
    print(f"🔍 공간 전파 분석: {source_station}의 혼잡도 변화 영향")
    print("="*70)
    
    station_to_idx = test_dataset.station_to_idx
    if source_station not in station_to_idx:
        raise ValueError(f"역 '{source_station}'을 찾을 수 없습니다.")
    
    source_idx = station_to_idx[source_station]
    adjacency_matrix = test_dataset.adjacency_matrix
    adjacency_tensor = torch.FloatTensor(adjacency_matrix).to(device)
    
    # 인접 역 찾기 (거리 기반)
    station_names = sorted(station_to_idx.keys(), key=lambda x: station_to_idx[x])
    
    # 결과 저장
    impact_results = {station: [] for station in station_names}
    
    print(f"\n📊 {num_samples}개 샘플로 분석 진행...")
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            sample = test_dataset[i]
            
            # 원본 예측
            X_original = sample['X'].unsqueeze(0).to(device)
            node_features = sample['node_features'].unsqueeze(0).to(device)
            date_features = sample['date_features'].unsqueeze(0).to(device)
            time_features = sample['time_features'].unsqueeze(0).to(device)
            
            batch_size = X_original.size(0)
            adjacency = adjacency_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            
            pred_original, _ = model(
                X_original, adjacency,
                target_length=1,
                node_features=node_features,
                date_features=date_features,
                time_features=time_features
            )
            # 모델 출력을 [N, 1] 형태로 변환
            # 가능한 형태: [1, 1, N, 1], [1, N, 1], [N, 1]
            while pred_original.dim() > 2:
                pred_original = pred_original.squeeze(0)
            if pred_original.dim() == 1:
                pred_original = pred_original.unsqueeze(-1)  # [N] -> [N, 1]
            
            # 원본 혼잡도 역변환 후 [N] 형태로 변환
            pred_orig_raw = test_dataset.inverse_transform(pred_original)
            # 모든 크기 1인 차원 제거하여 [N] 형태로
            while pred_orig_raw.dim() > 1:
                pred_orig_raw = pred_orig_raw.squeeze(-1)
            
            # 소스 역의 혼잡도 증가 시뮬레이션
            X_modified = X_original.clone()
            # 마지막 시점의 소스 역 혼잡도 증가
            X_modified[0, -1, source_idx, 0] += impact_value / test_dataset.log_std.item()
            
            pred_modified, _ = model(
                X_modified, adjacency,
                target_length=1,
                node_features=node_features,
                date_features=date_features,
                time_features=time_features
            )
            # 모델 출력을 [N, 1] 형태로 변환
            # 가능한 형태: [1, 1, N, 1], [1, N, 1], [N, 1]
            while pred_modified.dim() > 2:
                pred_modified = pred_modified.squeeze(0)
            if pred_modified.dim() == 1:
                pred_modified = pred_modified.unsqueeze(-1)  # [N] -> [N, 1]
            
            # 역변환 후 [N] 형태로 변환
            pred_mod_raw = test_dataset.inverse_transform(pred_modified)
            # 모든 크기 1인 차원 제거하여 [N] 형태로
            while pred_mod_raw.dim() > 1:
                pred_mod_raw = pred_mod_raw.squeeze(-1)
            
            # 각 역별 영향도 계산
            for station_name, station_idx in station_to_idx.items():
                # numpy 배열로 변환하여 안전하게 접근
                if isinstance(pred_orig_raw, torch.Tensor):
                    orig_val = pred_orig_raw[station_idx].item()
                    mod_val = pred_mod_raw[station_idx].item()
                else:
                    orig_val = float(pred_orig_raw[station_idx])
                    mod_val = float(pred_mod_raw[station_idx])
                impact = mod_val - orig_val
                impact_results[station_name].append(impact)
            
            if (i + 1) % 5 == 0:
                print(f"  진행: {i+1}/{num_samples}")
    
    # 통계 계산
    print(f"\n📈 영향도 분석 결과 ({source_station}에 {impact_value}명 추가 시):")
    print("-"*70)
    
    impact_stats = {}
    for station_name in station_names:
        impacts = np.array(impact_results[station_name])
        impact_stats[station_name] = {
            'mean': np.mean(impacts),
            'std': np.std(impacts),
            'max': np.max(impacts),
            'min': np.min(impacts)
        }
    
    # 영향도가 큰 역 순으로 정렬
    sorted_stations = sorted(
        impact_stats.items(),
        key=lambda x: abs(x[1]['mean']),
        reverse=True
    )
    
    print(f"{'역명':<15} {'평균 영향':<12} {'표준편차':<12} {'최대':<12} {'최소':<12}")
    print("-"*70)
    for station_name, stats in sorted_stations[:15]:  # 상위 15개
        print(f"{station_name:<15} "
              f"{stats['mean']:>+10.2f}명  "
              f"{stats['std']:>10.2f}명  "
              f"{stats['max']:>+10.2f}명  "
              f"{stats['min']:>+10.2f}명")
    
    # 시각화
    visualize_spatial_impact(source_station, impact_stats, adjacency_matrix, station_to_idx)
    
    return impact_stats


def visualize_spatial_impact(
    source_station: str,
    impact_stats: Dict[str, Dict],
    adjacency_matrix: np.ndarray,
    station_to_idx: Dict[str, int]
):
    """공간 전파 영향도 시각화"""
    os.makedirs('insights/visualizations', exist_ok=True)
    
    # 히트맵 생성
    station_names = sorted(station_to_idx.keys(), key=lambda x: station_to_idx[x])
    impact_values = [impact_stats[station]['mean'] for station in station_names]
    
    plt.figure(figsize=(14, 8))
    
    # 서브플롯 1: 막대 그래프
    plt.subplot(1, 2, 1)
    colors = ['red' if v > 0 else 'blue' for v in impact_values]
    plt.barh(range(len(station_names)), impact_values, color=colors, alpha=0.7)
    plt.yticks(range(len(station_names)), station_names)
    plt.xlabel('평균 영향도 (명)')
    plt.title(f'{source_station} 혼잡도 증가 시 다른 역들의 영향도')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    plt.grid(axis='x', alpha=0.3)
    
    # 서브플롯 2: 인접행렬 기반 영향도 매트릭스
    plt.subplot(1, 2, 2)
    impact_matrix = np.zeros((len(station_names), len(station_names)))
    source_idx = station_to_idx[source_station]
    
    for i, station in enumerate(station_names):
        impact_matrix[source_idx, i] = impact_stats[station]['mean']
    
    sns.heatmap(impact_matrix, 
                xticklabels=station_names,
                yticklabels=station_names,
                cmap='RdBu_r', center=0,
                annot=False, fmt='.1f',
                cbar_kws={'label': '영향도 (명)'})
    plt.title(f'{source_station} → 다른 역들의 영향도')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    output_path = f'insights/visualizations/spatial_propagation_{source_station}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 시각화 저장: {output_path}")
    plt.close()


def simulate_station_closure(
    model: DCRNN,
    test_dataset: OccupancyDataset,
    closed_station: str,
    num_samples: int = 10,
    device: torch.device = torch.device('cpu')
):
    """
    특정 역 폐쇄 시뮬레이션
    
    폐쇄된 역의 혼잡도를 0으로 설정하고 다른 역들의 변화를 관찰
    """
    print("\n" + "="*70)
    print(f"🚧 역 폐쇄 시뮬레이션: {closed_station}")
    print("="*70)
    
    station_to_idx = test_dataset.station_to_idx
    if closed_station not in station_to_idx:
        raise ValueError(f"역 '{closed_station}'을 찾을 수 없습니다.")
    
    closed_idx = station_to_idx[closed_station]
    adjacency_tensor = torch.FloatTensor(test_dataset.adjacency_matrix).to(device)
    
    station_names = sorted(station_to_idx.keys(), key=lambda x: station_to_idx[x])
    closure_impact = {station: [] for station in station_names}
    
    print(f"\n📊 {num_samples}개 샘플로 시뮬레이션 진행...")
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            sample = test_dataset[i]
            
            # 원본 예측
            X_original = sample['X'].unsqueeze(0).to(device)
            node_features = sample['node_features'].unsqueeze(0).to(device)
            date_features = sample['date_features'].unsqueeze(0).to(device)
            time_features = sample['time_features'].unsqueeze(0).to(device)
            
            batch_size = X_original.size(0)
            adjacency = adjacency_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            
            pred_original, _ = model(
                X_original, adjacency,
                target_length=1,
                node_features=node_features,
                date_features=date_features,
                time_features=time_features
            )
            # 모델 출력을 [N, 1] 형태로 변환
            # 가능한 형태: [1, 1, N, 1], [1, N, 1], [N, 1]
            while pred_original.dim() > 2:
                pred_original = pred_original.squeeze(0)
            if pred_original.dim() == 1:
                pred_original = pred_original.unsqueeze(-1)  # [N] -> [N, 1]
            
            # 역변환 후 [N] 형태로 변환
            pred_orig_raw = test_dataset.inverse_transform(pred_original)
            # 모든 크기 1인 차원 제거하여 [N] 형태로
            while pred_orig_raw.dim() > 1:
                pred_orig_raw = pred_orig_raw.squeeze(-1)
            
            # 폐쇄 시뮬레이션: 폐쇄된 역의 혼잡도를 0으로
            X_closed = X_original.clone()
            X_closed[0, :, closed_idx, 0] = 0.0  # 모든 시점에서 0으로
            
            pred_closed, _ = model(
                X_closed, adjacency,
                target_length=1,
                node_features=node_features,
                date_features=date_features,
                time_features=time_features
            )
            # 모델 출력을 [N, 1] 형태로 변환
            # 가능한 형태: [1, 1, N, 1], [1, N, 1], [N, 1]
            while pred_closed.dim() > 2:
                pred_closed = pred_closed.squeeze(0)
            if pred_closed.dim() == 1:
                pred_closed = pred_closed.unsqueeze(-1)  # [N] -> [N, 1]
            
            # 역변환 후 [N] 형태로 변환
            pred_closed_raw = test_dataset.inverse_transform(pred_closed)
            # 모든 크기 1인 차원 제거하여 [N] 형태로
            while pred_closed_raw.dim() > 1:
                pred_closed_raw = pred_closed_raw.squeeze(-1)
            
            # 영향도 계산
            for station_name, station_idx in station_to_idx.items():
                if station_idx != closed_idx:  # 폐쇄된 역 제외
                    if isinstance(pred_orig_raw, torch.Tensor):
                        orig_val = pred_orig_raw[station_idx].item()
                        closed_val = pred_closed_raw[station_idx].item()
                    else:
                        orig_val = float(pred_orig_raw[station_idx])
                        closed_val = float(pred_closed_raw[station_idx])
                    impact = closed_val - orig_val
                    closure_impact[station_name].append(impact)
    
    # 결과 출력
    print(f"\n📈 {closed_station} 폐쇄 시 다른 역들의 영향도:")
    print("-"*70)
    
    impact_stats = {}
    for station_name in station_names:
        if station_name != closed_station and closure_impact[station_name]:
            impacts = np.array(closure_impact[station_name])
            impact_stats[station_name] = {
                'mean': np.mean(impacts),
                'std': np.std(impacts)
            }
    
    sorted_impact = sorted(
        impact_stats.items(),
        key=lambda x: abs(x[1]['mean']),
        reverse=True
    )
    
    print(f"{'역명':<15} {'평균 영향':<12} {'표준편차':<12}")
    print("-"*70)
    for station_name, stats in sorted_impact[:10]:
        print(f"{station_name:<15} "
              f"{stats['mean']:>+10.2f}명  "
              f"{stats['std']:>10.2f}명")
    
    return impact_stats


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="공간 전파 영향도 분석")
    parser.add_argument('--station', type=str, default='정부청사역',
                       help='분석할 역 이름')
    parser.add_argument('--impact', type=float, default=50.0,
                       help='추가할 혼잡도 값')
    parser.add_argument('--samples', type=int, default=20,
                       help='테스트할 샘플 수')
    parser.add_argument('--closure', action='store_true',
                       help='역 폐쇄 시뮬레이션 실행')
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
    
    # 공간 전파 분석
    impact_stats = analyze_spatial_propagation(
        model, test_dataset, args.station,
        impact_value=args.impact,
        num_samples=args.samples,
        device=device
    )
    
    # 역 폐쇄 시뮬레이션 (옵션)
    if args.closure:
        closure_stats = simulate_station_closure(
            model, test_dataset, args.station,
            num_samples=args.samples,
            device=device
        )
    
    print("\n✅ 분석 완료!")


if __name__ == '__main__':
    main()

