#!/usr/bin/env python3
"""
인사이트 3: 노드 특성(역세권 인구, 환승 수, 상권 규모 등)이 혼잡도에 미치는 영향

- 강남역 = 상권·유동 인구 반영으로 높은 baseline
- 정부청사 = 출근 시간 피크 강함
→ 정책적 인사이트 제공
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
from scipy import stats

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN
from scripts.train_occupancy_model import OccupancyDataset
from scripts.simulate_poi_impact import load_model, load_node_features, convert_node_features_to_tensor


def analyze_node_feature_importance(
    model: DCRNN,
    test_dataset: OccupancyDataset,
    num_samples: int = 50,
    device: torch.device = torch.device('cpu')
):
    """
    노드 특성별 중요도 분석 (Permutation Importance)
    """
    print("="*70)
    print("🔍 노드 특성 중요도 분석")
    print("="*70)
    
    # 노드 특성 이름
    feature_names = [
        '관공서_밀도', '교육_밀도', '교통_밀도', '금융_밀도', '상업_밀도',
        '숙박_밀도', '의료_밀도', '종교_밀도', '주거_밀도', '체육문화_밀도',
        '총_POI_수', 'POI_다양성', '위도', '경도', '노선순서', '연면적'
    ]
    
    adjacency_tensor = torch.FloatTensor(test_dataset.adjacency_matrix).to(device)
    
    # 기준 성능 계산
    print(f"\n📊 기준 성능 계산 중 ({num_samples}개 샘플)...")
    baseline_errors = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
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
                node_features=node_features,
                date_features=date_features,
                time_features=time_features
            )
            pred_raw = test_dataset.inverse_transform(predictions.squeeze(1))
            
            if 'Y_raw' in sample:
                errors = torch.abs(pred_raw - sample['Y_raw'].to(device))
                baseline_errors.append(errors.cpu().numpy().flatten())
    
    baseline_mae = np.mean(np.concatenate(baseline_errors))
    
    # 각 특성별 중요도 계산
    print(f"\n📈 특성별 중요도 계산 중...")
    feature_importance = {}
    
    for feat_idx, feat_name in enumerate(feature_names):
        print(f"  분석 중: {feat_name} ({feat_idx+1}/{len(feature_names)})")
        
        permuted_errors = []
        
        with torch.no_grad():
            for i in range(min(num_samples, len(test_dataset))):
                sample = test_dataset[i]
                
                X = sample['X'].unsqueeze(0).to(device)
                node_features = sample['node_features'].clone().unsqueeze(0).to(device)
                
                # 특정 특성 셔플 (permutation)
                node_features[0, :, feat_idx] = node_features[0, :, feat_idx][torch.randperm(node_features.size(1))]
                
                date_features = sample['date_features'].unsqueeze(0).to(device)
                time_features = sample['time_features'].unsqueeze(0).to(device)
                
                batch_size = X.size(0)
                adjacency = adjacency_tensor.unsqueeze(0).expand(batch_size, -1, -1)
                
                predictions, _ = model(
                    X, adjacency,
                    target_length=1,
                    node_features=node_features,
                    date_features=date_features,
                    time_features=time_features
                )
                pred_raw = test_dataset.inverse_transform(predictions.squeeze(1))
                
                if 'Y_raw' in sample:
                    errors = torch.abs(pred_raw - sample['Y_raw'].to(device))
                    permuted_errors.append(errors.cpu().numpy().flatten())
        
        permuted_mae = np.mean(np.concatenate(permuted_errors))
        importance = permuted_mae - baseline_mae  # 증가량이 클수록 중요
        
        feature_importance[feat_name] = importance
    
    # 결과 출력
    print(f"\n📊 노드 특성 중요도 (Permutation Importance):")
    print("-"*70)
    print(f"{'특성명':<20} {'중요도 (MAE 증가)':<20} {'순위':<10}")
    print("-"*70)
    
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    for rank, (feat_name, importance) in enumerate(sorted_features, 1):
        print(f"{feat_name:<20} {importance:>+18.4f}명  {rank:>8}위")
    
    # 시각화
    visualize_feature_importance(feature_importance)
    
    return feature_importance


def analyze_station_characteristics(
    model: DCRNN,
    test_dataset: OccupancyDataset,
    station_name: str,
    device: torch.device = torch.device('cpu')
):
    """
    특정 역의 특성과 혼잡도 패턴 분석
    """
    print("\n" + "="*70)
    print(f"🏢 역 특성 분석: {station_name}")
    print("="*70)
    
    station_to_idx = test_dataset.station_to_idx
    if station_name not in station_to_idx:
        raise ValueError(f"역 '{station_name}'을 찾을 수 없습니다.")
    
    station_idx = station_to_idx[station_name]
    
    # 노드 특성 로드
    node_features_df = load_node_features('data/node_features.csv')
    station_features = node_features_df.loc[station_name]
    
    print(f"\n📊 {station_name}의 노드 특성:")
    print("-"*70)
    
    poi_features = [col for col in node_features_df.columns if '_밀도' in col]
    for feat in poi_features:
        if feat in station_features:
            print(f"  {feat:<20}: {station_features[feat]:.4f}")
    
    # 시간대별 혼잡 패턴 분석
    adjacency_tensor = torch.FloatTensor(test_dataset.adjacency_matrix).to(device)
    hourly_congestion = {hour: [] for hour in range(24)}
    
    print(f"\n📈 시간대별 혼잡 패턴 분석 중...")
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            hour = sample['metadata']['hour']
            
            X = sample['X'].unsqueeze(0).to(device)
            node_features = sample['node_features'].unsqueeze(0).to(device)
            date_features = sample['date_features'].unsqueeze(0).to(device)
            time_features = sample['time_features'].unsqueeze(0).to(device)
            
            batch_size = X.size(0)
            adjacency = adjacency_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            
            predictions, _ = model(
                X, adjacency,
                target_length=1,
                node_features=node_features,
                date_features=date_features,
                time_features=time_features
            )
            pred_raw = test_dataset.inverse_transform(predictions.squeeze(1))
            
            hourly_congestion[hour].append(pred_raw[station_idx, 0].item())
    
    # 시간대별 평균 혼잡도
    print(f"\n📊 {station_name} 시간대별 평균 혼잡도:")
    print("-"*70)
    print(f"{'시간':<8} {'평균 혼잡도':<15} {'특징':<30}")
    print("-"*70)
    
    for hour in range(24):
        if hourly_congestion[hour]:
            mean_occ = np.mean(hourly_congestion[hour])
            
            # 특징 설명
            if 7 <= hour < 9:
                feature = "출근 시간대"
            elif 17 <= hour < 19:
                feature = "퇴근 시간대"
            elif 12 <= hour < 14:
                feature = "점심 시간대"
            else:
                feature = "-"
            
            print(f"{hour:2d}시    {mean_occ:>12.1f}명  {feature:<30}")
    
    # 시각화
    visualize_station_characteristics(station_name, station_features, hourly_congestion)
    
    return station_features, hourly_congestion


def visualize_feature_importance(feature_importance: Dict[str, float]):
    """특성 중요도 시각화"""
    os.makedirs('insights/visualizations', exist_ok=True)
    
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    features = [f[0] for f in sorted_features]
    importances = [f[1] for f in sorted_features]
    colors = ['red' if imp > 0 else 'blue' for imp in importances]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(features)), importances, color=colors, alpha=0.7)
    plt.yticks(range(len(features)), features)
    plt.xlabel('중요도 (MAE 증가량, 명)')
    plt.title('노드 특성 중요도 (Permutation Importance)')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = 'insights/visualizations/node_feature_importance.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 시각화 저장: {output_path}")
    plt.close()


def visualize_station_characteristics(
    station_name: str,
    station_features: pd.Series,
    hourly_congestion: Dict[int, List[float]]
):
    """역 특성 시각화"""
    os.makedirs('insights/visualizations', exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. POI 밀도 비교
    poi_features = [col for col in station_features.index if '_밀도' in col]
    poi_values = [station_features[feat] for feat in poi_features]
    
    axes[0].barh(range(len(poi_features)), poi_values, alpha=0.7)
    axes[0].set_yticks(range(len(poi_features)))
    axes[0].set_yticklabels(poi_features)
    axes[0].set_xlabel('밀도')
    axes[0].set_title(f'{station_name} POI 밀도 분포')
    axes[0].grid(axis='x', alpha=0.3)
    
    # 2. 시간대별 혼잡 패턴
    hours = list(range(24))
    mean_occ = [np.mean(hourly_congestion[h]) if hourly_congestion[h] else 0 
                for h in hours]
    
    axes[1].plot(hours, mean_occ, marker='o', linewidth=2, markersize=6)
    axes[1].axvspan(7, 9, alpha=0.2, color='red', label='출근 시간대')
    axes[1].axvspan(17, 19, alpha=0.2, color='blue', label='퇴근 시간대')
    axes[1].axvspan(12, 14, alpha=0.2, color='orange', label='점심 시간대')
    axes[1].set_xlabel('시간 (시)')
    axes[1].set_ylabel('평균 혼잡도 (명)')
    axes[1].set_title(f'{station_name} 시간대별 혼잡 패턴')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(range(24))
    
    plt.tight_layout()
    output_path = f'insights/visualizations/station_characteristics_{station_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 시각화 저장: {output_path}")
    plt.close()


def compare_stations_by_features(
    model: DCRNN,
    test_dataset: OccupancyDataset,
    feature_name: str,
    top_n: int = 5,
    device: torch.device = torch.device('cpu')
):
    """
    특정 특성 값이 높은 역과 낮은 역 비교
    """
    print("\n" + "="*70)
    print(f"🔍 특성 기반 역 비교: {feature_name}")
    print("="*70)
    
    node_features_df = load_node_features('data/node_features.csv')
    
    if feature_name not in node_features_df.columns:
        raise ValueError(f"특성 '{feature_name}'을 찾을 수 없습니다.")
    
    # 특성 값 기준으로 역 정렬
    sorted_stations = node_features_df.sort_values(feature_name, ascending=False)
    
    top_stations = sorted_stations.head(top_n).index.tolist()
    bottom_stations = sorted_stations.tail(top_n).index.tolist()
    
    print(f"\n📊 {feature_name} 높은 역 (상위 {top_n}개):")
    for station in top_stations:
        print(f"  {station}: {sorted_stations.loc[station, feature_name]:.4f}")
    
    print(f"\n📊 {feature_name} 낮은 역 (하위 {top_n}개):")
    for station in bottom_stations:
        print(f"  {station}: {sorted_stations.loc[station, feature_name]:.4f}")
    
    # 시간대별 혼잡도 비교
    adjacency_tensor = torch.FloatTensor(test_dataset.adjacency_matrix).to(device)
    station_to_idx = test_dataset.station_to_idx
    
    top_hourly = {hour: [] for hour in range(24)}
    bottom_hourly = {hour: [] for hour in range(24)}
    
    print(f"\n📈 시간대별 혼잡도 비교 분석 중...")
    
    with torch.no_grad():
        for i in range(min(100, len(test_dataset))):  # 샘플링
            sample = test_dataset[i]
            hour = sample['metadata']['hour']
            
            X = sample['X'].unsqueeze(0).to(device)
            node_features = sample['node_features'].unsqueeze(0).to(device)
            date_features = sample['date_features'].unsqueeze(0).to(device)
            time_features = sample['time_features'].unsqueeze(0).to(device)
            
            batch_size = X.size(0)
            adjacency = adjacency_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            
            predictions, _ = model(
                X, adjacency,
                target_length=1,
                node_features=node_features,
                date_features=date_features,
                time_features=time_features
            )
            pred_raw = test_dataset.inverse_transform(predictions.squeeze(1))
            
            # 상위 역 평균
            top_avg = np.mean([pred_raw[station_to_idx[s], 0].item() 
                              for s in top_stations if s in station_to_idx])
            top_hourly[hour].append(top_avg)
            
            # 하위 역 평균
            bottom_avg = np.mean([pred_raw[station_to_idx[s], 0].item() 
                                 for s in bottom_stations if s in station_to_idx])
            bottom_hourly[hour].append(bottom_avg)
    
    # 시각화
    hours = list(range(24))
    top_mean = [np.mean(top_hourly[h]) if top_hourly[h] else 0 for h in hours]
    bottom_mean = [np.mean(bottom_hourly[h]) if bottom_hourly[h] else 0 for h in hours]
    
    plt.figure(figsize=(14, 6))
    plt.plot(hours, top_mean, marker='o', label=f'{feature_name} 높은 역 (평균)', linewidth=2)
    plt.plot(hours, bottom_mean, marker='s', label=f'{feature_name} 낮은 역 (평균)', linewidth=2)
    plt.axvspan(7, 9, alpha=0.2, color='red')
    plt.axvspan(17, 19, alpha=0.2, color='blue')
    plt.xlabel('시간 (시)')
    plt.ylabel('평균 혼잡도 (명)')
    plt.title(f'{feature_name} 기반 역 비교')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(range(24))
    
    output_path = f'insights/visualizations/feature_comparison_{feature_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 시각화 저장: {output_path}")
    plt.close()


def analyze_gradual_feature_change(
    model: DCRNN,
    test_dataset: OccupancyDataset,
    station_name: str,
    feature_name: str,
    start_value: float = 0.0,
    end_value: float = 1.0,
    num_steps: int = 20,
    num_samples: int = 20,
    device: torch.device = torch.device('cpu')
):
    """
    특정 역의 POI 특성을 점진적으로 변화시켜 혼잡도 변화를 시각화
    
    예: 정부청사역의 관공서 밀도를 0.0 → 1.0으로 점진적으로 높여서 혼잡도 변화 분석
    
    Args:
        station_name: 분석할 역 이름
        feature_name: 변화시킬 특성 이름 (예: '관공서_밀도')
        start_value: 시작 값
        end_value: 끝 값
        num_steps: 단계 수
        num_samples: 테스트할 샘플 수
    """
    print("="*70)
    print(f"📊 점진적 특성 변화 분석: {station_name}의 {feature_name}")
    print(f"   {start_value} → {end_value} ({num_steps}단계)")
    print("="*70)
    
    station_to_idx = test_dataset.station_to_idx
    if station_name not in station_to_idx:
        raise ValueError(f"역 '{station_name}'을 찾을 수 없습니다.")
    
    station_idx = station_to_idx[station_name]
    
    # 노드 특성 로드
    node_features_df = load_node_features('data/node_features.csv')
    
    # 원본 특성 값 확인
    original_value = node_features_df.loc[station_name, feature_name]
    print(f"\n📌 {station_name}의 현재 {feature_name}: {original_value:.4f}")
    
    # 특성 이름 리스트
    feature_names = [
        '관공서_밀도', '교육_밀도', '교통_밀도', '금융_밀도', '상업_밀도',
        '숙박_밀도', '의료_밀도', '종교_밀도', '주거_밀도', '체육문화_밀도',
        '총_POI_수', 'POI_다양성', '위도', '경도', '노선순서', '연면적'
    ]
    
    adjacency_tensor = torch.FloatTensor(test_dataset.adjacency_matrix).to(device)
    
    # 점진적 값 생성
    feature_values = np.linspace(start_value, end_value, num_steps)
    congestion_changes = []
    congestion_means = []
    congestion_stds = []
    
    print(f"\n📊 {num_steps}단계로 점진적 변화 분석 중...")
    
    with torch.no_grad():
        for step, feature_value in enumerate(feature_values):
            # 특성 수정
            modified_node_features_df = node_features_df.copy()
            modified_node_features_df.loc[station_name, feature_name] = feature_value
            
            # Tensor로 변환
            modified_node_features = convert_node_features_to_tensor(
                modified_node_features_df, station_to_idx, feature_names
            )
            
            # 샘플별 혼잡도 변화 계산
            step_changes = []
            step_predictions = []
            
            for i in range(min(num_samples, len(test_dataset))):
                sample = test_dataset[i]
                
                X = sample['X'].unsqueeze(0).to(device)
                date_features = sample['date_features'].unsqueeze(0).to(device)
                time_features = sample['time_features'].unsqueeze(0).to(device)
                
                batch_size = X.size(0)
                adjacency = adjacency_tensor.unsqueeze(0).expand(batch_size, -1, -1)
                
                # 원본 예측
                node_features_original = sample['node_features'].unsqueeze(0).to(device)
                pred_original, _ = model(
                    X, adjacency,
                    target_length=1,
                    node_features=node_features_original,
                    date_features=date_features,
                    time_features=time_features
                )
                
                # 수정된 특성으로 예측
                node_features_modified = modified_node_features.unsqueeze(0).to(device)
                pred_modified, _ = model(
                    X, adjacency,
                    target_length=1,
                    node_features=node_features_modified,
                    date_features=date_features,
                    time_features=time_features
                )
                
                # 모델 출력 처리
                while pred_original.dim() > 2:
                    pred_original = pred_original.squeeze(0)
                while pred_modified.dim() > 2:
                    pred_modified = pred_modified.squeeze(0)
                if pred_original.dim() == 1:
                    pred_original = pred_original.unsqueeze(-1)
                if pred_modified.dim() == 1:
                    pred_modified = pred_modified.unsqueeze(-1)
                
                # 역변환
                pred_orig_raw = test_dataset.inverse_transform(pred_original)
                pred_mod_raw = test_dataset.inverse_transform(pred_modified)
                while pred_orig_raw.dim() > 1:
                    pred_orig_raw = pred_orig_raw.squeeze(-1)
                while pred_mod_raw.dim() > 1:
                    pred_mod_raw = pred_mod_raw.squeeze(-1)
                
                # 변화량 계산
                orig_val = pred_orig_raw[station_idx].item()
                mod_val = pred_mod_raw[station_idx].item()
                change = mod_val - orig_val
                change_pct = (change / orig_val * 100) if orig_val > 0 else 0
                
                step_changes.append(change)
                step_predictions.append(mod_val)
            
            # 통계 계산
            mean_change = np.mean(step_changes)
            mean_pred = np.mean(step_predictions)
            std_pred = np.std(step_predictions)
            
            congestion_changes.append(mean_change)
            congestion_means.append(mean_pred)
            congestion_stds.append(std_pred)
            
            if (step + 1) % 5 == 0:
                print(f"  진행: {step+1}/{num_steps} ({feature_name}={feature_value:.2f} → 혼잡도 변화: {mean_change:+.1f}명)")
    
    # 결과 출력
    print(f"\n📈 점진적 변화 분석 결과:")
    print("-"*70)
    print(f"{feature_name:<20} {'혼잡도 변화':<15} {'평균 혼잡도':<15} {'표준편차':<12}")
    print("-"*70)
    for i in range(0, len(feature_values), max(1, len(feature_values)//5)):
        print(f"{feature_values[i]:>6.2f}              "
              f"{congestion_changes[i]:>+12.1f}명    "
              f"{congestion_means[i]:>12.1f}명    "
              f"{congestion_stds[i]:>10.1f}명")
    
    # 변화율 계산
    if len(congestion_changes) > 1:
        total_change = congestion_changes[-1] - congestion_changes[0]
        if congestion_means[0] > 0:
            total_change_pct = (total_change / congestion_means[0]) * 100
            print(f"\n💡 {feature_name} {start_value:.2f} → {end_value:.2f} 변화 시:")
            print(f"   {station_name} 혼잡도: {total_change:+.1f}명 ({total_change_pct:+.1f}%)")
    
    # 시각화
    visualize_gradual_feature_change(
        station_name, feature_name, feature_values,
        congestion_changes, congestion_means, congestion_stds,
        original_value
    )
    
    return {
        'feature_values': feature_values.tolist(),
        'congestion_changes': congestion_changes,
        'congestion_means': congestion_means,
        'congestion_stds': congestion_stds,
        'original_value': original_value
    }


def visualize_gradual_feature_change(
    station_name: str,
    feature_name: str,
    feature_values: np.ndarray,
    congestion_changes: List[float],
    congestion_means: List[float],
    congestion_stds: List[float],
    original_value: float
):
    """점진적 특성 변화 시각화 - 개선된 버전"""
    os.makedirs('insights/visualizations', exist_ok=True)
    
    # 변화율 계산
    start_idx = 0
    end_idx = len(feature_values) - 1
    start_congestion = congestion_means[start_idx]
    end_congestion = congestion_means[end_idx]
    total_change = congestion_changes[end_idx] - congestion_changes[start_idx]
    change_pct = (total_change / start_congestion * 100) if start_congestion > 0 else 0
    
    # 변화율 리스트 계산
    change_percentages = []
    for i, change in enumerate(congestion_changes):
        if congestion_means[0] > 0:
            pct = (change / congestion_means[0]) * 100
        else:
            pct = 0
        change_percentages.append(pct)
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, 
                          left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    # 1. 메인 플롯: 혼잡도 변화량 (큰 플롯)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(feature_values, congestion_changes, 'o-', linewidth=4, markersize=12, 
             color='#e74c3c', label='혼잡도 변화량', zorder=3)
    ax1.fill_between(feature_values, 
                     [c - 0.15*abs(c) for c in congestion_changes],
                     [c + 0.15*abs(c) for c in congestion_changes],
                     alpha=0.25, color='#e74c3c', zorder=1)
    ax1.axhline(y=0, color='#34495e', linestyle='--', linewidth=2, alpha=0.6, zorder=2)
    ax1.axvline(x=original_value, color='#27ae60', linestyle='--', linewidth=3, 
                label=f'현재 값: {original_value:.3f}', alpha=0.8, zorder=2)
    
    # 시작/끝 값 강조
    ax1.scatter([feature_values[start_idx], feature_values[end_idx]], 
               [congestion_changes[start_idx], congestion_changes[end_idx]],
               s=300, color='#c0392b', edgecolors='white', linewidths=3, 
               zorder=4, label='시작/끝 지점')
    
    # 핵심 정보 텍스트 박스
    info_text = f'{feature_name} {feature_values[start_idx]:.2f} → {feature_values[end_idx]:.2f}\n'
    info_text += f'혼잡도 변화: {total_change:+.1f}명 ({change_pct:+.1f}%)'
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
            fontsize=16, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#e74c3c', linewidth=2),
            color='#2c3e50')
    
    ax1.set_xlabel(f'{feature_name}', fontsize=18, fontweight='bold', color='#2c3e50')
    ax1.set_ylabel('혼잡도 변화 (명)', fontsize=18, fontweight='bold', color='#2c3e50')
    ax1.set_title(f'{station_name}: {feature_name} 점진적 변화에 따른 혼잡도 변화', 
                  fontsize=20, fontweight='bold', color='#2c3e50', pad=20)
    ax1.grid(alpha=0.4, linestyle='--', linewidth=1)
    ax1.legend(fontsize=14, loc='upper left', framealpha=0.9)
    ax1.tick_params(labelsize=14)
    
    # 2. 평균 혼잡도 (왼쪽 하단)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(feature_values, congestion_means, 'o-', linewidth=3, markersize=10,
             color='#3498db', label='평균 혼잡도', zorder=3)
    ax2.fill_between(feature_values,
                    [m - s for m, s in zip(congestion_means, congestion_stds)],
                    [m + s for m, s in zip(congestion_means, congestion_stds)],
                    alpha=0.3, color='#3498db', label='±1 표준편차', zorder=1)
    ax2.axvline(x=original_value, color='#27ae60', linestyle='--', linewidth=2,
                label=f'현재: {original_value:.3f}', alpha=0.8, zorder=2)
    ax2.set_xlabel(f'{feature_name}', fontsize=14, fontweight='bold')
    ax2.set_ylabel('평균 혼잡도 (명)', fontsize=14, fontweight='bold')
    ax2.set_title('평균 혼잡도 추이', fontsize=16, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.tick_params(labelsize=12)
    
    # 3. 변화율 (오른쪽 하단)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(feature_values, change_percentages, 's-', linewidth=3, markersize=10,
             color='#9b59b6', label='혼잡도 변화율 (%)', zorder=3)
    ax3.axhline(y=0, color='#34495e', linestyle='--', linewidth=2, alpha=0.6, zorder=2)
    ax3.axvline(x=original_value, color='#27ae60', linestyle='--', linewidth=2,
                label=f'현재: {original_value:.3f}', alpha=0.8, zorder=2)
    
    # 변화율 강조
    if change_percentages[end_idx] > 0:
        ax3.fill_between(feature_values, 0, change_percentages, 
                        where=np.array(change_percentages) >= 0,
                        alpha=0.2, color='#e74c3c', label='증가 구간')
    else:
        ax3.fill_between(feature_values, 0, change_percentages,
                        where=np.array(change_percentages) <= 0,
                        alpha=0.2, color='#3498db', label='감소 구간')
    
    ax3.set_xlabel(f'{feature_name}', fontsize=14, fontweight='bold')
    ax3.set_ylabel('혼잡도 변화율 (%)', fontsize=14, fontweight='bold')
    ax3.set_title('혼잡도 변화율', fontsize=16, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=11)
    ax3.tick_params(labelsize=12)
    
    # 전체 제목
    fig.suptitle(f'{station_name} - {feature_name} 점진적 변화 분석', 
                 fontsize=22, fontweight='bold', color='#2c3e50', y=0.98)
    
    output_path = f'insights/visualizations/gradual_change_{station_name}_{feature_name}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n💾 시각화 저장: {output_path}")
    plt.close()


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="노드 특성 영향 분석")
    parser.add_argument('--samples', type=int, default=30,
                       help='분석 샘플 수')
    parser.add_argument('--station', type=str, default='정부청사역',
                       help='분석할 역 이름')
    parser.add_argument('--feature', type=str, default='관공서_밀도',
                       help='비교할 특성 이름')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/occupancy_model/best_occupancy_model.pth',
                       help='모델 체크포인트 경로')
    parser.add_argument('--gradual', action='store_true',
                       help='점진적 특성 변화 분석 모드')
    parser.add_argument('--start', type=float, default=0.0,
                       help='점진적 분석: 시작 값')
    parser.add_argument('--end', type=float, default=1.0,
                       help='점진적 분석: 끝 값')
    parser.add_argument('--steps', type=int, default=20,
                       help='점진적 분석: 단계 수')
    
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
    
    # 점진적 특성 변화 분석 모드
    if args.gradual:
        gradual_results = analyze_gradual_feature_change(
            model, test_dataset,
            station_name=args.station,
            feature_name=args.feature,
            start_value=args.start,
            end_value=args.end,
            num_steps=args.steps,
            num_samples=args.samples,
            device=device
        )
    else:
        # 특성 중요도 분석
        feature_importance = analyze_node_feature_importance(
            model, test_dataset,
            num_samples=args.samples,
            device=device
        )
        
        # 역 특성 분석
        station_features, hourly_congestion = analyze_station_characteristics(
            model, test_dataset, args.station,
            device=device
        )
        
        # 특성 기반 역 비교
        compare_stations_by_features(
            model, test_dataset, args.feature,
            device=device
        )
    
    print("\n✅ 분석 완료!")


if __name__ == '__main__':
    main()

