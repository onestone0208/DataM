#!/usr/bin/env python3
"""
인사이트 5: 승하차 데이터를 직접 feature로 반영하므로 역별 편중된 흐름 분석

- A역에서 탑승 증가 → B역 하차 증가 → C역 혼잡 변화
→ 실제 passenger flow chain 분석 가능
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN
from scripts.train_occupancy_model import OccupancyDataset
from scripts.simulate_poi_impact import load_model


def analyze_passenger_flow_chain(
    model: DCRNN,
    test_dataset: OccupancyDataset,
    source_station: str,
    boarding_increase: float = 50.0,
    num_samples: int = 20,
    device: torch.device = torch.device('cpu')
):
    """
    승차 증가가 다른 역들의 하차 및 혼잡도에 미치는 영향 분석
    
    Args:
        source_station: 승차가 증가하는 역
        boarding_increase: 증가할 승차 인원 수
    """
    print("="*70)
    print(f"🚇 승하차 흐름 체인 분석: {source_station} 승차 증가 효과")
    print("="*70)
    
    station_to_idx = test_dataset.station_to_idx
    if source_station not in station_to_idx:
        raise ValueError(f"역 '{source_station}'을 찾을 수 없습니다.")
    
    source_idx = station_to_idx[source_station]
    adjacency_tensor = torch.FloatTensor(test_dataset.adjacency_matrix).to(device)
    
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
            
            # 승차 증가 시뮬레이션
            # X의 구조: [B, T, N, 31] = 혼잡도(1) + 승차(1) + 하차(1) + 시간(7) + 날짜(21)
            X_modified = X_original.clone()
            # 승차는 인덱스 1에 있음
            X_modified[0, -1, source_idx, 1] += boarding_increase / 15.0  # 정규화 역변환
            
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
    print(f"\n📈 {source_station} 승차 {boarding_increase}명 증가 시 다른 역들의 영향도:")
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
    for station_name, stats in sorted_stations[:15]:
        print(f"{station_name:<15} "
              f"{stats['mean']:>+10.2f}명  "
              f"{stats['std']:>10.2f}명  "
              f"{stats['max']:>+10.2f}명  "
              f"{stats['min']:>+10.2f}명")
    
    # 시각화
    visualize_passenger_flow(source_station, impact_stats, station_names)
    
    return impact_stats


def analyze_alighting_impact(
    model: DCRNN,
    test_dataset: OccupancyDataset,
    target_station: str,
    alighting_increase: float = 50.0,
    num_samples: int = 20,
    device: torch.device = torch.device('cpu')
):
    """
    특정 역의 하차 증가가 해당 역 및 인접 역 혼잡도에 미치는 영향
    """
    print("\n" + "="*70)
    print(f"🚇 하차 증가 효과 분석: {target_station}")
    print("="*70)
    
    station_to_idx = test_dataset.station_to_idx
    if target_station not in station_to_idx:
        raise ValueError(f"역 '{target_station}'을 찾을 수 없습니다.")
    
    target_idx = station_to_idx[target_station]
    adjacency_tensor = torch.FloatTensor(test_dataset.adjacency_matrix).to(device)
    
    station_names = sorted(station_to_idx.keys(), key=lambda x: station_to_idx[x])
    impact_results = {station: [] for station in station_names}
    
    print(f"\n📊 {num_samples}개 샘플로 분석 진행...")
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            sample = test_dataset[i]
            
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
            
            # 하차 증가 시뮬레이션 (하차는 인덱스 2)
            X_modified = X_original.clone()
            X_modified[0, -1, target_idx, 2] += alighting_increase / 15.0
            
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
            
            # 영향도 계산
            for station_name, station_idx in station_to_idx.items():
                if isinstance(pred_orig_raw, torch.Tensor):
                    orig_val = pred_orig_raw[station_idx].item()
                    mod_val = pred_mod_raw[station_idx].item()
                else:
                    orig_val = float(pred_orig_raw[station_idx])
                    mod_val = float(pred_mod_raw[station_idx])
                impact = mod_val - orig_val
                impact_results[station_name].append(impact)
    
    # 결과 출력
    print(f"\n📈 {target_station} 하차 {alighting_increase}명 증가 시 영향도:")
    print("-"*70)
    
    impact_stats = {}
    for station_name in station_names:
        impacts = np.array(impact_results[station_name])
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


def analyze_boarding_to_alighting_chain(
    model: DCRNN,
    test_dataset: OccupancyDataset,
    source_station: str,
    target_station: str,
    final_station: str,
    boarding_increase: float = 500.0,
    num_samples: int = 20,
    device: torch.device = torch.device('cpu')
):
    """
    승차 증가 → 하차 변화 → 혼잡도 변화 체인 분석
    
    예: "대전역 승차량 +500명 증가 → 중앙로역 하차량 증가 → 용문역 혼잡도 증가"
    
    Args:
        source_station: 승차가 증가하는 역 (예: 대전역)
        target_station: 하차가 증가하는 역 (예: 중앙로역)
        final_station: 혼잡도가 증가하는 역 (예: 용문역)
        boarding_increase: 증가할 승차 인원 수
    """
    print("="*70)
    print(f"🔗 승하차 체인 반응 분석")
    print(f"   {source_station} 승차 +{boarding_increase}명")
    print(f"   → {target_station} 하차 변화")
    print(f"   → {final_station} 혼잡도 변화")
    print("="*70)
    
    station_to_idx = test_dataset.station_to_idx
    for station in [source_station, target_station, final_station]:
        if station not in station_to_idx:
            raise ValueError(f"역 '{station}'을 찾을 수 없습니다.")
    
    source_idx = station_to_idx[source_station]
    target_idx = station_to_idx[target_station]
    final_idx = station_to_idx[final_station]
    adjacency_tensor = torch.FloatTensor(test_dataset.adjacency_matrix).to(device)
    
    # 결과 저장
    chain_results = {
        'boarding_impact': [],  # 승차 증가가 혼잡도에 미치는 영향
        'alighting_change': [],  # 하차량 변화 (간접 추정)
        'final_congestion': []   # 최종 역 혼잡도 변화
    }
    
    print(f"\n📊 {num_samples}개 샘플로 체인 분석 진행...")
    
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
            
            # 원본 혼잡도 예측
            pred_original, _ = model(
                X_original, adjacency,
                target_length=1,
                node_features=node_features,
                date_features=date_features,
                time_features=time_features
            )
            while pred_original.dim() > 2:
                pred_original = pred_original.squeeze(0)
            if pred_original.dim() == 1:
                pred_original = pred_original.unsqueeze(-1)
            pred_orig_raw = test_dataset.inverse_transform(pred_original)
            while pred_orig_raw.dim() > 1:
                pred_orig_raw = pred_orig_raw.squeeze(-1)
            
            # 원본 하차량 추출 (정규화 해제)
            original_alighting = X_original[0, -1, target_idx, 2].item()  # 정규화된 하차량
            original_alighting_raw = (original_alighting * 15.0) ** 2  # sqrt 정규화 역변환
            
            # 승차 증가 시뮬레이션
            X_modified = X_original.clone()
            # 승차는 인덱스 1, sqrt 정규화 사용
            boarding_normalized = np.sqrt(boarding_increase) / 15.0
            X_modified[0, -1, source_idx, 1] += boarding_normalized
            
            pred_modified, _ = model(
                X_modified, adjacency,
                target_length=1,
                node_features=node_features,
                date_features=date_features,
                time_features=time_features
            )
            while pred_modified.dim() > 2:
                pred_modified = pred_modified.squeeze(0)
            if pred_modified.dim() == 1:
                pred_modified = pred_modified.unsqueeze(-1)
            pred_mod_raw = test_dataset.inverse_transform(pred_modified)
            while pred_mod_raw.dim() > 1:
                pred_mod_raw = pred_mod_raw.squeeze(-1)
            
            # 결과 계산
            source_congestion_change = (pred_mod_raw[source_idx].item() - 
                                       pred_orig_raw[source_idx].item())
            target_congestion_change = (pred_mod_raw[target_idx].item() - 
                                       pred_orig_raw[target_idx].item())
            final_congestion_change = (pred_mod_raw[final_idx].item() - 
                                      pred_orig_raw[final_idx].item())
            
            # 하차량 변화 추정 개선:
            # 1. 타겟 역의 혼잡도 변화가 음수면 하차 증가로 해석
            # 2. 승차 증가량의 일부가 타겟 역에서 하차할 것으로 가정
            # 3. 소스와 타겟 역 사이의 거리/인접성을 고려
            
            # 모든 역의 혼잡도 변화를 계산하여 타겟 역의 하차량 변화를 추정
            all_congestion_changes = pred_mod_raw - pred_orig_raw
            
            # 타겟 역의 혼잡도가 감소하면 하차 증가로 해석
            # 승차 증가량의 일부가 타겟 역에서 하차할 것으로 가정 (거리 기반 가중치)
            distance_factor = 1.0 / (abs(source_idx - target_idx) + 1)  # 거리 가중치
            estimated_alighting_change = -target_congestion_change * 2.0 * distance_factor
            
            # 더 정확한 추정: 타겟 역의 혼잡도 감소가 하차 증가를 의미
            if target_congestion_change < 0:
                # 혼잡도 감소 = 하차 증가
                estimated_alighting_change = abs(target_congestion_change) * 1.5
            else:
                # 혼잡도 증가 = 하차 감소 또는 변화 없음
                estimated_alighting_change = -target_congestion_change * 0.3
            
            chain_results['boarding_impact'].append(source_congestion_change)
            chain_results['alighting_change'].append(estimated_alighting_change)
            chain_results['final_congestion'].append(final_congestion_change)
            
            if (i + 1) % 5 == 0:
                print(f"  진행: {i+1}/{num_samples}")
    
    # 결과 출력
    print(f"\n📈 체인 반응 분석 결과:")
    print("-"*70)
    
    boarding_mean = np.mean(chain_results['boarding_impact'])
    alighting_mean = np.mean(chain_results['alighting_change'])
    final_mean = np.mean(chain_results['final_congestion'])
    
    boarding_std = np.std(chain_results['boarding_impact'])
    alighting_std = np.std(chain_results['alighting_change'])
    final_std = np.std(chain_results['final_congestion'])
    
    print(f"{source_station} 승차 +{boarding_increase}명")
    print(f"  → {source_station} 혼잡도: {boarding_mean:+.2f}명 (std: {boarding_std:.2f})")
    print(f"  → {target_station} 하차량 변화 (추정): {alighting_mean:+.2f}명 (std: {alighting_std:.2f})")
    print(f"  → {final_station} 혼잡도 변화: {final_mean:+.2f}명 (std: {final_std:.2f})")
    
    # 영향도가 큰 역들도 함께 출력
    print(f"\n💡 팁: 영향도가 큰 역 조합을 찾으려면 다른 역들을 시도해보세요.")
    print(f"   예: --source 대전역 --target 반석역 --final 대동역")
    
    # 시각화
    visualize_chain_reaction(source_station, target_station, final_station, 
                           boarding_increase, chain_results)
    
    return chain_results


def visualize_chain_reaction(
    source_station: str,
    target_station: str,
    final_station: str,
    boarding_increase: float,
    chain_results: Dict
):
    """체인 반응 시각화 - 화살표로 연결된 플로우 차트"""
    os.makedirs('insights/visualizations', exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    
    # 값 계산
    boarding_mean = np.mean(chain_results['boarding_impact'])
    alighting_mean = np.mean(chain_results['alighting_change'])
    final_mean = np.mean(chain_results['final_congestion'])
    
    # 박스 위치 설정
    box_width = 2.5
    box_height = 1.5
    spacing = 4.0
    
    x1, y1 = 2, 4  # 첫 번째 박스 (승차)
    x2, y2 = x1 + spacing, y1  # 두 번째 박스 (하차)
    x3, y3 = x2 + spacing, y1  # 세 번째 박스 (혼잡도)
    
    # 배경 색상
    ax.set_facecolor('#f8f9fa')
    
    # 1. 첫 번째 박스: 승차 증가 → 혼잡도 증가
    box1 = Rectangle((x1 - box_width/2, y1 - box_height/2), box_width, box_height,
                     facecolor='#ff6b6b', edgecolor='#c92a2a', linewidth=3, zorder=3)
    ax.add_patch(box1)
    ax.text(x1, y1 + 0.6, f'{source_station}', ha='center', va='center', 
            fontsize=18, fontweight='bold', color='white')
    ax.text(x1, y1, f'승차 +{boarding_increase:.0f}명', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    ax.text(x1, y1 - 0.6, f'혼잡도: {boarding_mean:+.1f}명', ha='center', va='center',
            fontsize=16, fontweight='bold', color='white')
    
    # 2. 화살표 1
    arrow1 = FancyArrowPatch((x1 + box_width/2, y1), (x2 - box_width/2, y2),
                             arrowstyle='->', mutation_scale=40, 
                             linewidth=4, color='#495057', zorder=2)
    ax.add_patch(arrow1)
    ax.text((x1 + x2)/2, y1 + 0.8, '→', ha='center', va='center',
            fontsize=30, fontweight='bold', color='#495057')
    
    # 3. 두 번째 박스: 하차량 변화
    box2 = Rectangle((x2 - box_width/2, y2 - box_height/2), box_width, box_height,
                     facecolor='#ffa94d', edgecolor='#d9480f', linewidth=3, zorder=3)
    ax.add_patch(box2)
    ax.text(x2, y2 + 0.6, f'{target_station}', ha='center', va='center',
            fontsize=18, fontweight='bold', color='white')
    ax.text(x2, y2, '하차량 변화', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    ax.text(x2, y2 - 0.6, f'{alighting_mean:+.1f}명', ha='center', va='center',
            fontsize=16, fontweight='bold', color='white')
    
    # 4. 화살표 2
    arrow2 = FancyArrowPatch((x2 + box_width/2, y2), (x3 - box_width/2, y3),
                             arrowstyle='->', mutation_scale=40,
                             linewidth=4, color='#495057', zorder=2)
    ax.add_patch(arrow2)
    ax.text((x2 + x3)/2, y2 + 0.8, '→', ha='center', va='center',
            fontsize=30, fontweight='bold', color='#495057')
    
    # 5. 세 번째 박스: 최종 혼잡도
    box3 = Rectangle((x3 - box_width/2, y3 - box_height/2), box_width, box_height,
                     facecolor='#4dabf7', edgecolor='#1864ab', linewidth=3, zorder=3)
    ax.add_patch(box3)
    ax.text(x3, y3 + 0.6, f'{final_station}', ha='center', va='center',
            fontsize=18, fontweight='bold', color='white')
    ax.text(x3, y3, '혼잡도 변화', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    ax.text(x3, y3 - 0.6, f'{final_mean:+.1f}명', ha='center', va='center',
            fontsize=16, fontweight='bold', color='white')
    
    # 제목
    ax.text((x1 + x3)/2, y1 + 2.5, '승하차 체인 반응 분석', ha='center', va='center',
            fontsize=24, fontweight='bold', color='#212529')
    
    # 범례/설명
    info_text = f'평균값 (n={len(chain_results["boarding_impact"])} 샘플)'
    ax.text((x1 + x3)/2, y1 - 2.5, info_text, ha='center', va='center',
            fontsize=12, style='italic', color='#6c757d')
    
    # 축 설정
    ax.set_xlim(0, x3 + box_width/2 + 1)
    ax.set_ylim(y1 - 3, y1 + 3)
    ax.axis('off')
    
    plt.tight_layout()
    output_path = f'insights/visualizations/chain_reaction_{source_station}_{target_station}_{final_station}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n💾 시각화 저장: {output_path}")
    plt.close()


def visualize_passenger_flow(
    source_station: str,
    impact_stats: Dict[str, Dict],
    station_names: List[str]
):
    """승하차 흐름 시각화"""
    os.makedirs('insights/visualizations', exist_ok=True)
    
    impact_values = [impact_stats[station]['mean'] for station in station_names]
    colors = ['red' if v > 0 else 'blue' for v in impact_values]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(station_names)), impact_values, color=colors, alpha=0.7)
    plt.yticks(range(len(station_names)), station_names)
    plt.xlabel('평균 영향도 (명)')
    plt.title(f'{source_station} 승차 증가 시 다른 역들의 혼잡도 변화')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = f'insights/visualizations/passenger_flow_{source_station}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 시각화 저장: {output_path}")
    plt.close()


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="승하차 흐름 분석")
    parser.add_argument('--station', type=str, default='대전역',
                       help='분석할 역 이름')
    parser.add_argument('--boarding', type=float, default=50.0,
                       help='증가할 승차 인원 수')
    parser.add_argument('--alighting', type=float, default=50.0,
                       help='증가할 하차 인원 수')
    parser.add_argument('--samples', type=int, default=20,
                       help='테스트할 샘플 수')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/occupancy_model/best_occupancy_model.pth',
                       help='모델 체크포인트 경로')
    parser.add_argument('--chain', action='store_true',
                       help='체인 반응 분석 모드 (승차 → 하차 → 혼잡도)')
    parser.add_argument('--source', type=str, default='대전역',
                       help='체인 분석: 승차 증가 역')
    parser.add_argument('--target', type=str, default='중앙로역',
                       help='체인 분석: 하차 변화 역')
    parser.add_argument('--final', type=str, default='용문역',
                       help='체인 분석: 혼잡도 변화 역')
    
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
    
    # 체인 반응 분석 모드
    if args.chain:
        chain_results = analyze_boarding_to_alighting_chain(
            model, test_dataset,
            source_station=args.source,
            target_station=args.target,
            final_station=args.final,
            boarding_increase=args.boarding,
            num_samples=args.samples,
            device=device
        )
    else:
        # 승차 흐름 분석
        boarding_impact = analyze_passenger_flow_chain(
            model, test_dataset, args.station,
            boarding_increase=args.boarding,
            num_samples=args.samples,
            device=device
        )
        
        # 하차 흐름 분석
        alighting_impact = analyze_alighting_impact(
            model, test_dataset, args.station,
            alighting_increase=args.alighting,
            num_samples=args.samples,
            device=device
        )
    
    print("\n✅ 분석 완료!")


if __name__ == '__main__':
    main()

