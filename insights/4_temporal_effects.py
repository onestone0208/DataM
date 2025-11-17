#!/usr/bin/env python3
"""
인사이트 4: 공휴일/날씨/계절 변화에 따른 혼잡 예측

- 날짜 feature(21개)가 시계열과 결합되면서 특정 날짜 효과도 predict 가능
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
from scripts.simulate_poi_impact import load_model


def analyze_holiday_effects(
    model: DCRNN,
    test_dataset: OccupancyDataset,
    device: torch.device = torch.device('cpu')
):
    """
    공휴일 vs 평일 혼잡도 비교
    """
    print("="*70)
    print("📅 공휴일 효과 분석")
    print("="*70)
    
    # 날짜 특성 로드
    date_features_df = pd.read_csv('data/date_features.csv')
    date_features_df['date'] = pd.to_datetime(date_features_df['date'])
    
    adjacency_tensor = torch.FloatTensor(test_dataset.adjacency_matrix).to(device)
    
    holiday_data = {'pred': [], 'target': []}
    weekday_data = {'pred': [], 'target': []}
    weekend_data = {'pred': [], 'target': []}
    
    print(f"\n📊 전체 테스트 데이터 분석 중...")
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            date_str = sample['metadata']['date']
            date_obj = pd.to_datetime(date_str)
            
            # 공휴일 확인
            date_row = date_features_df[date_features_df['date'] == date_obj]
            is_holiday = False
            is_weekend = date_obj.weekday() >= 5
            
            if len(date_row) > 0:
                # 공휴일 플래그 확인 (컬럼명에 따라 조정 필요)
                if 'is_holiday' in date_row.columns:
                    is_holiday = date_row['is_holiday'].iloc[0] > 0
            
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
            # 모델 출력을 [N, 1] 형태로 변환
            while predictions.dim() > 2:
                predictions = predictions.squeeze(0)
            if predictions.dim() == 1:
                predictions = predictions.unsqueeze(-1)
            
            # 역변환 후 [N] 형태로 변환
            pred_raw = test_dataset.inverse_transform(predictions)
            while pred_raw.dim() > 1:
                pred_raw = pred_raw.squeeze(-1)
            
            avg_pred = pred_raw.mean().item()
            
            if is_holiday:
                holiday_data['pred'].append(avg_pred)
                if 'Y_raw' in sample:
                    holiday_data['target'].append(sample['Y_raw'].mean().item())
            elif is_weekend:
                weekend_data['pred'].append(avg_pred)
                if 'Y_raw' in sample:
                    weekend_data['target'].append(sample['Y_raw'].mean().item())
            else:
                weekday_data['pred'].append(avg_pred)
                if 'Y_raw' in sample:
                    weekday_data['target'].append(sample['Y_raw'].mean().item())
    
    # 결과 출력
    print(f"\n📈 날짜 유형별 평균 혼잡도:")
    print("-"*70)
    
    if holiday_data['pred']:
        holiday_mean = np.mean(holiday_data['pred'])
        print(f"공휴일: {holiday_mean:.1f}명 (샘플: {len(holiday_data['pred'])}개)")
    
    if weekday_data['pred']:
        weekday_mean = np.mean(weekday_data['pred'])
        print(f"평일:   {weekday_mean:.1f}명 (샘플: {len(weekday_data['pred'])}개)")
    
    if weekend_data['pred']:
        weekend_mean = np.mean(weekend_data['pred'])
        print(f"주말:   {weekend_mean:.1f}명 (샘플: {len(weekend_data['pred'])}개)")
    
    # 시각화
    visualize_holiday_effects(holiday_data, weekday_data, weekend_data)
    
    return holiday_data, weekday_data, weekend_data


def analyze_seasonal_effects(
    model: DCRNN,
    test_dataset: OccupancyDataset,
    device: torch.device = torch.device('cpu')
):
    """
    계절별 혼잡도 패턴 분석
    """
    print("\n" + "="*70)
    print("🍂 계절 효과 분석")
    print("="*70)
    
    date_features_df = pd.read_csv('data/date_features.csv')
    date_features_df['date'] = pd.to_datetime(date_features_df['date'])
    
    adjacency_tensor = torch.FloatTensor(test_dataset.adjacency_matrix).to(device)
    
    seasonal_data = {
        '봄': {'pred': [], 'target': []},
        '여름': {'pred': [], 'target': []},
        '가을': {'pred': [], 'target': []},
        '겨울': {'pred': [], 'target': []}
    }
    
    def get_season(month):
        if 3 <= month <= 5:
            return '봄'
        elif 6 <= month <= 8:
            return '여름'
        elif 9 <= month <= 11:
            return '가을'
        else:
            return '겨울'
    
    print(f"\n📊 계절별 데이터 분석 중...")
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            date_str = sample['metadata']['date']
            date_obj = pd.to_datetime(date_str)
            season = get_season(date_obj.month)
            
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
            # 모델 출력을 [N, 1] 형태로 변환
            while predictions.dim() > 2:
                predictions = predictions.squeeze(0)
            if predictions.dim() == 1:
                predictions = predictions.unsqueeze(-1)
            
            # 역변환 후 [N] 형태로 변환
            pred_raw = test_dataset.inverse_transform(predictions)
            while pred_raw.dim() > 1:
                pred_raw = pred_raw.squeeze(-1)
            
            seasonal_data[season]['pred'].append(pred_raw.mean().item())
            if 'Y_raw' in sample:
                seasonal_data[season]['target'].append(sample['Y_raw'].mean().item())
    
    # 결과 출력
    print(f"\n📈 계절별 평균 혼잡도:")
    print("-"*70)
    for season in ['봄', '여름', '가을', '겨울']:
        if seasonal_data[season]['pred']:
            mean_occ = np.mean(seasonal_data[season]['pred'])
            print(f"{season}: {mean_occ:.1f}명 (샘플: {len(seasonal_data[season]['pred'])}개)")
    
    # 시각화
    visualize_seasonal_effects(seasonal_data)
    
    return seasonal_data


def analyze_temperature_effects(
    model: DCRNN,
    test_dataset: OccupancyDataset,
    device: torch.device = torch.device('cpu')
):
    """
    기온에 따른 혼잡도 변화 분석
    """
    print("\n" + "="*70)
    print("🌡️ 기온 효과 분석")
    print("="*70)
    
    date_features_df = pd.read_csv('data/date_features.csv')
    date_features_df['date'] = pd.to_datetime(date_features_df['date'])
    
    # 기온 컬럼 찾기
    temp_cols = [col for col in date_features_df.columns if '기온' in col or 'temp' in col.lower()]
    
    if not temp_cols:
        print("⚠️ 기온 데이터를 찾을 수 없습니다.")
        return None
    
    temp_col = temp_cols[0]  # 첫 번째 기온 컬럼 사용
    
    adjacency_tensor = torch.FloatTensor(test_dataset.adjacency_matrix).to(device)
    
    # 기온 구간별 데이터 수집
    temp_ranges = {
        '매우 추움 (<0°C)': [],
        '추움 (0-10°C)': [],
        '보통 (10-20°C)': [],
        '따뜻함 (20-30°C)': [],
        '더움 (>30°C)': []
    }
    
    print(f"\n📊 기온별 데이터 분석 중...")
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            date_str = sample['metadata']['date']
            date_obj = pd.to_datetime(date_str)
            
            date_row = date_features_df[date_features_df['date'] == date_obj]
            if len(date_row) > 0 and temp_col in date_row.columns:
                temp = date_row[temp_col].iloc[0]
                
                if pd.notna(temp):
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
                    # 모델 출력을 [N, 1] 형태로 변환
                    while predictions.dim() > 2:
                        predictions = predictions.squeeze(0)
                    if predictions.dim() == 1:
                        predictions = predictions.unsqueeze(-1)
                    
                    # 역변환 후 [N] 형태로 변환
                    pred_raw = test_dataset.inverse_transform(predictions)
                    while pred_raw.dim() > 1:
                        pred_raw = pred_raw.squeeze(-1)
                    
                    if temp < 0:
                        temp_ranges['매우 추움 (<0°C)'].append(pred_raw.mean().item())
                    elif temp < 10:
                        temp_ranges['추움 (0-10°C)'].append(pred_raw.mean().item())
                    elif temp < 20:
                        temp_ranges['보통 (10-20°C)'].append(pred_raw.mean().item())
                    elif temp < 30:
                        temp_ranges['따뜻함 (20-30°C)'].append(pred_raw.mean().item())
                    else:
                        temp_ranges['더움 (>30°C)'].append(pred_raw.mean().item())
    
    # 결과 출력
    print(f"\n📈 기온 구간별 평균 혼잡도:")
    print("-"*70)
    for temp_range, data in temp_ranges.items():
        if data:
            mean_occ = np.mean(data)
            print(f"{temp_range:<20}: {mean_occ:.1f}명 (샘플: {len(data)}개)")
    
    # 시각화
    visualize_temperature_effects(temp_ranges)
    
    return temp_ranges


def visualize_holiday_effects(holiday_data, weekday_data, weekend_data):
    """공휴일 효과 시각화"""
    os.makedirs('insights/visualizations', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 박스플롯
    data_to_plot = []
    labels = []
    
    if holiday_data['pred']:
        data_to_plot.append(holiday_data['pred'])
        labels.append('공휴일')
    if weekday_data['pred']:
        data_to_plot.append(weekday_data['pred'])
        labels.append('평일')
    if weekend_data['pred']:
        data_to_plot.append(weekend_data['pred'])
        labels.append('주말')
    
    if data_to_plot:
        axes[0].boxplot(data_to_plot, labels=labels)
        axes[0].set_ylabel('혼잡도 (명)')
        axes[0].set_title('날짜 유형별 혼잡도 분포')
        axes[0].grid(alpha=0.3, axis='y')
    
    # 2. 막대 그래프
    means = []
    if holiday_data['pred']:
        means.append(np.mean(holiday_data['pred']))
    else:
        means.append(0)
    
    if weekday_data['pred']:
        means.append(np.mean(weekday_data['pred']))
    else:
        means.append(0)
    
    if weekend_data['pred']:
        means.append(np.mean(weekend_data['pred']))
    else:
        means.append(0)
    
    axes[1].bar(['공휴일', '평일', '주말'], means, alpha=0.7)
    axes[1].set_ylabel('평균 혼잡도 (명)')
    axes[1].set_title('날짜 유형별 평균 혼잡도')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = 'insights/visualizations/holiday_effects.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 시각화 저장: {output_path}")
    plt.close()


def visualize_seasonal_effects(seasonal_data):
    """계절 효과 시각화"""
    os.makedirs('insights/visualizations', exist_ok=True)
    
    seasons = ['봄', '여름', '가을', '겨울']
    means = [np.mean(seasonal_data[s]['pred']) if seasonal_data[s]['pred'] else 0 
             for s in seasons]
    
    plt.figure(figsize=(10, 6))
    plt.bar(seasons, means, alpha=0.7, color=['green', 'orange', 'brown', 'blue'])
    plt.ylabel('평균 혼잡도 (명)')
    plt.title('계절별 평균 혼잡도')
    plt.grid(alpha=0.3, axis='y')
    
    output_path = 'insights/visualizations/seasonal_effects.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 시각화 저장: {output_path}")
    plt.close()


def visualize_temperature_effects(temp_ranges):
    """기온 효과 시각화"""
    os.makedirs('insights/visualizations', exist_ok=True)
    
    ranges = ['매우 추움 (<0°C)', '추움 (0-10°C)', '보통 (10-20°C)', 
              '따뜻함 (20-30°C)', '더움 (>30°C)']
    means = [np.mean(temp_ranges[r]) if temp_ranges[r] else 0 for r in ranges]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(ranges)), means, alpha=0.7, 
            color=['blue', 'lightblue', 'green', 'orange', 'red'])
    plt.xticks(range(len(ranges)), ranges, rotation=45, ha='right')
    plt.ylabel('평균 혼잡도 (명)')
    plt.title('기온 구간별 평균 혼잡도')
    plt.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = 'insights/visualizations/temperature_effects.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 시각화 저장: {output_path}")
    plt.close()


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="시간/날짜 효과 분석")
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
    
    # 공휴일 효과 분석
    holiday_data, weekday_data, weekend_data = analyze_holiday_effects(
        model, test_dataset, device
    )
    
    # 계절 효과 분석
    seasonal_data = analyze_seasonal_effects(
        model, test_dataset, device
    )
    
    # 기온 효과 분석
    temp_data = analyze_temperature_effects(
        model, test_dataset, device
    )
    
    print("\n✅ 분석 완료!")


if __name__ == '__main__':
    main()

