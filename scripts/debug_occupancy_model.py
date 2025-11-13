#!/usr/bin/env python3
"""
혼잡도 모델 디버깅 - 모델 상태와 예측 정확성 확인
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

def debug_model_loading():
    print("🔍 혼잡도 모델 디버깅")
    print("="*60)
    
    # 🔧 최신 체크포인트 찾기
    checkpoint_dir = Path("checkpoints/occupancy_model/")
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    
    if checkpoint_files:
        # 가장 최신 에포크 파일 사용
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        checkpoint_path = str(latest_checkpoint)
        print(f"📂 최신 체크포인트 사용: {checkpoint_path}")
    else:
        checkpoint_path = "checkpoints/occupancy_model/best_occupancy_model.pth"
        print(f"📂 Best 모델 사용: {checkpoint_path}")
    
    device = torch.device('cpu')
    
    print(f"📂 체크포인트 파일: {checkpoint_path}")
    print(f"📂 파일 존재: {Path(checkpoint_path).exists()}")
    print(f"📂 파일 크기: {Path(checkpoint_path).stat().st_size / 1024 / 1024:.1f} MB")
    
    # 체크포인트 내용 확인
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"\n📋 체크포인트 정보:")
    print(f"  - Epoch: {checkpoint['epoch']}")
    print(f"  - Best Val Loss: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"  - Config keys: {list(checkpoint.get('config', {}).keys())}")
    
    # 모델 파라미터 확인
    state_dict = checkpoint['model_state_dict']
    print(f"\n🔧 모델 파라미터:")
    for name, param in list(state_dict.items())[:5]:  # 처음 5개만
        print(f"  - {name}: {param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")
    
    # 모델 생성 및 로드
    config = checkpoint['config']
    model = DCRNN(
        input_size=45,  # 🔧 혼잡도(1) + 추가특성(44) = 45차원
        hidden_size=config['model']['hidden_size'],
        output_size=1,
        num_layers=config['model']['num_layers'],
        diffusion_steps=config['model']['diffusion_steps'],
        use_attention=config['model']['use_attention'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # 로드 전 파라미터 상태
    print(f"\n🔧 로드 전 모델 파라미터 (처음 레이어):")
    first_param = next(model.parameters())
    print(f"  - Shape: {first_param.shape}")
    print(f"  - Mean: {first_param.mean().item():.6f}")
    print(f"  - Std: {first_param.std().item():.6f}")
    
    # 모델 로드
    model.load_state_dict(state_dict)
    
    # 로드 후 파라미터 상태
    print(f"\n🔧 로드 후 모델 파라미터 (처음 레이어):")
    first_param = next(model.parameters())
    print(f"  - Shape: {first_param.shape}")
    print(f"  - Mean: {first_param.mean().item():.6f}")
    print(f"  - Std: {first_param.std().item():.6f}")
    
    return model, config

def test_with_detailed_info():
    print("\n" + "="*60)
    print("🎯 상세 정보와 함께 예측 테스트 (개선된 모델)")
    print("="*60)
    
    model, config = debug_model_loading()
    
    # 실제 데이터 로드
    congestion_data = pd.read_csv('data/혼잡도.csv')
    node_features = pd.read_csv('data/node_features.csv')
    date_features = pd.read_csv('data/date_features.csv')
    time_features = pd.read_csv('data/time_features.csv')
    adjacency_matrix = np.load('data/adjacency_matrix.npy')
    
    stations = sorted(congestion_data['station'].unique())
    station_to_idx = {station: idx for idx, station in enumerate(stations)}
    idx_to_station = {idx: station for station, idx in station_to_idx.items()}
    
    # 테스트 데이터 (최근 날짜 몇 개)
    dates = sorted(congestion_data['date'].unique())
    test_dates = dates[-3:]  # 마지막 3일
    test_data = congestion_data[congestion_data['date'].isin(test_dates)]
    
    print(f"📅 테스트 날짜: {test_dates}")
    print(f"📊 테스트 데이터 크기: {len(test_data)}")
    
    test_dataset = OccupancyDataset(
        test_data, node_features, date_features, time_features,
        adjacency_matrix, station_to_idx, 
        sequence_length=12, prediction_length=1,
        target_columns=['occupancy']
    )
    
    print(f"🎯 테스트 시퀀스 수: {len(test_dataset)}")
    
    # 🔧 더 많은 샘플로 검증 (최소 10개)
    model.eval()
    device = torch.device('cpu')
    adjacency_tensor = torch.FloatTensor(adjacency_matrix).to(device)
    
    # 전체 성능 통계를 위한 리스트
    all_predictions = []
    all_targets = []
    sample_details = []
    
    num_samples = min(20, len(test_dataset))  # 최대 20개 샘플
    print(f"🎯 {num_samples}개 샘플로 상세 검증")
    
    for i in range(num_samples):
        print(f"\n📍 샘플 {i+1} 상세 분석:")
        
        # 원본 데이터에서 해당 샘플의 날짜/시간 정보 찾기
        sample = test_dataset[i]
        
        # 🔧 시퀀스 정보 추출 (더 상세하게)
        sequence_info = test_dataset.sequences[i] if hasattr(test_dataset, 'sequences') else None
        
        date_str = "N/A"
        time_str = "N/A"
        weekday_str = "N/A"
        
        if sequence_info:
            date_str = sequence_info.get('date', 'N/A')
            pred_hour = sequence_info.get('pred_hour', 'N/A')
            
            # 시간 정보 더 자세히
            if pred_hour != 'N/A':
                time_str = f"{pred_hour:02d}:00-{pred_hour+1:02d}:00"
            
            # 요일 정보 추가
            if date_str != 'N/A':
                try:
                    from datetime import datetime
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    weekdays = ['월', '화', '수', '목', '금', '토', '일']
                    weekday_str = weekdays[date_obj.weekday()]
                except:
                    weekday_str = "N/A"
        
        print(f"  📅 날짜: {date_str} ({weekday_str})")
        print(f"  🕐 시간: {time_str}")
        
        X = sample['X'].unsqueeze(0).to(device)  # [1, T, N, 1]
        Y = sample['Y'].unsqueeze(0).to(device)  # [1, N, 1]
        
        print(f"  📊 입력 X shape: {X.shape}")
        print(f"  📊 타겟 Y shape: {Y.shape}")
        print(f"  📊 입력 범위: [{X.min().item():.1f}, {X.max().item():.1f}]")
        print(f"  📊 타겟 범위: [{Y.min().item():.1f}, {Y.max().item():.1f}]")
        
        # 🔧 훈련 모드와 평가 모드 둘 다 테스트
        adjacency = adjacency_tensor.unsqueeze(0)  # [1, N, N]
        
        # 훈련 모드 예측
        model.train()
        with torch.no_grad():
            pred_train, _ = model(X, adjacency, target_length=1)
            pred_train = pred_train.squeeze()  # [N]
        
        # 평가 모드 예측  
        model.eval()
        with torch.no_grad():
            predictions, attention = model(X, adjacency, target_length=1)
            predictions = predictions.squeeze()  # [N]
            targets = Y.squeeze()  # [N]
        
        # 🔥 정규화된 값을 원본으로 역변환 (Log1p + 정규화 해제)
        if hasattr(test_dataset, 'inverse_transform'):
            # 예측값 역변환
            pred_train_raw = test_dataset.inverse_transform(pred_train.unsqueeze(-1)).squeeze(-1)
            predictions_raw = test_dataset.inverse_transform(predictions.unsqueeze(-1)).squeeze(-1)
            
            # 타겟값도 역변환 (정규화된 상태였다면)
            if targets.min() < 0 or targets.max() < 10:  # 정규화된 값 같으면
                targets_raw = test_dataset.inverse_transform(targets.unsqueeze(-1)).squeeze(-1)
            else:
                targets_raw = targets  # 이미 원본 값
            
            print(f"  🎯 훈련모드 예측 범위: [{pred_train_raw.min().item():.1f}, {pred_train_raw.max().item():.1f}]명")
            print(f"  🎯 평가모드 예측 범위: [{predictions_raw.min().item():.1f}, {predictions_raw.max().item():.1f}]명")
            print(f"  📊 실제 타겟 범위: [{targets_raw.min().item():.1f}, {targets_raw.max().item():.1f}]명")
            
            # 역변환된 값으로 업데이트
            predictions = predictions_raw
            targets = targets_raw
        else:
            print(f"  🎯 훈련모드 예측 범위: [{pred_train.min().item():.1f}, {pred_train.max().item():.1f}]")
            print(f"  🎯 평가모드 예측 범위: [{predictions.min().item():.1f}, {predictions.max().item():.1f}]")
        
        # 🔧 전체 통계에 추가
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        # 주요 역별 상세 결과 (처음 5개 샘플만 상세 출력)
        if i < 5:
            major_stations = ['대전역', '정부청사역', '시청역', '탄방역', '용문역']
            print(f"  🚇 주요 역별 예측:")
            
            for station in major_stations:
                if station in station_to_idx:
                    idx = station_to_idx[station]
                    pred_val = predictions[idx].item()
                    true_val = targets[idx].item()
                    error = abs(pred_val - true_val)
                    error_pct = (error / true_val * 100) if true_val > 0 else 0
                    
                    print(f"    {station:8s}: 실제 {true_val:6.1f}명 | 예측 {pred_val:6.1f}명 | "
                          f"오차 {error:6.1f}명 ({error_pct:5.1f}%)")
        else:
            # 나머지는 요약만
            sample_mae = np.mean(np.abs(predictions.cpu().numpy() - targets.cpu().numpy()))
            max_true = targets.max().item()
            max_pred = predictions.max().item()
            print(f"  📊 샘플 MAE: {sample_mae:.1f}명, 최대 실제: {max_true:.1f}명, 최대 예측: {max_pred:.1f}명")
        
        # 샘플 정보 저장
        sample_details.append({
            'date': date_str,
            'time': time_str,
            'weekday': weekday_str,
            'mae': np.mean(np.abs(predictions.cpu().numpy() - targets.cpu().numpy())),
            'max_true': targets.max().item(),
            'max_pred': predictions.max().item()
        })
    
    # 🔧 전체 통계 출력
    print(f"\n" + "="*60)
    print(f"📊 전체 {num_samples}개 샘플 통계")
    print("="*60)
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    overall_mae = np.mean(np.abs(all_predictions - all_targets))
    overall_rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
    
    print(f"전체 MAE:  {overall_mae:.2f}명")
    print(f"전체 RMSE: {overall_rmse:.2f}명")
    print(f"예측 범위: [{all_predictions.min():.1f}, {all_predictions.max():.1f}]명")
    print(f"실제 범위: [{all_targets.min():.1f}, {all_targets.max():.1f}]명")
    
    # 🔧 시간대별 성능 분석
    print(f"\n📈 시간대별 성능 분석:")
    time_performance = {}
    for detail in sample_details:
        time_key = detail['time']
        if time_key not in time_performance:
            time_performance[time_key] = []
        time_performance[time_key].append(detail['mae'])
    
    for time_key, maes in sorted(time_performance.items()):
        avg_mae = np.mean(maes)
        print(f"  {time_key:15s}: 평균 MAE {avg_mae:5.1f}명 (샘플 {len(maes)}개)")
    
    # 🔧 요일별 성능 분석
    print(f"\n📅 요일별 성능 분석:")
    weekday_performance = {}
    for detail in sample_details:
        weekday_key = detail['weekday']
        if weekday_key not in weekday_performance:
            weekday_performance[weekday_key] = []
        weekday_performance[weekday_key].append(detail['mae'])
    
    for weekday_key, maes in sorted(weekday_performance.items()):
        avg_mae = np.mean(maes)
        print(f"  {weekday_key:3s}요일: 평균 MAE {avg_mae:5.1f}명 (샘플 {len(maes)}개)")

if __name__ == '__main__':
    test_with_detailed_info()
