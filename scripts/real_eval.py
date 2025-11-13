#!/usr/bin/env python3
"""
실제 데이터로 모델 성능 평가
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN
from models.dataset import SubwayGraphDataset

def load_model_and_config():
    """모델과 설정 로드"""
    print("=== 실제 데이터로 모델 성능 평가 ===")
    
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

def create_test_dataset():
    """테스트 데이터셋 생성 (훈련 스크립트와 동일한 방식)"""
    print("\n=== 테스트 데이터셋 생성 ===")
    
    # 원본 데이터 로드
    congestion_data = pd.read_csv('data/혼잡도.csv')
    node_features = pd.read_csv('data/node_features.csv')
    date_features = pd.read_csv('data/date_features.csv')
    time_features = pd.read_csv('data/time_features.csv')
    adjacency_matrix = np.load('data/adjacency_matrix.npy')
    
    print(f"📊 전체 데이터: {len(congestion_data)} 레코드")
    
    # 역 인덱스 매핑
    stations = sorted(congestion_data['station'].unique())
    station_to_idx = {station: idx for idx, station in enumerate(stations)}
    
    print(f"📍 역 수: {len(stations)}개")
    
    # 데이터 분할 (test: 마지막 15%)
    dates = sorted(congestion_data['date'].unique())
    n_dates = len(dates)
    
    test_start = int(n_dates * 0.85)  # 마지막 15%를 테스트용으로
    test_dates = dates[test_start:]
    
    print(f"📅 테스트 기간: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}일)")
    
    # 테스트 데이터 필터링
    test_data = congestion_data[congestion_data['date'].isin(test_dates)]
    print(f"📊 테스트 데이터: {len(test_data)} 레코드")
    
    # 데이터셋 생성
    test_dataset = SubwayGraphDataset(
        test_data, node_features, date_features, time_features,
        adjacency_matrix, station_to_idx, 
        sequence_length=12, prediction_length=1,
        target_columns=['occupancy', 'total_flow']
    )
    
    print(f"✅ 테스트 데이터셋 생성 완료: {len(test_dataset)} 샘플")
    
    return test_dataset, adjacency_matrix

def evaluate_model():
    """모델 평가 실행"""
    model, device, checkpoint = load_model_and_config()
    
    # 테스트 데이터셋 생성
    test_dataset, adjacency_matrix = create_test_dataset()
    
    # 데이터로더 생성
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,  # 작은 배치 크기로 안정성 확보
        shuffle=False,
        num_workers=0,  # 멀티프로세싱 비활성화
        pin_memory=False
    )
    
    print(f"📦 테스트 배치 수: {len(test_loader)}")
    
    # 인접행렬 준비
    adjacency = torch.FloatTensor(adjacency_matrix).to(device)
    
    print("\n=== 예측 수행 ===")
    
    all_predictions = []
    all_targets = []
    processed_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            try:
                X = batch['X'].to(device)  # [B, T, N, 2]
                Y = batch['Y'].to(device)  # [B, N, 2]
                
                batch_size = X.size(0)
                adj_batch = adjacency.unsqueeze(0).expand(batch_size, -1, -1)
                
                # 모델 예측
                predictions, _ = model(X, adj_batch, target_length=1)
                predictions = predictions.squeeze(1)  # [B, N, 2]
                
                # CPU로 이동
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(Y.cpu().numpy())
                
                processed_samples += batch_size
                
                if batch_idx % 10 == 0:
                    print(f"  진행률: {batch_idx+1}/{len(test_loader)} 배치 ({processed_samples} 샘플)")
                    
            except Exception as e:
                print(f"❌ 배치 {batch_idx} 처리 중 오류: {e}")
                continue
    
    if not all_predictions:
        print("❌ 예측 결과가 없습니다!")
        return
    
    # 결과 합치기
    predictions = np.concatenate(all_predictions, axis=0)  # [N_samples, N_stations, 2]
    targets = np.concatenate(all_targets, axis=0)
    
    print(f"✅ 예측 완료: {predictions.shape}")
    
    # 성능 지표 계산
    print("\n=== 📈 성능 지표 ===")
    
    feature_names = ['혼잡도 (Occupancy)', '통행량 (Total Flow)']
    overall_results = {}
    
    for i, feature_name in enumerate(feature_names):
        pred_feature = predictions[:, :, i].flatten()
        target_feature = targets[:, :, i].flatten()
        
        # 유효한 값만 사용 (0이 아닌 값)
        valid_mask = target_feature != 0
        if valid_mask.sum() > 0:
            pred_valid = pred_feature[valid_mask]
            target_valid = target_feature[valid_mask]
            
            mae = mean_absolute_error(target_valid, pred_valid)
            rmse = np.sqrt(mean_squared_error(target_valid, pred_valid))
            mape = np.mean(np.abs((target_valid - pred_valid) / (target_valid + 1e-8))) * 100
            r2 = r2_score(target_valid, pred_valid)
            
            overall_results[feature_name] = {
                'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2,
                'valid_samples': len(pred_valid)
            }
            
            print(f"\n🎯 {feature_name}:")
            print(f"   MAE:  {mae:.3f}")
            print(f"   RMSE: {rmse:.3f}")
            print(f"   MAPE: {mape:.1f}%")
            print(f"   R²:   {r2:.3f}")
            print(f"   유효 샘플: {len(pred_valid):,}개")
            
            # 데이터 범위
            print(f"   실제값 범위: [{target_valid.min():.1f}, {target_valid.max():.1f}]")
            print(f"   예측값 범위: [{pred_valid.min():.1f}, {pred_valid.max():.1f}]")
            
        else:
            print(f"\n🎯 {feature_name}: 유효한 데이터 없음")
    
    # 전체 성능
    all_pred = predictions.flatten()
    all_target = targets.flatten()
    valid_mask = all_target != 0
    
    if valid_mask.sum() > 0:
        all_pred_valid = all_pred[valid_mask]
        all_target_valid = all_target[valid_mask]
        
        overall_mae = mean_absolute_error(all_target_valid, all_pred_valid)
        overall_rmse = np.sqrt(mean_squared_error(all_target_valid, all_pred_valid))
        overall_r2 = r2_score(all_target_valid, all_pred_valid)
        
        print(f"\n🏆 전체 성능:")
        print(f"   전체 MAE:  {overall_mae:.3f}")
        print(f"   전체 RMSE: {overall_rmse:.3f}")
        print(f"   전체 R²:   {overall_r2:.3f}")
    
    # 훈련 정보
    print(f"\n📚 모델 정보:")
    print(f"   현재 에포크: {checkpoint['epoch']}")
    print(f"   총 샘플 수: {len(predictions):,}개")
    print(f"   역 수: {predictions.shape[1]}개")
    print(f"   특성 수: {predictions.shape[2]}개")
    
    return predictions, targets, overall_results

if __name__ == '__main__':
    try:
        evaluate_model()
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
