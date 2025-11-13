#!/usr/bin/env python3
"""
빠른 모델 성능 평가
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN

def load_model_and_data():
    """모델과 데이터 로드"""
    print("=== 모델 성능 평가 ===")
    
    # 체크포인트 로드
    device = torch.device('cpu')
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    
    print(f"✅ 모델 로드 완료 (Epoch {checkpoint['epoch']})")
    train_loss = checkpoint.get('train_loss', 'N/A')
    val_loss = checkpoint.get('val_loss', 'N/A')
    
    if isinstance(train_loss, (int, float)):
        print(f"📊 최종 Train Loss: {train_loss:.4f}")
    else:
        print(f"📊 최종 Train Loss: {train_loss}")
        
    if isinstance(val_loss, (int, float)):
        print(f"📊 최종 Val Loss: {val_loss:.4f}")
    else:
        print(f"📊 최종 Val Loss: {val_loss}")
    
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

def create_synthetic_test():
    """간단한 테스트 데이터 생성"""
    print("\n=== 합성 데이터로 성능 테스트 ===")
    
    # 22개 지하철역, 2개 특성 (혼잡도, 통행량)
    n_samples = 50
    n_stations = 22
    seq_len = 12
    n_features = 2
    
    # 실제와 유사한 범위의 데이터 생성
    np.random.seed(42)
    
    # 입력 시퀀스 [batch, seq_len, stations, features]
    X = np.random.uniform(0, 100, (n_samples, seq_len, n_stations, n_features))
    
    # 타겟 [batch, stations, features] 
    Y = np.random.uniform(0, 100, (n_samples, n_stations, n_features))
    
    # 일부 현실적인 패턴 추가
    for i in range(n_samples):
        # 시간대별 패턴
        for t in range(seq_len):
            if 7 <= t <= 9 or 17 <= t <= 19:  # 출퇴근 시간
                X[i, t, :, 0] *= 1.5  # 혼잡도 증가
                X[i, t, :, 1] *= 1.3  # 통행량 증가
        
        # 타겟도 비슷한 패턴
        Y[i, :, 0] = X[i, -1, :, 0] * 0.9 + np.random.normal(0, 5, n_stations)
        Y[i, :, 1] = X[i, -1, :, 1] * 0.9 + np.random.normal(0, 10, n_stations)
    
    return torch.FloatTensor(X), torch.FloatTensor(Y)

def evaluate_performance():
    """성능 평가 실행"""
    model, device, checkpoint = load_model_and_data()
    
    # 테스트 데이터 생성
    X_test, Y_test = create_synthetic_test()
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    
    # 인접행렬 로드
    adjacency = torch.FloatTensor(np.load('data/adjacency_matrix.npy')).to(device)
    
    print(f"📊 테스트 데이터: {X_test.shape[0]}개 샘플, {X_test.shape[2]}개 역")
    
    # 예측 수행
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        batch_size = 10
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size]
            batch_Y = Y_test[i:i+batch_size]
            
            current_batch_size = batch_X.size(0)
            adj_batch = adjacency.unsqueeze(0).expand(current_batch_size, -1, -1)
            
            # 모델 예측
            predictions, attention = model(batch_X, adj_batch, target_length=1)
            predictions = predictions.squeeze(1)  # [batch, stations, features]
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_Y.cpu().numpy())
    
    # 결과 합치기
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    print(f"✅ 예측 완료: {predictions.shape}")
    
    # 성능 지표 계산
    print("\n=== 📈 성능 지표 ===")
    
    feature_names = ['혼잡도 (Occupancy)', '통행량 (Total Flow)']
    
    for i, feature_name in enumerate(feature_names):
        pred_feature = predictions[:, :, i].flatten()
        target_feature = targets[:, :, i].flatten()
        
        # 기본 지표
        mae = mean_absolute_error(target_feature, pred_feature)
        rmse = np.sqrt(mean_squared_error(target_feature, pred_feature))
        mape = np.mean(np.abs((target_feature - pred_feature) / (target_feature + 1e-8))) * 100
        r2 = r2_score(target_feature, pred_feature)
        
        print(f"\n🎯 {feature_name}:")
        print(f"   MAE:  {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   MAPE: {mape:.1f}%")
        print(f"   R²:   {r2:.3f}")
        
        # 추가 통계
        print(f"   실제값 범위: [{target_feature.min():.1f}, {target_feature.max():.1f}]")
        print(f"   예측값 범위: [{pred_feature.min():.1f}, {pred_feature.max():.1f}]")
    
    # 전체 요약
    overall_mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    overall_rmse = np.sqrt(mean_squared_error(targets.flatten(), predictions.flatten()))
    
    print(f"\n🏆 전체 성능:")
    print(f"   전체 MAE:  {overall_mae:.2f}")
    print(f"   전체 RMSE: {overall_rmse:.2f}")
    
    # 훈련 진행도
    total_epochs = checkpoint['config']['training']['epochs']
    current_epoch = checkpoint['epoch']
    progress = (current_epoch / total_epochs) * 100
    
    print(f"\n📚 훈련 현황:")
    print(f"   현재 에포크: {current_epoch}/{total_epochs} ({progress:.1f}%)")
    print(f"   학습률: {checkpoint.get('lr', 'N/A')}")
    print(f"   모델 파라미터: {sum(p.numel() for p in model.parameters()):,}개")
    
    return predictions, targets

if __name__ == '__main__':
    try:
        evaluate_performance()
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
