#!/usr/bin/env python3
"""
간단한 모델 평가 스크립트
"""

import sys
import pickle
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN
from models.dataset import SubwayGraphDataset

def load_model(checkpoint_path):
    """모델 로드"""
    print(f"Loading model from {checkpoint_path}")
    
    device = torch.device('cpu')  # CPU 사용
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
    
    print(f"Model loaded successfully (epoch {checkpoint['epoch']})")
    return model, device

def load_test_data():
    """테스트 데이터 로드"""
    print("Loading test data...")
    
    try:
        with open('data/test_dataset.pkl', 'rb') as f:
            test_dataset = pickle.load(f)
        print(f"Test dataset loaded: {len(test_dataset)} samples")
        return test_dataset
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return None

def evaluate_model():
    """모델 평가"""
    # 모델 로드
    model, device = load_model('checkpoints/best_model.pth')
    
    # 테스트 데이터 로드
    test_dataset = load_test_data()
    if test_dataset is None:
        print("Cannot load test dataset. Exiting...")
        return
    
    # 인접행렬 로드
    adjacency = torch.FloatTensor(np.load('data/adjacency_matrix.npy')).to(device)
    
    print("Starting evaluation...")
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(min(100, len(test_dataset))):  # 처음 100개 샘플만 평가
            try:
                sample = test_dataset[i]
                
                # 데이터 준비
                X = sample['X'].unsqueeze(0).to(device)  # [1, T, N, 2]
                Y = sample['Y'].unsqueeze(0).to(device)  # [1, N, 2]
                
                batch_size = X.size(0)
                adj_batch = adjacency.unsqueeze(0).expand(batch_size, -1, -1)
                
                # 예측
                predictions, _ = model(X, adj_batch, target_length=1)
                predictions = predictions.squeeze(1)  # [1, N, 2]
                
                # CPU로 이동
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(Y.cpu().numpy())
                
                if i % 20 == 0:
                    print(f"Processed {i+1} samples...")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    
    if not all_predictions:
        print("No valid predictions generated!")
        return
    
    # 결합
    predictions = np.concatenate(all_predictions, axis=0)  # [N_samples, N_stations, 2]
    targets = np.concatenate(all_targets, axis=0)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # 지표 계산
    print("\n=== Evaluation Results ===")
    
    for i, feature_name in enumerate(['occupancy', 'total_flow']):
        pred_feature = predictions[:, :, i].flatten()
        target_feature = targets[:, :, i].flatten()
        
        # 유효한 값만 사용 (0이 아닌 값)
        valid_mask = target_feature != 0
        if valid_mask.sum() > 0:
            pred_valid = pred_feature[valid_mask]
            target_valid = target_feature[valid_mask]
            
            mae = mean_absolute_error(target_valid, pred_valid)
            rmse = np.sqrt(mean_squared_error(target_valid, pred_valid))
            mape = np.mean(np.abs((target_valid - pred_valid) / target_valid)) * 100
            r2 = r2_score(target_valid, pred_valid)
            
            print(f"\n{feature_name.upper()}:")
            print(f"  MAE:  {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  R²:   {r2:.4f}")
            print(f"  Valid samples: {len(pred_valid)}")
        else:
            print(f"\n{feature_name.upper()}: No valid data")
    
    # 전체 통계
    print(f"\nTotal samples evaluated: {len(predictions)}")
    print(f"Total stations: {predictions.shape[1]}")
    
    return predictions, targets

if __name__ == '__main__':
    evaluate_model()
