#!/usr/bin/env python3
"""
직접 데이터 로드하여 모델 평가
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
    return model, device, checkpoint

def inspect_dataset():
    """데이터셋 구조 확인"""
    print("Inspecting dataset structure...")
    
    try:
        with open('data/test_dataset.pkl', 'rb') as f:
            test_dataset = pickle.load(f)
        
        print(f"Dataset type: {type(test_dataset)}")
        print(f"Dataset length: {len(test_dataset)}")
        
        # 첫 번째 샘플 확인
        sample = test_dataset[0]
        print(f"Sample type: {type(sample)}")
        
        if isinstance(sample, dict):
            print("Sample keys:", list(sample.keys()))
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
        
        return test_dataset
        
    except Exception as e:
        print(f"Error inspecting dataset: {e}")
        return None

def evaluate_with_raw_data():
    """원시 데이터로 평가"""
    # 모델 로드
    model, device, checkpoint = load_model('checkpoints/best_model.pth')
    
    # 데이터셋 구조 확인
    test_dataset = inspect_dataset()
    if test_dataset is None:
        return
    
    # 인접행렬 로드
    adjacency = torch.FloatTensor(np.load('data/adjacency_matrix.npy')).to(device)
    
    print("\nStarting evaluation...")
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(min(50, len(test_dataset))):  # 처음 50개 샘플만 평가
            try:
                sample = test_dataset[i]
                
                # 샘플이 딕셔너리인 경우
                if isinstance(sample, dict):
                    if 'X' in sample and 'Y' in sample:
                        X = sample['X']
                        Y = sample['Y']
                    else:
                        print(f"Sample {i}: Missing X or Y keys")
                        continue
                else:
                    # 튜플이나 리스트인 경우
                    if len(sample) >= 2:
                        X, Y = sample[0], sample[1]
                    else:
                        print(f"Sample {i}: Invalid sample format")
                        continue
                
                # 텐서로 변환 및 차원 확인
                if not isinstance(X, torch.Tensor):
                    X = torch.FloatTensor(X)
                if not isinstance(Y, torch.Tensor):
                    Y = torch.FloatTensor(Y)
                
                # 배치 차원 추가
                if X.dim() == 3:  # [T, N, F]
                    X = X.unsqueeze(0)  # [1, T, N, F]
                if Y.dim() == 2:  # [N, F]
                    Y = Y.unsqueeze(0)  # [1, N, F]
                
                X = X.to(device)
                Y = Y.to(device)
                
                print(f"Sample {i}: X shape: {X.shape}, Y shape: {Y.shape}")
                
                # 예측
                batch_size = X.size(0)
                adj_batch = adjacency.unsqueeze(0).expand(batch_size, -1, -1)
                
                predictions, _ = model(X, adj_batch, target_length=1)
                predictions = predictions.squeeze(1)  # [1, N, 2]
                
                # CPU로 이동
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(Y.cpu().numpy())
                
                if i % 10 == 0:
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
    
    print(f"\nPredictions shape: {predictions.shape}")
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
            
            # 기본 통계
            print(f"  Target range: [{target_valid.min():.2f}, {target_valid.max():.2f}]")
            print(f"  Prediction range: [{pred_valid.min():.2f}, {pred_valid.max():.2f}]")
        else:
            print(f"\n{feature_name.upper()}: No valid data")
    
    # 전체 통계
    print(f"\nTotal samples evaluated: {len(predictions)}")
    print(f"Total stations: {predictions.shape[1]}")
    
    return predictions, targets

if __name__ == '__main__':
    evaluate_with_raw_data()
