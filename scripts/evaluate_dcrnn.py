#!/usr/bin/env python3
"""
DCRNN 모델 평가 및 추론 스크립트

사용법:
    python scripts/evaluate_dcrnn.py --config config/model_config.yaml --checkpoint checkpoints/best_model.pth
"""

import os
import sys
import argparse
import yaml
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN
from models.dataset import SubwayGraphDataset


class DCRNNEvaluator:
    """DCRNN 모델 평가 클래스"""
    
    def __init__(self, config: Dict[str, Any], checkpoint_path: str):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 로드
        self.model = self._load_model(checkpoint_path)
        self.adjacency_matrix = self._load_adjacency()
        
        # 데이터 로드
        self.test_loader = self._load_test_data()
        
        # 결과 저장 디렉토리
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 역 이름 매핑 (인덱스 → 역명)
        self.station_names = [
            "판암", "신흥", "대동", "대전역", "중구청", "서대전네거리", "오룡", "용문",
            "탄방", "시청", "정부청사", "갈마", "월평", "갑천", "의회.충남도청",
            "반석", "학하", "반석", "지족", "노은", "월드컵경기장", "현충원"
        ]
    
    def _load_model(self, checkpoint_path: str) -> DCRNN:
        """모델 로드"""
        print(f"Loading model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_config = checkpoint['config']['model']
        
        model = DCRNN(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            output_size=model_config['output_size'],
            num_layers=model_config['num_layers'],
            diffusion_steps=model_config['diffusion_steps'],
            use_attention=model_config['use_attention'],
            dropout=model_config['dropout']
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded successfully (epoch {checkpoint['epoch']})")
        return model
    
    def _load_adjacency(self) -> torch.Tensor:
        """인접행렬 로드"""
        adjacency_path = self.config['data']['adjacency_path']
        adjacency = torch.FloatTensor(np.load(adjacency_path)).to(self.device)
        return adjacency
    
    def _load_test_data(self) -> DataLoader:
        """테스트 데이터 로드"""
        with open(self.config['data']['test_path'], 'rb') as f:
            test_dataset = pickle.load(f)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory']
        )
        
        return test_loader
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """전체 테스트 데이터에 대한 예측"""
        print("Generating predictions...")
        
        all_predictions = []
        all_targets = []
        all_attention_weights = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                X = batch['X'].to(self.device)  # [B, T, N, 2]
                Y = batch['Y'].to(self.device)  # [B, N, 2]
                
                batch_size = X.size(0)
                adjacency = self.adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
                
                # 예측
                predictions, attention_weights = self.model(X, adjacency, target_length=1)
                predictions = predictions.squeeze(1)  # [B, N, 2]
                
                # CPU로 이동
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(Y.cpu().numpy())
                
                if attention_weights is not None:
                    all_attention_weights.append(attention_weights.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(self.test_loader)} batches")
        
        # 결합
        predictions = np.concatenate(all_predictions, axis=0)  # [N_samples, N_stations, 2]
        targets = np.concatenate(all_targets, axis=0)
        
        if all_attention_weights:
            attention_weights = np.concatenate(all_attention_weights, axis=0)
        else:
            attention_weights = None
        
        print(f"Predictions shape: {predictions.shape}")
        return predictions, targets, attention_weights
    
    def compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Dict[str, float]]:
        """평가 지표 계산"""
        print("Computing evaluation metrics...")
        
        metrics = {}
        
        # 전체 지표
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
                
                metrics[feature_name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'r2': r2
                }
            else:
                metrics[feature_name] = {
                    'mae': np.nan,
                    'rmse': np.nan,
                    'mape': np.nan,
                    'r2': np.nan
                }
        
        # 역별 지표 (혼잡도만)
        station_metrics = {}
        for station_idx in range(predictions.shape[1]):
            station_name = self.station_names[station_idx] if station_idx < len(self.station_names) else f"Station_{station_idx}"
            
            pred_station = predictions[:, station_idx, 0]  # occupancy
            target_station = targets[:, station_idx, 0]
            
            valid_mask = target_station != 0
            if valid_mask.sum() > 0:
                pred_valid = pred_station[valid_mask]
                target_valid = target_station[valid_mask]
                
                mae = mean_absolute_error(target_valid, pred_valid)
                rmse = np.sqrt(mean_squared_error(target_valid, pred_valid))
                
                station_metrics[station_name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'samples': len(pred_valid)
                }
        
        return {
            'overall': metrics,
            'by_station': station_metrics
        }
    
    def plot_predictions(self, predictions: np.ndarray, targets: np.ndarray, num_samples: int = 5):
        """예측 결과 시각화"""
        print("Creating prediction plots...")
        
        plot_dir = self.results_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # 샘플 선택
        sample_indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)
        
        for sample_idx in sample_indices:
            pred_sample = predictions[sample_idx]  # [N_stations, 2]
            target_sample = targets[sample_idx]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Prediction vs Target - Sample {sample_idx}', fontsize=16)
            
            # Occupancy 산점도
            axes[0, 0].scatter(target_sample[:, 0], pred_sample[:, 0], alpha=0.6)
            axes[0, 0].plot([target_sample[:, 0].min(), target_sample[:, 0].max()], 
                           [target_sample[:, 0].min(), target_sample[:, 0].max()], 'r--')
            axes[0, 0].set_xlabel('True Occupancy')
            axes[0, 0].set_ylabel('Predicted Occupancy')
            axes[0, 0].set_title('Occupancy Prediction')
            
            # Total Flow 산점도
            axes[0, 1].scatter(target_sample[:, 1], pred_sample[:, 1], alpha=0.6)
            axes[0, 1].plot([target_sample[:, 1].min(), target_sample[:, 1].max()], 
                           [target_sample[:, 1].min(), target_sample[:, 1].max()], 'r--')
            axes[0, 1].set_xlabel('True Total Flow')
            axes[0, 1].set_ylabel('Predicted Total Flow')
            axes[0, 1].set_title('Total Flow Prediction')
            
            # 역별 혼잡도 비교
            station_indices = range(len(self.station_names))
            axes[1, 0].bar([i - 0.2 for i in station_indices], target_sample[:, 0], 
                          width=0.4, label='True', alpha=0.7)
            axes[1, 0].bar([i + 0.2 for i in station_indices], pred_sample[:, 0], 
                          width=0.4, label='Predicted', alpha=0.7)
            axes[1, 0].set_xlabel('Station Index')
            axes[1, 0].set_ylabel('Occupancy')
            axes[1, 0].set_title('Station-wise Occupancy')
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 오차 분석
            errors = np.abs(pred_sample[:, 0] - target_sample[:, 0])
            axes[1, 1].bar(station_indices, errors)
            axes[1, 1].set_xlabel('Station Index')
            axes[1, 1].set_ylabel('Absolute Error')
            axes[1, 1].set_title('Prediction Error by Station')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(plot_dir / f'prediction_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_attention_weights(self, attention_weights: np.ndarray, num_samples: int = 3):
        """Attention 가중치 시각화"""
        if attention_weights is None:
            print("No attention weights available")
            return
        
        print("Creating attention plots...")
        
        plot_dir = self.results_dir / "attention"
        plot_dir.mkdir(exist_ok=True)
        
        sample_indices = np.random.choice(len(attention_weights), min(num_samples, len(attention_weights)), replace=False)
        
        for sample_idx in sample_indices:
            attention_sample = attention_weights[sample_idx]  # [target_length, seq_len]
            
            plt.figure(figsize=(12, 6))
            
            # Heatmap
            sns.heatmap(attention_sample, 
                       xticklabels=[f't-{11-i}' for i in range(12)],
                       yticklabels=[f'pred_t+{i}' for i in range(attention_sample.shape[0])],
                       cmap='Blues', annot=True, fmt='.3f')
            
            plt.title(f'Attention Weights - Sample {sample_idx}')
            plt.xlabel('Input Time Steps')
            plt.ylabel('Output Time Steps')
            
            plt.tight_layout()
            plt.savefig(plot_dir / f'attention_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def analyze_errors(self, predictions: np.ndarray, targets: np.ndarray):
        """오차 분석"""
        print("Analyzing prediction errors...")
        
        analysis_dir = self.results_dir / "error_analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # 혼잡도 오차 분석
        occupancy_pred = predictions[:, :, 0]
        occupancy_target = targets[:, :, 0]
        occupancy_errors = np.abs(occupancy_pred - occupancy_target)
        
        # 역별 평균 오차
        station_errors = np.mean(occupancy_errors, axis=0)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(station_errors)), station_errors)
        plt.xlabel('Station Index')
        plt.ylabel('Mean Absolute Error')
        plt.title('Average Prediction Error by Station')
        plt.xticks(range(len(self.station_names)), 
                  [name[:8] + '...' if len(name) > 8 else name for name in self.station_names], 
                  rotation=45)
        
        # 색상 구분 (오차가 큰 역 강조)
        max_error = max(station_errors)
        for i, bar in enumerate(bars):
            if station_errors[i] > max_error * 0.8:
                bar.set_color('red')
            elif station_errors[i] > max_error * 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('blue')
        
        plt.tight_layout()
        plt.savefig(analysis_dir / 'station_errors.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 오차 분포
        plt.figure(figsize=(10, 6))
        plt.hist(occupancy_errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        plt.axvline(np.mean(occupancy_errors), color='red', linestyle='--', label=f'Mean: {np.mean(occupancy_errors):.2f}')
        plt.axvline(np.median(occupancy_errors), color='green', linestyle='--', label=f'Median: {np.median(occupancy_errors):.2f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(analysis_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, metrics: Dict, predictions: np.ndarray, targets: np.ndarray):
        """결과 저장"""
        print("Saving results...")
        
        # 지표 저장
        with open(self.results_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # 예측 결과 저장
        np.save(self.results_dir / 'predictions.npy', predictions)
        np.save(self.results_dir / 'targets.npy', targets)
        
        # CSV 형태로도 저장
        results_df = []
        for sample_idx in range(len(predictions)):
            for station_idx in range(predictions.shape[1]):
                station_name = self.station_names[station_idx] if station_idx < len(self.station_names) else f"Station_{station_idx}"
                
                results_df.append({
                    'sample_idx': sample_idx,
                    'station': station_name,
                    'station_idx': station_idx,
                    'pred_occupancy': predictions[sample_idx, station_idx, 0],
                    'true_occupancy': targets[sample_idx, station_idx, 0],
                    'pred_flow': predictions[sample_idx, station_idx, 1],
                    'true_flow': targets[sample_idx, station_idx, 1],
                    'occupancy_error': abs(predictions[sample_idx, station_idx, 0] - targets[sample_idx, station_idx, 0]),
                    'flow_error': abs(predictions[sample_idx, station_idx, 1] - targets[sample_idx, station_idx, 1])
                })
        
        results_df = pd.DataFrame(results_df)
        results_df.to_csv(self.results_dir / 'detailed_results.csv', index=False)
        
        print(f"Results saved to {self.results_dir}")
    
    def evaluate(self):
        """전체 평가 실행"""
        print("Starting evaluation...")
        
        # 예측 생성
        predictions, targets, attention_weights = self.predict()
        
        # 지표 계산
        metrics = self.compute_metrics(predictions, targets)
        
        # 결과 출력
        print("\n=== Evaluation Results ===")
        for feature, feature_metrics in metrics['overall'].items():
            print(f"\n{feature.upper()}:")
            for metric_name, value in feature_metrics.items():
                print(f"  {metric_name.upper()}: {value:.4f}")
        
        # 시각화
        eval_config = self.config.get('evaluation', {})
        viz_config = eval_config.get('visualization', {})
        
        if viz_config.get('enabled', True):
            # 예측 결과 플롯
            if viz_config.get('prediction_plots', {}).get('enabled', True):
                num_samples = viz_config.get('prediction_plots', {}).get('num_samples', 5)
                self.plot_predictions(predictions, targets, num_samples)
            
            # Attention 플롯
            if viz_config.get('attention_plots', True):
                self.plot_attention_weights(attention_weights)
            
            # 오차 분석
            if viz_config.get('error_analysis', {}).get('enabled', True):
                self.analyze_errors(predictions, targets)
        
        # 결과 저장
        self.save_results(metrics, predictions, targets)
        
        print("\nEvaluation completed!")
        return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate DCRNN model')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 평가기 생성 및 실행
    evaluator = DCRNNEvaluator(config, args.checkpoint)
    metrics = evaluator.evaluate()
    
    return metrics


if __name__ == '__main__':
    main()
