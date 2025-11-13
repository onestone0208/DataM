#!/usr/bin/env python3
"""
계절별 균등 분할로 훈련된 혼잡도 전용 DCRNN 모델 평가 스크립트

훈련과 동일한 방식으로 Test 데이터를 사용하여 평가
"""

import os
import sys
import argparse
import yaml
import pickle
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN
from scripts.train_occupancy_model import OccupancyDataset


class SeasonalOccupancyEvaluator:
    """계절별 균등 분할 혼잡도 모델 평가 클래스"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config_path = config_path
        self.device = torch.device('cpu')  # 평가는 CPU에서
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 데이터 로드
        self.congestion_data = pd.read_csv('data/혼잡도.csv')
        self.node_features = pd.read_csv('data/node_features.csv')
        self.date_features = pd.read_csv('data/date_features.csv')
        self.time_features = pd.read_csv('data/time_features.csv')
        self.adjacency_matrix = np.load('data/adjacency_matrix.npy')
        
        stations = sorted(self.congestion_data['station'].unique())
        self.station_to_idx = {station: idx for idx, station in enumerate(stations)}
        
        print("🔍 계절별 균등 분할 혼잡도 모델 평가 시작")
        print("=" * 60)
    
    def load_seasonal_split(self) -> Dict[str, List[str]]:
        """저장된 계절별 분할 정보 로드"""
        split_file = "data/seasonal_split.pkl"
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"분할 정보 파일이 없습니다: {split_file}")
        
        with open(split_file, 'rb') as f:
            split_data = pickle.load(f)
        
        print(f"📂 분할 정보 로드: {split_file}")
        print(f"  분할 방식: {split_data['split_method']}")
        print(f"  시드: {split_data['seed']}")
        print(f"  Train: {len(split_data['train_dates'])}일")
        print(f"  Val: {len(split_data['val_dates'])}일")
        print(f"  Test: {len(split_data['test_dates'])}일")
        
        return split_data
    
    def load_test_dataset(self) -> Tuple[DataLoader, OccupancyDataset]:
        """Test 데이터셋 로드"""
        print("\n📊 Test 데이터셋 준비 중...")
        
        # 분할 정보 로드
        split_data = self.load_seasonal_split()
        test_dates = split_data['test_dates']
        train_dates = split_data['train_dates']  # 정규화 통계용
        
        # 데이터 로드
        congestion_data = pd.read_csv('data/혼잡도.csv')
        node_features = pd.read_csv('data/node_features.csv')
        date_features = pd.read_csv('data/date_features.csv')
        time_features = pd.read_csv('data/time_features.csv')
        adjacency_matrix = np.load('data/adjacency_matrix.npy')
        
        # 역 인덱스 매핑
        stations = sorted(congestion_data['station'].unique())
        station_to_idx = {station: idx for idx, station in enumerate(stations)}
        
        # Train 데이터로 정규화 통계 계산
        train_data = congestion_data[congestion_data['date'].isin(train_dates)]
        train_dataset = OccupancyDataset(
            train_data, node_features, date_features, time_features,
            adjacency_matrix, station_to_idx, 
            sequence_length=12, prediction_length=1,
            target_columns=['occupancy']
        )
        
        # Test 데이터셋 생성 (Train 정규화 통계 사용)
        test_data = congestion_data[congestion_data['date'].isin(test_dates)]
        normalization_stats = (train_dataset.log_mean, train_dataset.log_std)
        test_dataset = OccupancyDataset(
            test_data, node_features, date_features, time_features,
            adjacency_matrix, station_to_idx, 
            sequence_length=12, prediction_length=1,
            target_columns=['occupancy'],
            normalization_stats=normalization_stats
        )
        
        # DataLoader 생성 (훈련 코드와 동일한 설정 사용)
        data_config = self.config['data']
        test_loader = DataLoader(
            test_dataset,
            batch_size=data_config['batch_size'],  # 🔧 훈련과 동일 (16)
            shuffle=False,
            num_workers=data_config['num_workers'],  # 🔧 훈련과 동일 (2)
            pin_memory=data_config['pin_memory']  # 🔧 훈련과 동일
        )
        
        print(f"✅ Test 데이터셋 준비 완료")
        print(f"  Test 날짜 범위: {min(test_dates)} ~ {max(test_dates)}")
        print(f"  Test 배치 수: {len(test_loader)}")
        
        return test_loader, test_dataset
    
    def load_model(self) -> DCRNN:
        """최신 체크포인트에서 모델 로드"""
        checkpoint_dir = Path("checkpoints/occupancy_model")
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"체크포인트 디렉토리가 없습니다: {checkpoint_dir}")
        
        # 🔥 무조건 Best 모델 사용 (성능이 가장 좋은 모델)
        best_model = checkpoint_dir / "best_occupancy_model.pth"
        
        if not best_model.exists():
            raise FileNotFoundError(f"best_occupancy_model.pth not found in {checkpoint_dir}")
        
        latest_checkpoint = best_model
        
        print(f"\n📂 모델 로드: {latest_checkpoint}")
        
        checkpoint = torch.load(latest_checkpoint, map_location=self.device)
        model_config = checkpoint['config']['model']
        
        print(f"📋 모델 정보:")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Config Input Size: {model_config['input_size']}")
        print(f"  Config Output Size: {model_config['output_size']}")
        
        # 🔧 실제 가중치 차원에 맞게 모델 생성 (47차원 입력, 1차원 출력)
        model = DCRNN(
            input_size=47,  # 🔧 시계열(혼잡도+승하차+시간+날짜)(31) + 노드(16) = 47차원
            hidden_size=model_config['hidden_size'],
            output_size=1,  # 혼잡도만 출력
            num_layers=model_config['num_layers'],
            diffusion_steps=model_config['diffusion_steps'],
            use_attention=model_config['use_attention'],
            dropout=model_config['dropout']
        ).to(self.device)
        
        print(f"  실제 사용: input_size=47, output_size=1")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def evaluate(self) -> Dict[str, float]:
        """모델 평가 실행"""
        print("\n🎯 모델 평가 시작...")
        
        # 데이터 및 모델 로드
        test_loader, test_dataset = self.load_test_dataset()
        model = self.load_model()
        adjacency_matrix = np.load('data/adjacency_matrix.npy')
        adjacency_tensor = torch.FloatTensor(adjacency_matrix).to(self.device)
        
        all_predictions = []
        all_targets = []
        
        print(f"\n📊 배치별 평가 진행...")
        
        sample_details = []  # 🔥 구체적인 예측 사례 저장
        
        # 역 이름 매핑은 이미 self.station_to_idx로 초기화됨
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # 데이터 준비
                # 데이터 준비 (훈련 코드와 동일)
                X = batch['X'].to(self.device)  # [B, T, N, 31] - 혼잡도+승하차+시간+날짜
                Y_raw = batch['Y_raw'].to(self.device)  # [B, N, 1]
                
                # 추가 특성들 추출 (시간/날짜 특성은 이미 X에 포함됨)
                node_features = batch['node_features'].to(self.device)
                # date_features와 time_features는 이미 X에 포함되어 있음
                
                # 인접행렬
                batch_size = X.size(0)
                adjacency = adjacency_tensor.unsqueeze(0).expand(batch_size, -1, -1)
                
                # 🔧 평가 시에는 Teacher Forcing 사용 안함 (실제 예측 능력 측정)
                predictions, _ = model(
                    X, adjacency,
                    target_length=1,
                    teacher_forcing_ratio=0.0,  # 평가 시 0%
                    node_features=node_features,
                    date_features=None,  # 🔥 날짜 특성은 X에 포함되어 있음
                    time_features=None  # 🔥 시간 특성은 X에 포함되어 있음
                )
                predictions = predictions.squeeze(1)  # [B, N, 1]
                
                # 🔍 디버깅: 첫 배치에서만 출력
                if batch_idx == 0:
                    print(f"\n🔍 디버깅 정보 (첫 배치):")
                    print(f"  predictions shape: {predictions.shape}")
                    print(f"  predictions range (정규화): [{predictions.min().item():.3f}, {predictions.max().item():.3f}]")
                    print(f"  predictions mean: {predictions.mean().item():.3f}, std: {predictions.std().item():.3f}")
                
                # 정규화 해제
                pred_raw = test_dataset.inverse_transform(predictions)
                
                # 🔍 디버깅: 첫 배치에서만 출력
                if batch_idx == 0:
                    print(f"  pred_raw range (역변환): [{pred_raw.min().item():.1f}, {pred_raw.max().item():.1f}]명")
                    print(f"  pred_raw mean: {pred_raw.mean().item():.1f}명")
                
                # 결과 저장
                all_predictions.extend(pred_raw.cpu().numpy().flatten())
                all_targets.extend(Y_raw.cpu().numpy().flatten())
                
                # 🔥 구체적인 예측 사례 수집 (처음 3개 배치만)
                if batch_idx < 3:
                    for sample_idx in range(min(batch_size, 2)):  # 배치당 최대 2개 샘플
                        sample_date = batch['metadata']['date'][sample_idx]
                        sample_hour = batch['metadata']['hour'][sample_idx]
                        
                        # 주요 역들만 선택 (5개 역)
                        major_stations = ['대전역', '중앙로역', '서대전네거리역', '정부청사역', '갈마역']
                        for station_name in major_stations:
                            if station_name in self.station_to_idx:
                                station_idx = self.station_to_idx[station_name]
                                actual = Y_raw[sample_idx, station_idx, 0].item()
                                predicted = pred_raw[sample_idx, station_idx].item()
                                
                                sample_details.append({
                                    'date': sample_date,
                                    'hour': sample_hour,
                                    'station': station_name,
                                    'actual': actual,
                                    'predicted': predicted,
                                    'error': abs(actual - predicted),
                                    'error_pct': (abs(actual - predicted) / max(actual, 1)) * 100
                                })
                
                if batch_idx % 10 == 0:
                    print(f"  배치 {batch_idx+1}/{len(test_loader)} 완료")
        
        # 성능 지표 계산
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        metrics = self.compute_metrics(predictions, targets)
        self.print_results(metrics, predictions, targets, sample_details)
        
        return metrics
    
    def compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """성능 지표 계산"""
        # MAE
        mae = np.mean(np.abs(predictions - targets))
        
        # RMSE
        rmse = np.sqrt(np.mean((predictions - targets)**2))
        
        # 🔧 MAPE: 20명 이상일 때만 계산 (0명 근처에서 폭발 방지)
        mask = targets > 20
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((predictions[mask] - targets[mask]) / targets[mask])) * 100
            mape_count = np.sum(mask)
        else:
            mape = 0.0
            mape_count = 0
        
        # R2 (결정계수)
        ss_total = np.sum((targets - np.mean(targets))**2)
        ss_residual = np.sum((targets - predictions)**2)
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0
        
        return {
            'mae': mae,
            'rmse': rmse, 
            'mape': mape,
            'mape_count': mape_count,
            'r2': r2
        }
    
    def print_results(self, metrics: Dict[str, float], predictions: np.ndarray, targets: np.ndarray, sample_details: list):
        """결과 출력"""
        print("\n" + "=" * 60)
        print("🎉 계절별 균등 분할 모델 평가 결과")
        print("=" * 60)
        print(f"MAE:  {metrics['mae']:.2f}명")
        print(f"RMSE: {metrics['rmse']:.2f}명")
        print(f"MAPE: {metrics['mape']:.1f}% (20명 이상 {metrics['mape_count']}개 샘플 기준)")
        print(f"R²:   {metrics['r2']:.3f}")
        print()
        print(f"📈 예측 범위: [{predictions.min():.1f}, {predictions.max():.1f}]명")
        print(f"📈 실제 범위: [{targets.min():.1f}, {targets.max():.1f}]명")
        print(f"📊 총 예측 수: {len(predictions):,}개")
        
        # 🔥 구체적인 예측 사례 출력
        if sample_details:
            print(f"\n🔍 구체적인 예측 사례 (샘플 {len(sample_details)}개):")
            print("-" * 80)
            
            # 오차가 큰 순서로 정렬해서 상위 15개만 출력
            sample_details_sorted = sorted(sample_details, key=lambda x: x['error'], reverse=True)
            
            for i, detail in enumerate(sample_details_sorted[:15]):
                date_str = detail['date']
                hour = detail['hour']
                station = detail['station']
                actual = detail['actual']
                predicted = detail['predicted']
                error = detail['error']
                error_pct = detail['error_pct']
                
                print(f"{i+1:2d}. {date_str} {hour:2d}시 {station:8s} | "
                      f"실제: {actual:5.1f}명 | 예측: {predicted:5.1f}명 | "
                      f"오차: {error:5.1f}명 ({error_pct:4.1f}%)")
            
            # 통계 요약
            avg_error = np.mean([d['error'] for d in sample_details])
            avg_error_pct = np.mean([d['error_pct'] for d in sample_details])
            print(f"\n📊 샘플 평균 오차: {avg_error:.1f}명 ({avg_error_pct:.1f}%)")
        
        print(f"\n✅ 평가 완료!")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Seasonal Occupancy DCRNN model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml', help='Config file path')
    args = parser.parse_args()
    
    evaluator = SeasonalOccupancyEvaluator(args.config)
    metrics = evaluator.evaluate()
    
    print("\n✅ 평가 완료!")


if __name__ == "__main__":
    main()
