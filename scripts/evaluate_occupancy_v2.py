#!/usr/bin/env python3
"""
혼잡도 전용 모델 평가 스크립트 (v2)
- 45차원 입력 모델 지원
- 정규화 통계 공유
- 상세한 성능 지표
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN

# 🔧 OccupancyDataset을 직접 정의 (훈련 스크립트와 동일)

# OccupancyDataset 클래스 정의 (훈련 스크립트와 동일)
from models.dataset import SubwayGraphDataset

class OccupancyDataset(SubwayGraphDataset):
    """혼잡도 전용 데이터셋 (Log1p + 정규화 적용)"""
    
    def __init__(self, *args, normalization_stats=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        if normalization_stats is not None:
            # 🔧 외부에서 전달받은 정규화 통계 사용 (val/test용)
            self.log_mean, self.log_std = normalization_stats
            print(f"  📊 외부 정규화 통계 사용: mean={self.log_mean:.3f}, std={self.log_std:.3f}")
        else:
            # 🔧 자체 데이터로 정규화 통계 계산 (train용)
            self._compute_normalization_stats()
        
    def _compute_normalization_stats(self):
        """전체 데이터에서 log1p + 정규화 통계 계산"""
        print("📊 Log1p + 정규화 통계 계산 중...")
        
        all_occupancy = []
        for i in range(len(self.sequences)):
            sample = super().__getitem__(i)
            
            # X에서 혼잡도 추출 [T, N, 1]
            x_occ = sample['X'][:, :, 0].numpy()  # [T, N]
            all_occupancy.extend(x_occ.flatten())
            
            # Y에서 혼잡도 추출 [N, 1]  
            y_occ = sample['Y'][:, 0].numpy()  # [N]
            all_occupancy.extend(y_occ.flatten())
        
        all_occupancy = np.array(all_occupancy)
        
        # Log1p 변환
        log_occupancy = np.log1p(all_occupancy)
        
        # 정규화 통계
        self.log_mean = log_occupancy.mean()
        self.log_std = log_occupancy.std()
        
        print(f"  📈 원본 범위: [{all_occupancy.min():.1f}, {all_occupancy.max():.1f}]명")
        print(f"  📈 Log1p 범위: [{log_occupancy.min():.3f}, {log_occupancy.max():.3f}]")
        print(f"  📈 정규화 통계: mean={self.log_mean:.3f}, std={self.log_std:.3f}")
        
    def _transform_occupancy(self, occ_tensor):
        """혼잡도를 log1p + 정규화 변환"""
        # log1p 변환
        log_occ = torch.log1p(occ_tensor)
        
        # 정규화
        norm_occ = (log_occ - self.log_mean) / self.log_std
        
        return norm_occ
    
    def inverse_transform(self, norm_tensor):
        """정규화된 값을 원본 혼잡도로 역변환"""
        # 정규화 해제
        log_tensor = norm_tensor * self.log_std + self.log_mean
        
        # expm1 (log1p의 역함수)
        occ_tensor = torch.expm1(log_tensor)
        
        return occ_tensor
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        
        # 🔥 혼잡도만 사용 (1차원 입력)
        X_full = sample['X']  # [T, N, F] - 모든 특성
        Y_occupancy = sample['Y'][:, 0:1]  # [N, 1] - 혼잡도만 예측
        
        # 🔥 혼잡도 채널만 추출하고 Log1p + 정규화 적용
        X_occupancy = X_full[:, :, 0:1]  # [T, N, 1] - 혼잡도 채널만
        X_occupancy_transformed = self._transform_occupancy(X_occupancy)
        
        # 🔥 타겟은 혼잡도만 Log1p + 정규화
        Y_transformed = self._transform_occupancy(Y_occupancy)
        
        sample['X'] = X_occupancy_transformed  # [T, N, 1] - 혼잡도만
        sample['Y'] = Y_transformed
        
        # 🔥 원본 값도 저장 (loss 계산용)
        sample['Y_raw'] = Y_occupancy  # 원본 혼잡도
        
        return sample

class OccupancyEvaluator:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config_path = config_path
        self.device = torch.device('cpu')
    
    def extract_continuous_sequence(self, congestion_data, target_date, target_hour, 
                                   node_features, date_features, time_features, station_to_idx):
        """🔧 특정 시점의 연속된 12시간 데이터 추출"""
        import pandas as pd
        from datetime import datetime, timedelta
        
        # 타겟 시점
        target_datetime = datetime.strptime(f"{target_date} {target_hour:02d}:00:00", "%Y-%m-%d %H:%M:%S")
        
        # 이전 12시간 시점들 생성
        sequence_hours = []
        for i in range(12, 0, -1):  # 12시간 전부터 1시간 전까지
            seq_datetime = target_datetime - timedelta(hours=i)
            sequence_hours.append((seq_datetime.strftime("%Y-%m-%d"), seq_datetime.hour))
        
        # 연속된 12시간 데이터 추출
        sequence_data = []
        for seq_date, seq_hour in sequence_hours:
            hour_data = congestion_data[
                (congestion_data['date'] == seq_date) & 
                (congestion_data['hour'] == seq_hour)
            ]
            if len(hour_data) == 0:
                print(f"⚠️ 데이터 없음: {seq_date} {seq_hour}시")
                return None
            sequence_data.append(hour_data)
        
        # 타겟 시점 데이터
        target_data = congestion_data[
            (congestion_data['date'] == target_date) & 
            (congestion_data['hour'] == target_hour)
        ]
        if len(target_data) == 0:
            print(f"⚠️ 타겟 데이터 없음: {target_date} {target_hour}시")
            return None
        
        # 🔧 OccupancyDataset과 동일한 방식으로 데이터 구성
        X_sequence = []
        for hour_data in sequence_data:
            # 각 시간의 모든 역 데이터를 순서대로 정렬
            hour_data_sorted = hour_data.sort_values('station')
            occupancy_values = hour_data_sorted['occupancy'].values
            total_flow_values = hour_data_sorted['total_flow'].values
            
            # [N, 2] 형태로 구성
            hour_features = np.column_stack([occupancy_values, total_flow_values])
            X_sequence.append(hour_features)
        
        # [T=12, N=22, F=2] 형태로 변환
        X = np.array(X_sequence)  # [12, 22, 2]
        
        # 타겟 데이터 (혼잡도만)
        target_data_sorted = target_data.sort_values('station')
        Y = target_data_sorted['occupancy'].values.reshape(-1, 1)  # [22, 1]
        
        # 추가 특성들 추출 (첫 번째 시간 기준)
        first_hour_data = sequence_data[0].iloc[0]
        
        # Node features
        node_feat = node_features.iloc[0, 1:].values.astype(float)  # 첫 번째 컬럼 제외
        node_features_tensor = torch.FloatTensor(node_feat).unsqueeze(0).repeat(22, 1)  # [22, node_dim]
        
        # Date features  
        date_feat = date_features[date_features['date'] == sequence_hours[0][0]].iloc[0, 1:].values.astype(float)
        date_features_tensor = torch.FloatTensor(date_feat)  # [date_dim]
        
        # Time features
        time_feat = time_features[time_features['hour'] == sequence_hours[0][1]].iloc[0, 1:].values.astype(float)
        time_features_tensor = torch.FloatTensor(time_feat)  # [time_dim]
        
        return {
            'X': torch.FloatTensor(X),  # [12, 22, 2]
            'Y': torch.FloatTensor(Y),  # [22, 1]
            'node_features': node_features_tensor,  # [22, node_dim]
            'date_features': date_features_tensor,  # [date_dim]
            'time_features': time_features_tensor,  # [time_dim]
            'metadata': {
                'date': target_date,
                'hour': target_hour,
                'sequence_hours': sequence_hours
            }
        }
        
        # 설정 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print("🔍 혼잡도 모델 평가 시작")
        print("="*60)
    
    def load_model(self) -> DCRNN:
        """최신 모델 로드"""
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
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        print(f"📋 모델 정보: Epoch {checkpoint['epoch']}")
        
        # 🔧 체크포인트에서 모델 설정 직접 사용
        model_config = checkpoint['config']['model']
        print(f"📋 체크포인트 모델 설정: input_size={model_config.get('input_size', 'N/A')}, output_size={model_config.get('output_size', 'N/A')}")
        
        model = DCRNN(
            input_size=45,  # 🔧 혼잡도 전용 모델: 45차원 입력
            hidden_size=model_config['hidden_size'],
            output_size=1,  # 🔧 혼잡도 전용 모델: 1차원 출력
            num_layers=model_config['num_layers'],
            diffusion_steps=model_config['diffusion_steps'],
            use_attention=model_config['use_attention'],
            dropout=model_config['dropout']
        ).to(self.device)
        
        # 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def load_test_data(self) -> Tuple[OccupancyDataset, torch.Tensor]:
        """🔧 훈련 스크립트와 동일한 방식으로 테스트 데이터 로드"""
        print("📊 테스트 데이터 로드 중...")
        
        # 🔧 훈련 스크립트와 동일한 데이터 로드 방식
        import pandas as pd
        
        congestion_data = pd.read_csv('data/혼잡도.csv')
        node_features = pd.read_csv('data/node_features.csv')
        date_features = pd.read_csv('data/date_features.csv')
        time_features = pd.read_csv('data/time_features.csv')
        adjacency_matrix = np.load('data/adjacency_matrix.npy')
        
        # 역 인덱스 매핑
        stations = sorted(congestion_data['station'].unique())
        station_to_idx = {station: idx for idx, station in enumerate(stations)}
        
        # 🔧 훈련 스크립트와 동일한 데이터 분할
        dates = sorted(congestion_data['date'].unique())
        n_dates = len(dates)
        
        train_end = int(n_dates * 0.7)
        val_end = int(n_dates * 0.85)
        
        train_dates = dates[:train_end]
        test_dates = dates[val_end:]  # 마지막 15% (실제 테스트 데이터)
        
        print(f"  📅 전체 날짜: {n_dates}일")
        print(f"  📅 훈련 날짜: {len(train_dates)}일 ({train_dates[0]} ~ {train_dates[-1]})")
        print(f"  📅 테스트 날짜: {len(test_dates)}일 ({test_dates[0]} ~ {test_dates[-1]})")
        
        # 🔧 Train 데이터로 정규화 통계 계산 (훈련과 동일)
        train_data = congestion_data[congestion_data['date'].isin(train_dates)]
        train_dataset = OccupancyDataset(
            train_data, node_features, date_features, time_features,
            adjacency_matrix, station_to_idx, 
            sequence_length=12, prediction_length=1,
            target_columns=['occupancy']  # 혼잡도만
        )
        
        # 🔧 Test 데이터 (Train 정규화 통계 공유)
        test_data = congestion_data[congestion_data['date'].isin(test_dates)]
        normalization_stats = (train_dataset.log_mean, train_dataset.log_std)
        test_dataset = OccupancyDataset(
            test_data, node_features, date_features, time_features,
            adjacency_matrix, station_to_idx, 
            sequence_length=12, prediction_length=1,
            target_columns=['occupancy'],  # 혼잡도만
            normalization_stats=normalization_stats  # 🔧 Train 통계 공유
        )
        
        adjacency_tensor = torch.FloatTensor(adjacency_matrix).to(self.device)
        
        print(f"  📈 Test 시퀀스 수: {len(test_dataset)}")
        print(f"  📈 정규화 통계: mean={normalization_stats[0]:.3f}, std={normalization_stats[1]:.3f}")
        
        return test_dataset, adjacency_tensor
    
    def compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """성능 지표 계산"""
        # MAE
        mae = np.mean(np.abs(predictions - targets))
        
        # RMSE
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        # MAPE (0이 아닌 값들만)
        non_zero_mask = targets != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((predictions[non_zero_mask] - targets[non_zero_mask]) / targets[non_zero_mask])) * 100
        else:
            mape = float('inf')
        
        # R²
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('-inf')
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
    
    def evaluate_model(self, num_samples: int = 100):
        """모델 평가 실행"""
        model = self.load_model()
        test_dataset, adjacency_tensor = self.load_test_data()
        
        print(f"\n🎯 {num_samples}개 샘플로 평가 시작...")
        
        all_predictions = []
        all_targets = []
        sample_details = []
        
        with torch.no_grad():
            for i in range(min(num_samples, len(test_dataset))):
                sample = test_dataset[i]
                
                # 🔧 훈련 스크립트와 동일한 데이터 준비 방식
                X = sample['X'].unsqueeze(0).to(self.device)  # [1, T, N, 1] - 혼잡도만
                Y_raw = sample['Y_raw'].to(self.device)  # [N, 1] - 원본 혼잡도 값
                
                # 인접행렬 확장
                batch_size = X.size(0)
                adjacency = adjacency_tensor.unsqueeze(0).expand(batch_size, -1, -1)
                
                # 🔧 추가 특성들 추출 (훈련과 동일)
                node_features = sample['node_features'].unsqueeze(0).to(self.device)  # [1, N, node_dim]
                date_features = sample['date_features'].unsqueeze(0).to(self.device)  # [1, date_dim]
                time_features = sample['time_features'].unsqueeze(0).to(self.device)  # [1, time_dim]
                
                # 🔧 예측 (훈련과 동일한 방식)
                predictions, _ = model(
                    X, adjacency, 
                    target_length=1,
                    node_features=node_features,
                    date_features=date_features,
                    time_features=time_features
                )
                predictions = predictions.squeeze(1)  # [1, N, 1] -> [N, 1]
                
                # 🔧 정규화 해제 (원본 값으로 변환) - 디버깅 정보 추가
                if i == 0:  # 첫 번째 샘플에서만 디버깅 출력
                    print(f"🔧 정규화 해제 디버깅:")
                    print(f"   예측값 shape: {predictions.shape}")
                    print(f"   정규화된 예측값 범위: [{predictions.min().item():.3f}, {predictions.max().item():.3f}]")
                    print(f"   정규화 통계: mean={test_dataset.log_mean:.3f}, std={test_dataset.log_std:.3f}")
                
                # 🔧 훈련 스크립트와 동일한 방식으로 역변환
                pred_raw = test_dataset.inverse_transform(predictions).squeeze()  # [22] 형태
                
                if i == 0:  # 첫 번째 샘플에서만 디버깅 출력
                    print(f"   역변환 후 shape: {pred_raw.shape}")
                    print(f"   역변환된 예측값 범위: [{pred_raw.min().item():.1f}, {pred_raw.max().item():.1f}]명")
                    print(f"   실제 타겟 범위: [{Y_raw.min().item():.1f}, {Y_raw.max().item():.1f}]명")
                
                # 결과 저장
                all_predictions.extend(pred_raw.cpu().numpy())
                all_targets.extend(Y_raw.squeeze().cpu().numpy())
                
                # 샘플 상세 정보 (처음 10개만)
                if i < 10:
                    metadata = sample['metadata']
                    
                    # 🔧 주요 역별 예측 결과
                    major_stations = ['대전역', '중앙로역', '서대전네거리역', '정부청사역', '갈마역']
                    station_to_idx = {
                        '판암역': 0, '신흥역': 1, '대동역': 2, '대전역': 3, '중앙로역': 4,
                        '중구청역': 5, '서대전네거리역': 6, '오룡역': 7, '용문역': 8, '탄방역': 9,
                        '시청역': 10, '정부청사역': 11, '갈마역': 12, '월평역': 13, '갑천역': 14,
                        '유성온천역': 15, '구암역': 16, '현충원역': 17, '월드컵경기장역': 18,
                        '노은역': 19, '지족역': 20, '반석역': 21
                    }
                    
                    station_results = {}
                    for station, idx in station_to_idx.items():
                        if station in major_stations:
                            station_results[station] = {
                                'pred': pred_raw[idx].item(),
                                'target': Y_raw[idx].item()
                            }
                    
                    sample_details.append({
                        'date': metadata['date'],
                        'hour': metadata['hour'],
                        'pred_range': [pred_raw.min().item(), pred_raw.max().item()],
                        'target_range': [Y_raw.min().item(), Y_raw.max().item()],
                        'avg_pred': pred_raw.mean().item(),
                        'avg_target': Y_raw.mean().item(),
                        'stations': station_results
                    })
        
        # 성능 지표 계산
        predictions_array = np.array(all_predictions)
        targets_array = np.array(all_targets)
        
        metrics = self.compute_metrics(predictions_array, targets_array)
        
        # 결과 출력
        print(f"\n📊 평가 결과 ({len(all_predictions)}개 예측):")
        print("="*50)
        print(f"MAE:  {metrics['MAE']:.2f}명")
        print(f"RMSE: {metrics['RMSE']:.2f}명")
        print(f"MAPE: {metrics['MAPE']:.1f}%")
        print(f"R²:   {metrics['R2']:.3f}")
        
        print(f"\n📈 예측 범위: [{predictions_array.min():.1f}, {predictions_array.max():.1f}]명")
        print(f"📈 실제 범위: [{targets_array.min():.1f}, {targets_array.max():.1f}]명")
        
        # 상세 샘플 정보
        print(f"\n🔍 샘플별 상세 정보 (처음 10개):")
        print("-"*100)
        for i, detail in enumerate(sample_details):
            print(f"📅 샘플 {i+1}: {detail['date']} {detail['hour']}시")
            print(f"   전체 평균 - 예측: {detail['avg_pred']:.1f}명, 실제: {detail['avg_target']:.1f}명")
            print(f"   🚇 주요 역별 예측:")
            
            for station, result in detail['stations'].items():
                pred = result['pred']
                target = result['target']
                error = abs(pred - target)
                error_pct = (error / target * 100) if target > 0 else 0
                print(f"      {station:10s}: 예측 {pred:6.1f}명 | 실제 {target:6.1f}명 | 오차 {error:6.1f}명 ({error_pct:5.1f}%)")
            print()
        
        return metrics

def main():
    evaluator = OccupancyEvaluator()
    metrics = evaluator.evaluate_model(num_samples=200)
    
    print("🎉 평가 완료!")

if __name__ == "__main__":
    main()
