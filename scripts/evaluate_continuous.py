#!/usr/bin/env python3
"""
🔧 연속된 시간 데이터로 정확한 혼잡도 모델 평가

특정 시점의 이전 12시간 연속 데이터를 사용하여 다음 1시간 혼잡도 예측
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN
from scripts.train_occupancy_model import OccupancyDataset


class ContinuousEvaluator:
    """연속된 시간 데이터로 정확한 평가"""
    
    def __init__(self):
        self.device = torch.device('cpu')
        
    def load_model(self):
        """최신 혼잡도 모델 로드"""
        import glob
        
        checkpoint_files = glob.glob('checkpoints/occupancy_model/checkpoint_epoch_*.pth')
        if not checkpoint_files:
            raise FileNotFoundError("체크포인트 파일을 찾을 수 없습니다")
        
        # 가장 최신 체크포인트 선택
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f"📂 최신 체크포인트 사용: {latest_checkpoint}")
        
        # 체크포인트 로드
        checkpoint = torch.load(latest_checkpoint, map_location=self.device)
        print(f"📋 모델 정보: Epoch {checkpoint['epoch']}")
        
        # 모델 생성 (혼잡도 전용)
        model = DCRNN(
            input_size=45,  # 혼잡도(1) + 추가특성(44) = 45차원
            hidden_size=64,
            output_size=1,  # 혼잡도만 출력
            num_layers=2,
            diffusion_steps=3,
            use_attention=True,
            dropout=0.1
        ).to(self.device)
        
        # 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def extract_continuous_sequence(self, congestion_data, target_date, target_hour, 
                                   node_features, date_features, time_features, station_to_idx):
        """특정 시점의 연속된 12시간 데이터 추출"""
        
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
            time_slot = f"{seq_hour:02d}-{(seq_hour+1):02d}시"
            hour_data = congestion_data[
                (congestion_data['date'] == seq_date) & 
                (congestion_data['time_slot'] == time_slot)
            ]
            if len(hour_data) == 0:
                print(f"⚠️ 데이터 없음: {seq_date} {time_slot}")
                return None
            sequence_data.append(hour_data)
        
        # 타겟 시점 데이터
        target_time_slot = f"{target_hour:02d}-{(target_hour+1):02d}시"
        target_data = congestion_data[
            (congestion_data['date'] == target_date) & 
            (congestion_data['time_slot'] == target_time_slot)
        ]
        if len(target_data) == 0:
            print(f"⚠️ 타겟 데이터 없음: {target_date} {target_hour}시")
            return None
        
        # 데이터 구성
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
        
        # 추가 특성들 추출
        # Node features (모든 역에 동일한 특성 적용)
        node_feat = node_features.iloc[0, 1:].values.astype(float)  # 첫 번째 컬럼 제외
        node_features_tensor = torch.FloatTensor(node_feat).unsqueeze(0).repeat(22, 1)  # [22, node_dim]
        
        # 🔧 Date features (SubwayGraphDataset와 동일한 방식)
        date_row = date_features[date_features['date'] == sequence_hours[0][0]].iloc[0]
        exclude_date_cols = ['weekday_name', 'holiday_name']  # 부모 클래스와 동일
        date_numeric_cols = []
        for col in date_features.columns[1:]:  # 'date' 컬럼 제외
            if col not in exclude_date_cols:
                try:
                    pd.to_numeric([date_row[col]], errors='raise')  # 부모 클래스와 동일한 검증
                    val = float(date_row[col]) if not pd.isna(date_row[col]) else 0.0
                    date_numeric_cols.append(val)
                except (ValueError, TypeError):
                    continue
        date_features_tensor = torch.FloatTensor(date_numeric_cols)  # [date_dim]
        
        # 🔧 Time features (SubwayGraphDataset와 동일한 방식)
        time_row = time_features[time_features['hour'] == sequence_hours[0][1]].iloc[0]
        exclude_time_cols = ['time_slot']  # 부모 클래스와 동일
        time_numeric_cols = []
        for col in time_features.columns:
            if col not in exclude_time_cols and col != 'hour':  # 부모 클래스와 동일
                val = float(time_row[col])
                time_numeric_cols.append(val)
        time_features_tensor = torch.FloatTensor(time_numeric_cols)  # [time_dim]
        
        # 🔧 훈련과 동일: 혼잡도만 추출 [T, N, 1]
        X_occupancy_only = X[:, :, 0:1]  # [12, 22, 1] - 혼잡도만
        
        return {
            'X': torch.FloatTensor(X_occupancy_only),  # 🔧 [12, 22, 1] - 혼잡도만 (훈련과 동일)
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
    
    def compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """성능 지표 계산"""
        # MAE
        mae = np.mean(np.abs(predictions - targets))
        
        # RMSE
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        # MAPE
        mask = targets > 0  # 0이 아닌 값만
        if np.any(mask):
            mape = np.mean(np.abs((predictions[mask] - targets[mask]) / targets[mask])) * 100
        else:
            mape = float('inf')
        
        # R²
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
    
    def evaluate(self):
        """연속된 시간 데이터로 정확한 평가"""
        print("🔍 연속된 시간 데이터로 혼잡도 모델 평가 시작")
        print("="*60)
        
        model = self.load_model()
        
        # 원본 데이터 로드
        congestion_data = pd.read_csv('data/혼잡도.csv')
        node_features = pd.read_csv('data/node_features.csv')
        date_features = pd.read_csv('data/date_features.csv')
        time_features = pd.read_csv('data/time_features.csv')
        adjacency_matrix = np.load('data/adjacency_matrix.npy')
        adjacency_tensor = torch.FloatTensor(adjacency_matrix).to(self.device)
        
        # 역 인덱스 매핑
        stations = sorted(congestion_data['station'].unique())
        station_to_idx = {station: idx for idx, station in enumerate(stations)}
        
        # 테스트 날짜 범위
        dates = sorted(congestion_data['date'].unique())
        n_dates = len(dates)
        val_end = int(n_dates * 0.85)
        test_dates = dates[val_end:]
        
        print(f"📅 테스트 기간: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}일)")
        
        # 테스트 케이스 선택
        test_cases = []
        for date in test_dates[:5]:  # 처음 5일
            for hour in [9, 12, 15, 18]:  # 주요 시간대
                test_cases.append((date, hour))
        
        print(f"🎯 {len(test_cases)}개 테스트 케이스로 평가...")
        
        all_predictions = []
        all_targets = []
        case_details = []
        
        # 정규화 통계 로드 (훈련과 동일)
        train_dates = dates[:int(n_dates * 0.7)]
        train_data = congestion_data[congestion_data['date'].isin(train_dates)]
        train_dataset = OccupancyDataset(
            train_data, node_features, date_features, time_features,
            adjacency_matrix, station_to_idx, 
            sequence_length=12, prediction_length=1,
            target_columns=['occupancy']
        )
        
        with torch.no_grad():
            for case_idx, (target_date, target_hour) in enumerate(test_cases):
                print(f"\n📍 케이스 {case_idx+1}: {target_date} {target_hour}시 예측")
                
                # 연속된 12시간 데이터 추출
                sequence_data = self.extract_continuous_sequence(
                    congestion_data, target_date, target_hour,
                    node_features, date_features, time_features, station_to_idx
                )
                
                if sequence_data is None:
                    continue
                
                # 🔧 훈련과 동일한 전처리 적용
                X_occupancy = sequence_data['X']  # [12, 22, 1] - 이미 혼잡도만 추출됨
                Y_raw = sequence_data['Y']  # [22, 1]
                
                # 🔧 Log1p + 정규화 적용 (훈련과 동일)
                X_occupancy_transformed = train_dataset._transform_occupancy(X_occupancy)
                
                # 모델 입력 준비
                X = X_occupancy_transformed.unsqueeze(0).to(self.device)  # [1, 12, 22, 1]
                node_features_batch = sequence_data['node_features'].unsqueeze(0).to(self.device)
                date_features_batch = sequence_data['date_features'].unsqueeze(0).to(self.device)
                time_features_batch = sequence_data['time_features'].unsqueeze(0).to(self.device)
                
                # 인접행렬
                batch_size = X.size(0)
                adjacency = adjacency_tensor.unsqueeze(0).expand(batch_size, -1, -1)
                
                # 예측
                predictions, _ = model(
                    X, adjacency,
                    target_length=1,
                    node_features=node_features_batch,
                    date_features=date_features_batch,
                    time_features=time_features_batch
                )
                predictions = predictions.squeeze()  # [22, 1] or [22]
                
                # 정규화 해제
                if predictions.dim() == 1:
                    predictions = predictions.unsqueeze(-1)
                pred_raw = train_dataset.inverse_transform(predictions).squeeze(-1)
                
                # 결과 저장
                all_predictions.extend(pred_raw.cpu().numpy())
                all_targets.extend(Y_raw.squeeze().cpu().numpy())
                
                # 상세 정보
                major_stations = ['대전역', '중앙로역', '서대전네거리역', '정부청사역', '갈마역']
                station_results = {}
                for station in major_stations:
                    if station in station_to_idx:
                        idx = station_to_idx[station]
                        station_results[station] = {
                            'pred': pred_raw[idx].item(),
                            'target': Y_raw[idx].item(),
                            'error': abs(pred_raw[idx].item() - Y_raw[idx].item())
                        }
                
                case_details.append({
                    'date': target_date,
                    'hour': target_hour,
                    'sequence_hours': sequence_data['metadata']['sequence_hours'],
                    'avg_pred': pred_raw.mean().item(),
                    'avg_target': Y_raw.mean().item(),
                    'stations': station_results
                })
                
                print(f"   평균 예측: {pred_raw.mean().item():.1f}명, 실제: {Y_raw.mean().item():.1f}명")
        
        # 전체 성능 지표 계산
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        metrics = self.compute_metrics(all_predictions, all_targets)
        
        print(f"\n📊 연속 시간 평가 결과 ({len(all_predictions)}개 예측):")
        print("="*50)
        print(f"MAE:  {metrics['mae']:.2f}명")
        print(f"RMSE: {metrics['rmse']:.2f}명")
        print(f"MAPE: {metrics['mape']:.1f}%")
        print(f"R²:   {metrics['r2']:.3f}")
        
        print(f"\n📈 예측 범위: [{all_predictions.min():.1f}, {all_predictions.max():.1f}]명")
        print(f"📈 실제 범위: [{all_targets.min():.1f}, {all_targets.max():.1f}]명")
        
        # 케이스별 상세 정보
        print(f"\n🔍 케이스별 상세 정보:")
        print("-"*80)
        for detail in case_details:
            print(f"📅 {detail['date']} {detail['hour']}시")
            print(f"   🕐 입력 시간: {detail['sequence_hours'][0][0]} {detail['sequence_hours'][0][1]}시 ~ {detail['sequence_hours'][-1][0]} {detail['sequence_hours'][-1][1]}시")
            print(f"   전체 평균 - 예측: {detail['avg_pred']:.1f}명, 실제: {detail['avg_target']:.1f}명")
            print(f"   🚇 주요 역별:")
            for station, result in detail['stations'].items():
                error_pct = (result['error'] / max(result['target'], 1)) * 100
                print(f"      {station:10s}: 예측 {result['pred']:5.1f}명 | 실제 {result['target']:5.1f}명 | 오차 {result['error']:5.1f}명 ({error_pct:4.1f}%)")
            print()
        
        return metrics


def main():
    evaluator = ContinuousEvaluator()
    
    print("🚀 연속된 시간 데이터로 정확한 평가 시작!")
    metrics = evaluator.evaluate()
    
    print(f"\n🎉 최종 성능:")
    print(f"MAE: {metrics['mae']:.2f}명")
    print(f"RMSE: {metrics['rmse']:.2f}명") 
    print(f"R²: {metrics['r2']:.3f}")


if __name__ == "__main__":
    main()
