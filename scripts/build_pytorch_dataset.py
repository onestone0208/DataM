#!/usr/bin/env python3
"""
PyTorch Dataset 구축 스크립트

시공간 그래프 신경망을 위한 데이터 로더를 구축합니다:
- 슬라이딩 윈도우 기반 시계열 데이터
- 그래프 인접행렬
- 노드/시간 특징
- 배치 텐서 구조
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import json
import pickle
from datetime import datetime, timedelta


class SubwayGraphDataset(Dataset):
    """지하철 그래프 데이터셋"""
    
    def __init__(self, 
                 congestion_data: pd.DataFrame,
                 node_features: pd.DataFrame,
                 date_features: pd.DataFrame,
                 time_features: pd.DataFrame,
                 adjacency_matrix: np.ndarray,
                 station_to_idx: Dict[str, int],
                 sequence_length: int = 12,
                 prediction_length: int = 1,
                 target_columns: List[str] = None):
        
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.station_to_idx = station_to_idx
        self.n_stations = len(station_to_idx)
        
        # 타깃 컬럼 설정
        if target_columns is None:
            self.target_columns = ['occupancy', 'total_flow']
        else:
            self.target_columns = target_columns
        
        # 데이터 전처리
        self.adjacency_matrix = torch.FloatTensor(adjacency_matrix)
        self.node_features = self._process_node_features(node_features)
        self.time_features = self._process_time_features(time_features)
        self.date_features = self._process_date_features(date_features)
        
        # 시계열 데이터 구성
        self.sequences = self._build_sequences(congestion_data)
        
        print(f"Dataset 생성 완료:")
        print(f"  - 시퀀스 수: {len(self.sequences)}")
        print(f"  - 입력 길이: {sequence_length}시간")
        print(f"  - 예측 길이: {prediction_length}시간")
        print(f"  - 역 수: {self.n_stations}개")
        print(f"  - 타깃: {self.target_columns}")
    
    def _process_node_features(self, node_features: pd.DataFrame) -> torch.Tensor:
        """노드 특징 처리"""
        # 역명을 인덱스로 정렬
        node_features = node_features.set_index('역명')
        station_names = [name for name, _ in sorted(self.station_to_idx.items(), key=lambda x: x[1])]
        node_features = node_features.loc[station_names]
        
        # 수치형 컬럼만 선택
        numeric_cols = node_features.select_dtypes(include=[np.number]).columns
        features = node_features[numeric_cols].values
        
        return torch.FloatTensor(features)
    
    def _process_time_features(self, time_features: pd.DataFrame) -> torch.Tensor:
        """시간 특징 처리"""
        # hour 순서로 정렬
        time_features = time_features.sort_values('hour')
        
        # 수치형 컬럼만 선택 (hour, time_slot 제외)
        numeric_cols = [col for col in time_features.columns 
                       if col not in ['hour', 'time_slot'] and 
                       time_features[col].dtype in [np.int64, np.float64]]
        
        features = time_features[numeric_cols].values
        return torch.FloatTensor(features)
    
    def _process_date_features(self, date_features: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """날짜 특징 처리"""
        date_features = date_features.set_index('date')
        
        # 수치형 컬럼만 선택 (문자열 컬럼 제외)
        exclude_cols = ['weekday_name', 'holiday_name']
        numeric_cols = []
        
        for col in date_features.columns:
            if col not in exclude_cols:
                try:
                    # 수치형으로 변환 가능한지 확인
                    pd.to_numeric(date_features[col], errors='raise')
                    numeric_cols.append(col)
                except (ValueError, TypeError):
                    # 변환 불가능하면 제외
                    continue
        
        features_dict = {}
        for date, row in date_features.iterrows():
            # 수치형 데이터만 추출하고 float으로 변환
            numeric_values = []
            for col in numeric_cols:
                val = row[col]
                if pd.isna(val):
                    numeric_values.append(0.0)
                else:
                    numeric_values.append(float(val))
            
            features_dict[date] = torch.FloatTensor(numeric_values)
        
        return features_dict
    
    def _build_sequences(self, congestion_data: pd.DataFrame) -> List[Dict]:
        """시계열 시퀀스 구성"""
        sequences = []
        
        # 날짜별로 그룹화
        congestion_data['datetime'] = pd.to_datetime(
            congestion_data['date'] + ' ' + 
            congestion_data['time_slot'].str.replace('시', ':00').str.replace('-', ':00-')
        )
        
        # 시간 슬롯을 시간으로 변환
        def parse_time_slot(time_slot):
            start_hour = int(time_slot.split('-')[0])
            return start_hour
        
        congestion_data['hour'] = congestion_data['time_slot'].apply(parse_time_slot)
        
        # 날짜별 처리
        dates = sorted(congestion_data['date'].unique())
        
        for i in range(len(dates) - 1):  # 마지막 날은 예측 타깃이 없으므로 제외
            current_date = dates[i]
            next_date = dates[i + 1]
            
            # 현재 날짜 데이터
            current_data = congestion_data[congestion_data['date'] == current_date]
            next_data = congestion_data[congestion_data['date'] == next_date]
            
            # 시간대별로 슬라이딩 윈도우
            for start_hour in range(24 - self.sequence_length - self.prediction_length + 1):
                end_hour = start_hour + self.sequence_length
                pred_hour = end_hour + self.prediction_length - 1
                
                # 입력 시퀀스 (현재 날짜)
                input_hours = list(range(start_hour, end_hour))
                input_data = current_data[current_data['hour'].isin(input_hours)]
                
                # 예측 타깃 (다음 시간 또는 다음 날)
                if pred_hour < 24:
                    target_data = current_data[current_data['hour'] == pred_hour]
                else:
                    # 다음 날 첫 시간
                    target_hour = pred_hour - 24
                    target_data = next_data[next_data['hour'] == target_hour]
                
                # 데이터 유효성 검사
                if len(input_data) == self.sequence_length * self.n_stations and \
                   len(target_data) == self.n_stations:
                    
                    sequence = {
                        'date': current_date,
                        'start_hour': start_hour,
                        'input_data': input_data,
                        'target_data': target_data
                    }
                    sequences.append(sequence)
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # 입력 시계열 구성 [T, N, F]
        input_data = sequence['input_data']
        target_data = sequence['target_data']
        
        # 시간별로 정렬
        input_data = input_data.sort_values(['hour', 'station'])
        target_data = target_data.sort_values('station')
        
        # 입력 텐서 구성
        X = []  # [T, N, F] - 시간, 노드, 특징
        for t in range(self.sequence_length):
            hour_data = input_data[input_data['hour'] == sequence['start_hour'] + t]
            
            # 역별로 정렬
            station_features = []
            for station_name in sorted(self.station_to_idx.keys(), key=lambda x: self.station_to_idx[x]):
                station_row = hour_data[hour_data['station'] == station_name]
                if len(station_row) > 0:
                    # 타깃 컬럼 값 추출
                    features = [station_row[col].iloc[0] for col in self.target_columns]
                else:
                    # 결측값은 0으로 처리
                    features = [0.0] * len(self.target_columns)
                station_features.append(features)
            
            X.append(station_features)
        
        X = torch.FloatTensor(X)  # [T, N, F]
        
        # 타깃 텐서 구성 [N, F]
        Y = []
        for station_name in sorted(self.station_to_idx.keys(), key=lambda x: self.station_to_idx[x]):
            station_row = target_data[target_data['station'] == station_name]
            if len(station_row) > 0:
                features = [station_row[col].iloc[0] for col in self.target_columns]
            else:
                features = [0.0] * len(self.target_columns)
            Y.append(features)
        
        Y = torch.FloatTensor(Y)  # [N, F]
        
        # 날짜/시간 특징
        date = sequence['date']
        hour = sequence['start_hour']
        
        date_feat = self.date_features.get(date, torch.zeros(self.date_features[list(self.date_features.keys())[0]].shape))
        time_feat = self.time_features[hour]
        
        return {
            'X': X,                              # [T, N, F] 입력 시계열
            'Y': Y,                              # [N, F] 타깃
            'adjacency': self.adjacency_matrix,  # [N, N] 인접행렬
            'node_features': self.node_features, # [N, node_dim] 노드 특징
            'date_features': date_feat,          # [date_dim] 날짜 특징
            'time_features': time_feat,          # [time_dim] 시간 특징
            'metadata': {
                'date': date,
                'hour': hour,
                'idx': idx
            }
        }


def load_all_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, Dict]:
    """모든 데이터 로드"""
    
    # 혼잡도 데이터
    congestion_df = pd.read_csv(data_dir / "혼잡도.csv")
    
    # 특징 데이터
    node_features_df = pd.read_csv(data_dir / "node_features.csv")
    date_features_df = pd.read_csv(data_dir / "date_features.csv")
    time_features_df = pd.read_csv(data_dir / "time_features.csv")
    
    # 그래프 데이터
    adjacency_matrix = np.load(data_dir / "adjacency_matrix.npy")
    with open(data_dir / "graph_info.json", 'r', encoding='utf-8') as f:
        graph_info = json.load(f)
    
    return (congestion_df, node_features_df, date_features_df, time_features_df, 
            adjacency_matrix, graph_info["station_to_idx"])


def create_datasets(data_dir: Path, 
                   sequence_length: int = 12,
                   prediction_length: int = 1,
                   target_columns: List[str] = None) -> Tuple[SubwayGraphDataset, SubwayGraphDataset, SubwayGraphDataset]:
    """train/val/test 데이터셋 생성"""
    
    # 데이터 로드
    congestion_df, node_features_df, date_features_df, time_features_df, adjacency_matrix, station_to_idx = load_all_data(data_dir)
    
    # 날짜별 분할 정보 로드
    split_df = pd.read_csv(data_dir / "날짜정보_split.csv")
    
    # 분할별 데이터 필터링
    train_dates = split_df[split_df['split'] == 'train']['date'].tolist()
    val_dates = split_df[split_df['split'] == 'val']['date'].tolist()
    test_dates = split_df[split_df['split'] == 'test']['date'].tolist()
    
    train_congestion = congestion_df[congestion_df['date'].isin(train_dates)]
    val_congestion = congestion_df[congestion_df['date'].isin(val_dates)]
    test_congestion = congestion_df[congestion_df['date'].isin(test_dates)]
    
    # 데이터셋 생성
    train_dataset = SubwayGraphDataset(
        train_congestion, node_features_df, date_features_df, time_features_df,
        adjacency_matrix, station_to_idx, sequence_length, prediction_length, target_columns
    )
    
    val_dataset = SubwayGraphDataset(
        val_congestion, node_features_df, date_features_df, time_features_df,
        adjacency_matrix, station_to_idx, sequence_length, prediction_length, target_columns
    )
    
    test_dataset = SubwayGraphDataset(
        test_congestion, node_features_df, date_features_df, time_features_df,
        adjacency_matrix, station_to_idx, sequence_length, prediction_length, target_columns
    )
    
    return train_dataset, val_dataset, test_dataset


def save_datasets(datasets: Tuple[SubwayGraphDataset, SubwayGraphDataset, SubwayGraphDataset],
                 output_dir: Path) -> None:
    """데이터셋 저장"""
    train_dataset, val_dataset, test_dataset = datasets
    
    # 피클로 저장
    with open(output_dir / "train_dataset.pkl", 'wb') as f:
        pickle.dump(train_dataset, f)
    
    with open(output_dir / "val_dataset.pkl", 'wb') as f:
        pickle.dump(val_dataset, f)
    
    with open(output_dir / "test_dataset.pkl", 'wb') as f:
        pickle.dump(test_dataset, f)
    
    print(f"데이터셋 저장 완료: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch Dataset 구축")
    parser.add_argument(
        "--data-dir", 
        type=Path, 
        default=Path("data"),
        help="데이터 디렉토리"
    )
    parser.add_argument(
        "--sequence-length", 
        type=int, 
        default=12,
        help="입력 시퀀스 길이 (시간)"
    )
    parser.add_argument(
        "--prediction-length", 
        type=int, 
        default=1,
        help="예측 길이 (시간)"
    )
    parser.add_argument(
        "--target-columns", 
        nargs='+',
        default=['occupancy', 'total_flow'],
        help="예측 타깃 컬럼"
    )
    
    args = parser.parse_args()
    
    print("PyTorch Dataset 구축 시작...")
    
    # 데이터셋 생성
    datasets = create_datasets(
        args.data_dir, 
        args.sequence_length, 
        args.prediction_length,
        args.target_columns
    )
    
    # 저장
    save_datasets(datasets, args.data_dir)
    
    # 샘플 데이터 확인
    train_dataset = datasets[0]
    sample = train_dataset[0]
    
    print(f"\n=== 샘플 데이터 확인 ===")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key:15s}: {value.shape}")
        else:
            print(f"{key:15s}: {value}")
    
    print(f"\nPyTorch Dataset 구축 완료!")


if __name__ == "__main__":
    main()
