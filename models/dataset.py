#!/usr/bin/env python3
"""
지하철 그래프 데이터셋 클래스

PyTorch Dataset for subway congestion prediction
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
from datetime import datetime


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
        self.date_features = self._process_date_features(date_features)
        self.time_features = self._process_time_features(time_features)
        
        # 시퀀스 구성
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
        
        # 역 순서대로 정렬
        station_names = [name for name, _ in sorted(self.station_to_idx.items(), key=lambda x: x[1])]
        node_features = node_features.reindex(station_names)
        
        # 수치형 컬럼만 선택
        numeric_cols = node_features.select_dtypes(include=[np.number]).columns
        features = node_features[numeric_cols].values
        
        return torch.FloatTensor(features)
    
    def _process_time_features(self, time_features: pd.DataFrame) -> Dict[int, torch.Tensor]:
        """시간 특징 처리"""
        # 수치형 컬럼만 선택
        exclude_cols = ['time_slot']
        numeric_cols = [col for col in time_features.columns 
                       if col not in exclude_cols and col != 'hour']
        
        features_dict = {}
        for _, row in time_features.iterrows():
            hour = int(row['hour'])
            numeric_values = [float(row[col]) for col in numeric_cols]
            features_dict[hour] = torch.FloatTensor(numeric_values)
        
        return features_dict
    
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
        congestion_data = congestion_data.copy()
        
        def parse_time_slot(time_slot_str):
            """시간대 문자열을 시간으로 변환"""
            try:
                # "00-01시" -> 0, "01-02시" -> 1 형태로 변환
                if isinstance(time_slot_str, str) and '시' in time_slot_str:
                    hour_part = time_slot_str.split('-')[0]
                    return int(hour_part)
                else:
                    return int(str(time_slot_str).split('-')[0])
            except:
                return 0
        
        # 시간 컬럼 먼저 추가
        congestion_data['hour'] = congestion_data['time_slot'].apply(parse_time_slot)
        
        # datetime 컬럼 생성 (시간 기반)
        congestion_data['datetime'] = pd.to_datetime(
            congestion_data['date'] + ' ' + congestion_data['hour'].astype(str) + ':00:00',
            format='%Y-%m-%d %H:00:00'
        )
        
        # 날짜별로 처리
        for date in sorted(congestion_data['date'].unique()):
            date_data = congestion_data[congestion_data['date'] == date].copy()
            
            # 시간순 정렬
            date_data = date_data.sort_values('hour')
            
            # 슬라이딩 윈도우
            max_hour = date_data['hour'].max()
            for start_hour in range(max_hour - self.sequence_length - self.prediction_length + 2):
                end_hour = start_hour + self.sequence_length
                pred_hour = end_hour + self.prediction_length - 1
                
                # 입력 시퀀스 데이터
                input_data = date_data[
                    (date_data['hour'] >= start_hour) & 
                    (date_data['hour'] < end_hour)
                ]
                
                # 예측 타깃 데이터
                target_data = date_data[date_data['hour'] == pred_hour]
                
                # 데이터 완전성 확인
                if len(input_data) == self.sequence_length * self.n_stations and \
                   len(target_data) == self.n_stations:
                    
                    sequences.append({
                        'date': date,
                        'start_hour': start_hour,
                        'end_hour': end_hour,
                        'pred_hour': pred_hour,
                        'input_data': input_data,
                        'target_data': target_data
                    })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # 입력 시계열 구성
        X = torch.zeros(self.sequence_length, self.n_stations, len(self.target_columns))
        
        input_data = sequence['input_data']
        for _, row in input_data.iterrows():
            station_idx = self.station_to_idx[row['station']]
            time_idx = int(row['hour'] - sequence['start_hour'])
            
            for feat_idx, col in enumerate(self.target_columns):
                X[time_idx, station_idx, feat_idx] = row[col]
        
        # 예측 타깃 구성
        Y = torch.zeros(self.n_stations, len(self.target_columns))
        
        target_data = sequence['target_data']
        for _, row in target_data.iterrows():
            station_idx = self.station_to_idx[row['station']]
            
            for feat_idx, col in enumerate(self.target_columns):
                Y[station_idx, feat_idx] = row[col]
        
        # 날짜 특징
        date_features = self.date_features[sequence['date']]
        
        # 🔥 시간 특징을 시계열로 구성 (각 시점별로 다른 시간 특성)
        time_features_sequence = []
        
        for t in range(self.sequence_length):
            hour = sequence['start_hour'] + t
            # 24시간 순환 처리
            hour = hour % 24
            time_features_sequence.append(self.time_features[hour])
        
        time_features = torch.stack(time_features_sequence, dim=0)  # [seq_len, time_feat_dim]
        
        return {
            'X': X,  # [seq_len, n_stations, n_features]
            'Y': Y,  # [n_stations, n_features]
            'adjacency': self.adjacency_matrix,  # [n_stations, n_stations]
            'node_features': self.node_features,  # [n_stations, node_feat_dim]
            'date_features': date_features,  # [date_feat_dim]
            'time_features': time_features,  # [seq_len, time_feat_dim]
            'metadata': {
                'date': sequence['date'],
                'hour': sequence['pred_hour'],
                'idx': idx
            }
        }
