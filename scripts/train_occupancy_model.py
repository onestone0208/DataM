#!/usr/bin/env python3
"""
혼잡도 전용 DCRNN 모델 훈련 스크립트

혼잡도(occupancy)만을 예측하는 단일 태스크 모델
- 입력: 과거 12시간의 혼잡도 + 통행량 데이터
- 출력: 1시간 후 혼잡도 예측
"""

import os
import sys
import argparse
import yaml
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN
from models.dataset import SubwayGraphDataset


class OccupancyDataset(SubwayGraphDataset):
    """혼잡도 전용 데이터셋 (Log1p + 정규화 적용, 승하차 별도 처리)"""
    
    def __init__(self, *args, normalization_stats=None, boarding_alighting_data=None, **kwargs):
        # 🔥 승하차 데이터 로드 및 사전 처리
        if boarding_alighting_data is None:
            import pandas as pd
            self.boarding_alighting_data = pd.read_csv('data/승하차.csv', encoding='cp949')
        else:
            self.boarding_alighting_data = boarding_alighting_data
        
        # 🔧 승하차 데이터 사전 처리 (성능 최적화)
        self._preprocess_boarding_alighting()
            
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
        
    def _preprocess_boarding_alighting(self):
        """승하차 데이터 사전 처리 (성능 최적화)"""
        print("🔧 승하차 데이터 사전 처리 중...")
        
        # 승차/하차 데이터를 딕셔너리로 변환
        self.boarding_dict = {}
        self.alighting_dict = {}
        
        # 시간 컬럼 리스트
        time_cols = [f"{h:02d}-{h+1:02d}시" if h < 23 else "23-00시" for h in range(24)]
        
        for _, row in self.boarding_alighting_data.iterrows():
            date = row['날짜']
            station = row['역명']
            category = row['구분']
            
            if category == '승차':
                if date not in self.boarding_dict:
                    self.boarding_dict[date] = {}
                if station not in self.boarding_dict[date]:
                    self.boarding_dict[date][station] = {}
                
                for i, col in enumerate(time_cols):
                    self.boarding_dict[date][station][i] = float(row[col])
                    
            elif category == '하차':
                if date not in self.alighting_dict:
                    self.alighting_dict[date] = {}
                if station not in self.alighting_dict[date]:
                    self.alighting_dict[date][station] = {}
                
                for i, col in enumerate(time_cols):
                    self.alighting_dict[date][station][i] = float(row[col])
        
        print(f"✅ 승하차 데이터 사전 처리 완료")
        
    def _get_boarding_alighting(self, date: str, hour: int, station: str) -> Tuple[float, float]:
        """특정 날짜/시간/역의 승차/하차 인원 추출 (최적화된 버전)"""
        try:
            # 🔧 사전 처리된 딕셔너리에서 빠른 조회
            boarding = self.boarding_dict.get(date, {}).get(station, {}).get(hour, 0.0)
            alighting = self.alighting_dict.get(date, {}).get(station, {}).get(hour, 0.0)
            
            return float(boarding), float(alighting)
            
        except Exception as e:
            # 데이터가 없으면 0 반환
            return 0.0, 0.0
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        
        # 🔥 혼잡도 + 실제 승하차 정보 사용
        X_full = sample['X']  # [T, N, F] - 모든 특성
        Y_occupancy = sample['Y'][:, 0:1]  # [N, 1] - 혼잡도만 예측
        
        # 🔥 혼잡도 추출
        X_occupancy = X_full[:, :, 0:1]  # [T, N, 1] - 혼잡도
        X_occupancy_transformed = self._transform_occupancy(X_occupancy)
        
        # 🔥 실제 승하차 데이터 추출
        sequence = self.sequences[idx]
        T, N = X_occupancy.shape[0], X_occupancy.shape[1]
        
        X_boarding = torch.zeros(T, N, 1)  # 승차 데이터
        X_alighting = torch.zeros(T, N, 1)  # 하차 데이터
        
        # 각 시점별로 승하차 데이터 수집
        for t in range(T):
            hour = sequence['start_hour'] + t
            date = sequence['date']
            
            for station_name, station_idx in self.station_to_idx.items():
                boarding, alighting = self._get_boarding_alighting(date, hour, station_name)
                X_boarding[t, station_idx, 0] = boarding
                X_alighting[t, station_idx, 0] = alighting
        
        # 🔥 승하차 정규화 (sqrt 정규화 - 0 근처 값이 많음)
        X_boarding_normalized = torch.sqrt(X_boarding) / 15.0  # 승차 정규화
        X_alighting_normalized = torch.sqrt(X_alighting) / 15.0  # 하차 정규화
        
        # 🔥 시간 특성을 시계열로 추가
        time_features_seq = sample['time_features']  # [T, time_feat_dim]
        T, time_feat_dim = time_features_seq.shape
        
        # 시간 특성을 모든 노드에 복제
        time_features_expanded = time_features_seq.unsqueeze(1).expand(-1, N, -1)  # [T, N, time_feat_dim]
        
        # 🔥 혼잡도 + 승차 + 하차 + 시간특성 결합
        X_combined = torch.cat([
            X_occupancy_transformed,  # [T, N, 1] - 혼잡도 (Log1p + 정규화)
            X_boarding_normalized,    # [T, N, 1] - 승차 (sqrt 정규화)
            X_alighting_normalized,   # [T, N, 1] - 하차 (sqrt 정규화)
            time_features_expanded    # [T, N, 7] - 시간 특성 (시계열)
        ], dim=-1)  # [T, N, 10]
        
        # 🔥 타겟은 혼잡도만 Log1p + 정규화
        Y_transformed = self._transform_occupancy(Y_occupancy)
        
        sample['X'] = X_combined  # [T, N, 10] - 혼잡도 + 승하차 + 시간특성
        sample['Y'] = Y_transformed
        
        # 🔥 원본 값도 저장 (loss 계산용)
        sample['Y_raw'] = Y_occupancy  # 원본 혼잡도
        
        return sample


class OccupancyTrainer:
    """혼잡도 전용 DCRNN 훈련 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._setup_device()
        self.logger = self._setup_logging()
        
        # 재현성 설정
        self._set_seed(config['experiment']['seed'])
        
        # 모델 및 데이터 초기화
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.train_loader = None
        self.val_loader = None
        self.adjacency_matrix = None
        
        # 학습 상태
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # 로깅
        self.writer = None
        if config['experiment']['logging']['tensorboard']:
            log_dir = Path(config['experiment']['logging']['log_dir']) / "occupancy_model"
            self.writer = SummaryWriter(log_dir)
    
    def _setup_device(self) -> torch.device:
        """디바이스 설정"""
        device_config = self.config['hardware']['device']
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device_config)
        
        print(f"Using device: {device}")
        return device
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        log_config = self.config['experiment']['logging']
        
        # 로그 디렉토리 생성
        log_dir = Path(log_config['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 로거 설정
        logger = logging.getLogger('DCRNN-Occupancy')
        logger.setLevel(getattr(logging, log_config['level']))
        
        # 파일 핸들러
        log_file = log_dir / "occupancy_model.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포매터
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _set_seed(self, seed: int):
        """재현성을 위한 시드 설정"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if self.config['experiment']['deterministic']:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def load_data(self):
        """데이터 로드"""
        self.logger.info("혼잡도 전용 데이터셋 로드 중...")
        
        # 원본 데이터에서 생성
        import pandas as pd
        
        congestion_data = pd.read_csv('data/혼잡도.csv')
        node_features = pd.read_csv('data/node_features.csv')
        date_features = pd.read_csv('data/date_features.csv')
        time_features = pd.read_csv('data/time_features.csv')
        adjacency_matrix = np.load('data/adjacency_matrix.npy')
        
        # 역 인덱스 매핑
        stations = sorted(congestion_data['station'].unique())
        station_to_idx = {station: idx for idx, station in enumerate(stations)}
        
        # 🔥 계절별 균등 분할 (train: 70%, val: 15%, test: 15%)
        dates = sorted(congestion_data['date'].unique())
        
        # 월별로 날짜 그룹화
        monthly_dates = {}
        for date in dates:
            month = int(date.split('-')[1])
            if month not in monthly_dates:
                monthly_dates[month] = []
            monthly_dates[month].append(date)
        
        # 각 월에서 비율대로 분할
        train_dates = []
        val_dates = []
        test_dates = []
        
        import random
        random.seed(42)  # 재현성을 위한 시드 고정
        
        for month, month_dates in monthly_dates.items():
            n_month = len(month_dates)
            train_end = int(n_month * 0.7)
            val_end = int(n_month * 0.85)
            
            # 각 월에서 랜덤하게 분할
            month_dates_shuffled = month_dates.copy()
            random.shuffle(month_dates_shuffled)
            
            train_dates.extend(month_dates_shuffled[:train_end])
            val_dates.extend(month_dates_shuffled[train_end:val_end])
            test_dates.extend(month_dates_shuffled[val_end:])
        
        self.logger.info(f"계절별 균등 분할 완료:")
        self.logger.info(f"  Train: {len(train_dates)}일")
        self.logger.info(f"  Val: {len(val_dates)}일") 
        self.logger.info(f"  Test: {len(test_dates)}일")
        
        # 🔥 분할 결과를 파일로 저장 (평가에서 사용)
        import pickle
        split_data = {
            'train_dates': train_dates,
            'val_dates': val_dates,
            'test_dates': test_dates,
            'split_method': 'seasonal_equal',
            'seed': 42
        }
        with open('data/seasonal_split.pkl', 'wb') as f:
            pickle.dump(split_data, f)
        self.logger.info(f"분할 정보 저장: data/seasonal_split.pkl")
        
        # 데이터셋 생성
        train_data = congestion_data[congestion_data['date'].isin(train_dates)]
        val_data = congestion_data[congestion_data['date'].isin(val_dates)]
        
        # 🔧 Train 데이터셋 먼저 생성 (정규화 통계 계산)
        train_dataset = OccupancyDataset(
            train_data, node_features, date_features, time_features,
            adjacency_matrix, station_to_idx, 
            sequence_length=12, prediction_length=1,
            target_columns=['occupancy']  # 혼잡도만
        )
        
        # 🔧 Train에서 계산한 정규화 통계를 Val에 공유
        normalization_stats = (train_dataset.log_mean, train_dataset.log_std)
        val_dataset = OccupancyDataset(
            val_data, node_features, date_features, time_features,
            adjacency_matrix, station_to_idx, 
            sequence_length=12, prediction_length=1,
            target_columns=['occupancy'],  # 혼잡도만
            normalization_stats=normalization_stats  # 🔧 Train 통계 공유
        )
        
        # 인접행렬 로드
        self.adjacency_matrix = torch.FloatTensor(adjacency_matrix).to(self.device)
        
        # 데이터 로더 생성
        data_config = self.config['data']
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory']
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory']
        )
        
        self.logger.info(f"Train batches: {len(self.train_loader)}")
        self.logger.info(f"Val batches: {len(self.val_loader)}")
    
    def build_model(self):
        """혼잡도 전용 모델 구축"""
        self.logger.info("혼잡도 전용 DCRNN 모델 구성 중...")
        
        model_config = self.config['model']
        
        # 🔧 시계열에 시간특성 포함 (10 + 16 + 21 = 47차원)
        self.model = DCRNN(
            input_size=47,  # 🔧 시계열(혼잡도+승하차+시간)(10) + 노드(16) + 날짜(21) = 47차원  
            hidden_size=model_config['hidden_size'],
            output_size=1,  # 혼잡도만 출력
            num_layers=model_config['num_layers'],
            diffusion_steps=model_config['diffusion_steps'],
            use_attention=model_config['use_attention'],
            dropout=model_config['dropout']
        ).to(self.device)
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def setup_training(self):
        """학습 설정"""
        # 🔥 가중치 Loss function (큰 값에 더 민감하게)
        self.loss_fn = self._weighted_mae_loss
        
        # 옵티마이저
        opt_config = self.config['optimizer']
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config['weight_decay'],
            betas=opt_config['betas'],
            eps=opt_config['eps']
        )
        
        # 스케줄러
        sched_config = self.config['scheduler']
        if sched_config['type'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config['T_max'],
                eta_min=sched_config['eta_min']
            )
        
        self.logger.info("혼잡도 모델 훈련 설정 완료")
        
    def _weighted_mae_loss(self, pred_norm, target_norm, raw_target, alpha=0.5, C=300.0):
        """
        큰 혼잡도 값에 더 가중치를 주는 MAE Loss
        
        Args:
            pred_norm: 정규화된 예측값 [B, N, 1]
            target_norm: 정규화된 타겟값 [B, N, 1]  
            raw_target: 원본 혼잡도 값 [B, N, 1] (명 단위)
            alpha: 가중치 강도 (0.5 = 50% 추가 가중치, 과도한 bias 억제)
            C: 가중치 스케일링 상수 (300명 기준, 더 완만한 가중치)
        """
        # 기본 MAE
        base_loss = torch.abs(pred_norm - target_norm)
        
        # 원본 값에 비례한 가중치 (큰 값일수록 가중치 증가)
        weights = 1.0 + alpha * (raw_target / C)
        
        # 가중치 적용
        weighted_loss = weights * base_loss
        
        return weighted_loss.mean()
    
    def get_teacher_forcing_ratio(self, epoch: int) -> float:
        """Teacher forcing 비율 계산"""
        tf_config = self.config['training']['teacher_forcing']
        
        if epoch >= tf_config['decay_epochs']:
            return tf_config['final_ratio']
        
        # Cosine decay
        progress = epoch / tf_config['decay_epochs']
        ratio = tf_config['final_ratio'] + (tf_config['initial_ratio'] - tf_config['final_ratio']) * \
                (1 + np.cos(np.pi * progress)) / 2
        
        return ratio
    
    def train_epoch(self) -> Dict[str, float]:
        """한 에포크 학습"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # 예측 범위 모니터링
        pred_min, pred_max = float('inf'), float('-inf')
        
        teacher_forcing_ratio = self.get_teacher_forcing_ratio(self.current_epoch)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 🔧 첫 번째 배치에서 디버깅 정보 출력
            if batch_idx == 0 and self.current_epoch == 0:
                self.logger.info(f"첫 배치 처리 시작 - X shape: {batch['X'].shape}, Y shape: {batch['Y'].shape}")
            
            # 데이터 준비
            X = batch['X'].to(self.device)  # [B, T, N, 1] - 혼잡도만
            Y = batch['Y'].to(self.device)  # [B, N, 1] - 혼잡도만
            Y_raw = batch['Y_raw'].to(self.device)  # 🔧 디바이스 mismatch 방지
            
            # 인접행렬 확장
            batch_size = X.size(0)
            adjacency = self.adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
            
            # 추가 특성들 추출 (시간 특성은 이미 X에 포함됨)
            node_features = batch['node_features'].to(self.device)  # [B, N, node_dim]
            date_features = batch['date_features'].to(self.device)  # [B, date_dim]
            # time_features는 이미 X에 포함되어 있음
            
            # Forward pass
            predictions, _ = self.model(
                X, adjacency, 
                target_length=1,
                teacher_forcing_ratio=teacher_forcing_ratio,
                targets=Y.unsqueeze(1),  # [B, 1, N, 1]
                node_features=node_features,
                date_features=date_features,
                time_features=None  # 🔥 시간 특성은 X에 포함되어 있음
            )
            
            predictions = predictions.squeeze(1)  # [B, N, 1]
            
            # 예측 범위 업데이트
            pred_min = min(pred_min, predictions.min().item())
            pred_max = max(pred_max, predictions.max().item())
            
            # 🔥 가중치 손실 계산 (정규화된 값 + 원본 값)
            loss = self.loss_fn(predictions, Y, Y_raw)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # 그래디언트 클리핑
            if self.config['training']['gradient_clipping']['enabled']:
                max_norm = self.config['training']['gradient_clipping']['max_norm']
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            
            self.optimizer.step()
            
            # 통계 업데이트
            total_loss += loss.item()
            
            # 로깅
            if batch_idx % 50 == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {loss.item():.4f}, "
                    f"TF Ratio: {teacher_forcing_ratio:.3f}"
                )
        
        return {
            'loss': total_loss / num_batches,
            'teacher_forcing_ratio': teacher_forcing_ratio,
            'pred_range': [pred_min, pred_max]
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """한 에포크 검증"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                X = batch['X'].to(self.device)
                Y = batch['Y'].to(self.device)
                Y_raw = batch['Y_raw'].to(self.device)  # 🔧 디바이스 mismatch 방지
                
                batch_size = X.size(0)
                adjacency = self.adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
                
                # 추가 특성들 추출 (시간 특성은 이미 X에 포함됨)
                node_features = batch['node_features'].to(self.device)  # [B, N, node_dim]
                date_features = batch['date_features'].to(self.device)  # [B, date_dim]
                # time_features는 이미 X에 포함되어 있음
                
                # Forward pass (no teacher forcing)
                predictions, _ = self.model(
                    X, adjacency, 
                    target_length=1,
                    node_features=node_features,
                    date_features=date_features,
                    time_features=None  # 🔥 시간 특성은 X에 포함되어 있음
                )
                predictions = predictions.squeeze(1)
                
                # 🔥 가중치 손실 계산 (정규화된 값 + 원본 값)
                loss = self.loss_fn(predictions, Y, Y_raw)
                total_loss += loss.item()
        
        return {'loss': total_loss / num_batches}
    
    def save_checkpoint(self, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint_dir = Path("checkpoints/occupancy_model")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # 일반 체크포인트
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 최고 성능 모델
        if is_best:
            best_path = checkpoint_dir / "best_occupancy_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best occupancy model saved: {best_path}")
    
    def train(self, start_epoch: int = 0):
        """전체 학습 루프"""
        self.logger.info("혼잡도 모델 훈련 시작...")
        
        training_config = self.config['training']
        early_stopping_config = training_config['early_stopping']
        
        self.logger.info(f"총 {training_config['epochs']} 에포크 훈련 예정")
        self.logger.info(f"시작 에포크: {start_epoch}")
        self.logger.info(f"디바이스: {self.device}")
        
        for epoch in range(start_epoch, training_config['epochs']):
            self.current_epoch = epoch
            start_time = time.time()
            
            # 학습
            train_metrics = self.train_epoch()
            
            # 검증
            val_metrics = self.validate_epoch()
            
            # 스케줄러 업데이트
            if self.scheduler:
                self.scheduler.step()
            
            # 로깅
            epoch_time = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"LR: {current_lr:.6f}, "
                f"Time: {epoch_time:.1f}s"
            )
            
            # 예측 범위 로깅
            pred_range = train_metrics['pred_range']
            
            # 🔥 정규화된 예측 범위를 원본 스케일로 역변환
            try:
                pred_min_tensor = torch.tensor([pred_range[0]])
                pred_max_tensor = torch.tensor([pred_range[1]])
                
                # DataLoader에서 dataset 객체 가져오기
                dataset = self.train_loader.dataset
                pred_min_raw = dataset.inverse_transform(pred_min_tensor).item()
                pred_max_raw = dataset.inverse_transform(pred_max_tensor).item()
            except Exception as e:
                # 역변환 실패 시 정규화된 값만 표시
                self.logger.warning(f"역변환 실패: {e}")
                pred_min_raw = pred_range[0]
                pred_max_raw = pred_range[1]
            
            self.logger.info(
                f"혼잡도 예측 범위 (정규화): [{pred_range[0]:.2f}, {pred_range[1]:.2f}]"
            )
            self.logger.info(
                f"혼잡도 예측 범위 (실제): [{pred_min_raw:.1f}, {pred_max_raw:.1f}]명"
            )
            
            # TensorBoard 로깅
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                self.writer.add_scalar('Teacher_Forcing_Ratio', train_metrics['teacher_forcing_ratio'], epoch)
            
            # 체크포인트 저장
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # 정기 저장
            if epoch % training_config['checkpoint']['save_every'] == 0:
                self.save_checkpoint(is_best)
            
            # 최고 성능 모델 저장
            if is_best and training_config['checkpoint']['save_best']:
                self.save_checkpoint(is_best=True)
            
            # Early stopping
            if early_stopping_config['enabled']:
                if self.early_stopping_counter >= early_stopping_config['patience']:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        self.logger.info("혼잡도 모델 훈련 완료!")
        
        # 최종 체크포인트 저장
        self.save_checkpoint()
        
        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Occupancy DCRNN model')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 트레이너 생성
    trainer = OccupancyTrainer(config)
    
    # 데이터 로드
    trainer.load_data()
    
    # 모델 구축
    trainer.build_model()
    
    # 학습 설정
    trainer.setup_training()
    
    # 🔧 체크포인트 복원
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        trainer.current_epoch = checkpoint['epoch'] + 1
        start_epoch = checkpoint['epoch'] + 1
        print(f"🔄 Resumed from epoch {checkpoint['epoch']}, starting at epoch {start_epoch}")
    
    # 학습 시작
    trainer.train(start_epoch)


if __name__ == '__main__':
    main()
