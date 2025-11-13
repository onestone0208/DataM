#!/usr/bin/env python3
"""
통행량 전용 DCRNN 모델 훈련 스크립트

통행량(total_flow)만을 예측하는 단일 태스크 모델
- 입력: 과거 12시간의 혼잡도 + 통행량 데이터
- 출력: 1시간 후 통행량 예측
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


class FlowDataset(SubwayGraphDataset):
    """통행량 전용 데이터셋"""
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        
        # 통행량만 추출 (인덱스 1)
        sample['Y'] = sample['Y'][:, 1:2]  # [N, 1] - 통행량만
        
        return sample


class FlowTrainer:
    """통행량 전용 DCRNN 훈련 클래스"""
    
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
            log_dir = Path(config['experiment']['logging']['log_dir']) / "flow_model"
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
        logger = logging.getLogger('DCRNN-Flow')
        logger.setLevel(getattr(logging, log_config['level']))
        
        # 파일 핸들러
        log_file = log_dir / "flow_model.log"
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
        self.logger.info("통행량 전용 데이터셋 로드 중...")
        
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
        
        # 데이터 분할 (train: 70%, val: 15%, test: 15%)
        dates = sorted(congestion_data['date'].unique())
        n_dates = len(dates)
        
        train_end = int(n_dates * 0.7)
        val_end = int(n_dates * 0.85)
        
        train_dates = dates[:train_end]
        val_dates = dates[train_end:val_end]
        
        # 데이터셋 생성
        train_data = congestion_data[congestion_data['date'].isin(train_dates)]
        val_data = congestion_data[congestion_data['date'].isin(val_dates)]
        
        train_dataset = FlowDataset(
            train_data, node_features, date_features, time_features,
            adjacency_matrix, station_to_idx, 
            sequence_length=12, prediction_length=1,
            target_columns=['total_flow']  # 통행량만
        )
        
        val_dataset = FlowDataset(
            val_data, node_features, date_features, time_features,
            adjacency_matrix, station_to_idx, 
            sequence_length=12, prediction_length=1,
            target_columns=['total_flow']  # 통행량만
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
        """통행량 전용 모델 구축"""
        self.logger.info("통행량 전용 DCRNN 모델 구성 중...")
        
        model_config = self.config['model']
        
        self.model = DCRNN(
            input_size=2,  # occupancy, total_flow 입력
            hidden_size=model_config['hidden_size'],
            output_size=1,  # 통행량만 출력
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
        # 🆕 통행량에 특화된 손실 함수 (Huber Loss - 이상치에 강함)
        self.loss_fn = nn.SmoothL1Loss(beta=100.0)  # 통행량 스케일에 맞춤
        
        # 🆕 통행량에 특화된 옵티마이저 (더 높은 학습률)
        opt_config = self.config['optimizer']
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=opt_config['lr'] * 1.5,  # 통행량은 더 높은 학습률
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
        
        self.logger.info("통행량 모델 훈련 설정 완료")
    
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
            # 데이터 준비
            X = batch['X'].to(self.device)  # [B, T, N, 2]
            Y = batch['Y'].to(self.device)  # [B, N, 1] - 통행량만
            
            # 인접행렬 확장
            batch_size = X.size(0)
            adjacency = self.adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Forward pass
            predictions, _ = self.model(
                X, adjacency, 
                target_length=1,
                teacher_forcing_ratio=teacher_forcing_ratio,
                targets=Y.unsqueeze(1)  # [B, 1, N, 1]
            )
            
            predictions = predictions.squeeze(1)  # [B, N, 1]
            
            # 🆕 통행량 예측값을 양수로 제한
            predictions = torch.clamp(predictions, min=0.0, max=5000.0)
            
            # 예측 범위 업데이트
            pred_min = min(pred_min, predictions.min().item())
            pred_max = max(pred_max, predictions.max().item())
            
            # 손실 계산
            loss = self.loss_fn(predictions, Y)
            
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
                
                batch_size = X.size(0)
                adjacency = self.adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
                
                # Forward pass (no teacher forcing)
                predictions, _ = self.model(X, adjacency, target_length=1)
                predictions = predictions.squeeze(1)
                
                # 통행량 예측값을 양수로 제한
                predictions = torch.clamp(predictions, min=0.0, max=5000.0)
                
                # 손실 계산
                loss = self.loss_fn(predictions, Y)
                total_loss += loss.item()
        
        return {'loss': total_loss / num_batches}
    
    def save_checkpoint(self, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint_dir = Path("checkpoints/flow_model")
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
            best_path = checkpoint_dir / "best_flow_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best flow model saved: {best_path}")
    
    def train(self):
        """전체 학습 루프"""
        self.logger.info("통행량 모델 훈련 시작...")
        
        training_config = self.config['training']
        early_stopping_config = training_config['early_stopping']
        
        for epoch in range(training_config['epochs']):
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
            self.logger.info(
                f"통행량 예측 범위: [{pred_range[0]:.1f}, {pred_range[1]:.1f}]명"
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
        
        self.logger.info("통행량 모델 훈련 완료!")
        
        # 최종 체크포인트 저장
        self.save_checkpoint()
        
        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Flow DCRNN model')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 트레이너 생성
    trainer = FlowTrainer(config)
    
    # 데이터 로드
    trainer.load_data()
    
    # 모델 구축
    trainer.build_model()
    
    # 학습 설정
    trainer.setup_training()
    
    # 학습 시작
    trainer.train()


if __name__ == '__main__':
    main()
