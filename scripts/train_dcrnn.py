#!/usr/bin/env python3
"""
DCRNN 모델 학습 스크립트

사용법:
    python scripts/train_dcrnn.py --config config/model_config.yaml
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
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN
from models.losses import get_loss_function, TemporalConsistencyLoss, SpatialConsistencyLoss
from models.dataset import SubwayGraphDataset


class DCRNNTrainer:
    """DCRNN 모델 학습 클래스"""
    
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
            log_dir = Path(config['experiment']['logging']['log_dir']) / config['experiment']['name']
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
        logger = logging.getLogger('DCRNN')
        logger.setLevel(getattr(logging, log_config['level']))
        
        # 파일 핸들러
        log_file = log_dir / f"{self.config['experiment']['name']}.log"
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
        self.logger.info("Loading datasets...")
        
        data_config = self.config['data']
        
        try:
            # 기존 데이터셋 로드 시도
            with open(data_config['train_path'], 'rb') as f:
                train_dataset = pickle.load(f)
            
            with open(data_config['val_path'], 'rb') as f:
                val_dataset = pickle.load(f)
                
            # 호환성 확인
            sample = train_dataset[0]
            if not all(key in sample for key in ['X', 'Y', 'adjacency', 'node_features']):
                raise ValueError("Incompatible dataset format")
                
        except (FileNotFoundError, ValueError, KeyError) as e:
            self.logger.warning(f"Cannot load existing datasets: {e}")
            self.logger.info("Creating new datasets from raw data...")
            
            # 원본 데이터에서 새로 생성
            train_dataset, val_dataset = self._create_datasets_from_raw()
        
        # 인접행렬 로드
        self.adjacency_matrix = torch.FloatTensor(np.load(data_config['adjacency_path']))
        self.adjacency_matrix = self.adjacency_matrix.to(self.device)
        
        # 🔧 디바이스별 DataLoader 최적화
        use_cuda = self.device.type == 'cuda'
        use_mps = self.device.type == 'mps'
        num_workers = 0 if use_mps else data_config['num_workers']
        pin_memory = data_config['pin_memory'] and use_cuda
        
        # 데이터 로더 생성
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        self.logger.info(f"Train batches: {len(self.train_loader)}")
        self.logger.info(f"Val batches: {len(self.val_loader)}")
    
    def _create_datasets_from_raw(self):
        """원본 데이터에서 데이터셋 생성"""
        import pandas as pd
        
        # 원본 데이터 로드
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
        
        train_dataset = SubwayGraphDataset(
            train_data, node_features, date_features, time_features,
            adjacency_matrix, station_to_idx, sequence_length=12, prediction_length=1
        )
        
        val_dataset = SubwayGraphDataset(
            val_data, node_features, date_features, time_features,
            adjacency_matrix, station_to_idx, sequence_length=12, prediction_length=1
        )
        
        return train_dataset, val_dataset
    
    def build_model(self):
        """모델 구축"""
        self.logger.info("Building DCRNN model...")
        
        model_config = self.config['model']
        
        self.model = DCRNN(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            output_size=model_config['output_size'],
            num_layers=model_config['num_layers'],
            diffusion_steps=model_config['diffusion_steps'],
            use_attention=model_config['use_attention'],
            dropout=model_config['dropout']
        ).to(self.device)
        
        # 모델 컴파일 (PyTorch 2.0+)
        if self.config['hardware']['compile_model']:
            self.model = torch.compile(self.model)
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def setup_training(self):
        """학습 설정"""
        # 손실 함수
        self.loss_fn = get_loss_function(self.config['loss'])
        
        # 정규화 손실
        self.temporal_loss = None
        self.spatial_loss = None
        
        if self.config['loss'].get('temporal_consistency', {}).get('enabled', False):
            weight = self.config['loss']['temporal_consistency']['weight']
            self.temporal_loss = TemporalConsistencyLoss(weight)
        
        if self.config['loss'].get('spatial_consistency', {}).get('enabled', False):
            weight = self.config['loss']['spatial_consistency']['weight']
            self.spatial_loss = SpatialConsistencyLoss(weight)
        
        # 옵티마이저
        opt_config = self.config['optimizer']
        
        if opt_config['type'] == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay'],
                betas=opt_config['betas'],
                eps=opt_config['eps']
            )
        elif opt_config['type'] == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay'],
                betas=opt_config['betas'],
                eps=opt_config['eps']
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['type']}")
        
        # 스케줄러
        sched_config = self.config['scheduler']
        
        if sched_config['type'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config['T_max'],
                eta_min=sched_config['eta_min']
            )
        elif sched_config['type'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
        elif sched_config['type'] == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=sched_config['patience'],
                factor=sched_config['factor'],
                threshold=sched_config['threshold']
            )
        
        self.logger.info("Training setup completed")
    
    def get_teacher_forcing_ratio(self, epoch: int) -> float:
        """Teacher forcing 비율 계산"""
        tf_config = self.config['training']['teacher_forcing']
        
        if epoch >= tf_config['decay_epochs']:
            return tf_config['final_ratio']
        
        # 🆕 더 부드러운 감소 (cosine decay)
        progress = epoch / tf_config['decay_epochs']
        ratio = tf_config['final_ratio'] + (tf_config['initial_ratio'] - tf_config['final_ratio']) * \
                (1 + np.cos(np.pi * progress)) / 2
        
        return ratio
    
    def train_epoch(self) -> Dict[str, float]:
        """한 에포크 학습"""
        self.model.train()
        
        total_loss = 0.0
        total_occupancy_loss = 0.0
        total_flow_loss = 0.0
        num_batches = len(self.train_loader)
        
        # 🆕 예측 범위 모니터링
        pred_occupancy_min, pred_occupancy_max = float('inf'), float('-inf')
        pred_flow_min, pred_flow_max = float('inf'), float('-inf')
        # 🆕 실제 값 범위 모니터링
        true_occupancy_min, true_occupancy_max = float('inf'), float('-inf')
        true_flow_min, true_flow_max = float('inf'), float('-inf')
        
        teacher_forcing_ratio = self.get_teacher_forcing_ratio(self.current_epoch)
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            # 데이터 준비
            self.logger.debug(f"Batch {batch_idx}: 데이터 준비 시작")
            X = batch['X'].to(self.device)  # [B, T, N, 2]
            Y = batch['Y'].to(self.device)  # [B, N, 2]
            
            # 인접행렬 확장
            batch_size = X.size(0)
            adjacency = self.adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
            
            self.logger.debug(f"Batch {batch_idx}: Forward pass 시작")
            # 🔥 모든 특성 추출
            node_features = batch['node_features'].to(self.device)  # [B, N, node_dim]
            date_features = batch['date_features'].to(self.device)  # [B, date_dim]
            time_features = batch['time_features'].to(self.device)  # [B, time_dim]
            
            # Forward pass (모든 특성 활용)
            predictions, attention_weights = self.model(
                X, adjacency, 
                target_length=1,
                teacher_forcing_ratio=teacher_forcing_ratio,
                targets=Y.unsqueeze(1),  # [B, 1, N, 2]
                node_features=node_features,
                date_features=date_features,
                time_features=time_features
            )
            
            predictions = predictions.squeeze(1)  # [B, N, 2]
            self.logger.debug(f"Batch {batch_idx}: Forward pass 완료")
            
            # 🆕 예측 범위 업데이트
            pred_occupancy_min = min(pred_occupancy_min, predictions[:, :, 0].min().item())
            pred_occupancy_max = max(pred_occupancy_max, predictions[:, :, 0].max().item())
            pred_flow_min = min(pred_flow_min, predictions[:, :, 1].min().item())
            pred_flow_max = max(pred_flow_max, predictions[:, :, 1].max().item())
            
            # 🆕 실제 값 범위 업데이트
            true_occupancy_min = min(true_occupancy_min, Y[:, :, 0].min().item())
            true_occupancy_max = max(true_occupancy_max, Y[:, :, 0].max().item())
            true_flow_min = min(true_flow_min, Y[:, :, 1].min().item())
            true_flow_max = max(true_flow_max, Y[:, :, 1].max().item())
            
            # 손실 계산
            self.logger.debug(f"Batch {batch_idx}: 손실 계산 시작")
            loss_dict = self.loss_fn(predictions, Y)
            loss = loss_dict['total_loss']
            
            # 정규화 손실 추가
            if self.temporal_loss is not None:
                temp_loss = self.temporal_loss(predictions.unsqueeze(1))
                loss += temp_loss
            
            if self.spatial_loss is not None:
                spatial_loss = self.spatial_loss(predictions.unsqueeze(1), adjacency)
                loss += spatial_loss
            
            self.logger.debug(f"Batch {batch_idx}: Backward pass 시작")
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # 그래디언트 클리핑
            if self.config['training']['gradient_clipping']['enabled']:
                max_norm = self.config['training']['gradient_clipping']['max_norm']
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            
            self.optimizer.step()
            self.logger.debug(f"Batch {batch_idx}: Optimizer step 완료")
            
            # 통계 업데이트
            total_loss += loss.item()
            total_occupancy_loss += loss_dict['occupancy_loss'].item()
            total_flow_loss += loss_dict['flow_loss'].item()
            
            # 로깅
            if batch_idx % 50 == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {loss.item():.4f}, "
                    f"TF Ratio: {teacher_forcing_ratio:.3f}"
                )
        
        return {
            'loss': total_loss / num_batches,
            'occupancy_loss': total_occupancy_loss / num_batches,
            'flow_loss': total_flow_loss / num_batches,
            'teacher_forcing_ratio': teacher_forcing_ratio,
            # 🆕 예측 범위 정보
            'pred_occupancy_range': [pred_occupancy_min, pred_occupancy_max],
            'pred_flow_range': [pred_flow_min, pred_flow_max],
            # 🆕 실제 값 범위 정보
            'true_occupancy_range': [true_occupancy_min, true_occupancy_max],
            'true_flow_range': [true_flow_min, true_flow_max]
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """한 에포크 검증"""
        self.model.eval()
        
        total_loss = 0.0
        total_occupancy_loss = 0.0
        total_flow_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 데이터 준비
                X = batch['X'].to(self.device)
                Y = batch['Y'].to(self.device)
                
                batch_size = X.size(0)
                adjacency = self.adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
                
                # Forward pass (no teacher forcing)
                predictions, _ = self.model(X, adjacency, target_length=1)
                predictions = predictions.squeeze(1)
                
                # 손실 계산
                loss_dict = self.loss_fn(predictions, Y)
                
                total_loss += loss_dict['total_loss'].item()
                total_occupancy_loss += loss_dict['occupancy_loss'].item()
                total_flow_loss += loss_dict['flow_loss'].item()
        
        return {
            'loss': total_loss / num_batches,
            'occupancy_loss': total_occupancy_loss / num_batches,
            'flow_loss': total_flow_loss / num_batches
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint_dir = Path(self.config['training']['checkpoint']['save_dir'])
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
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved: {best_path}")
    
    def train(self, start_epoch: int = 0):
        """전체 학습 루프"""
        self.logger.info("Starting training...")
        
        training_config = self.config['training']
        early_stopping_config = training_config['early_stopping']
        
        for epoch in range(start_epoch, training_config['epochs']):
            self.current_epoch = epoch
            start_time = time.time()
            
            # 학습
            train_metrics = self.train_epoch()
            
            # 검증
            val_metrics = self.validate_epoch()
            
            # 스케줄러 업데이트
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
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
            
            # 🆕 예측 범위 로깅
            if 'pred_occupancy_range' in train_metrics:
                pred_occ = train_metrics['pred_occupancy_range']
                pred_flow = train_metrics['pred_flow_range']
                true_occ = train_metrics['true_occupancy_range']
                true_flow = train_metrics['true_flow_range']
                self.logger.info(
                    f"예측 범위 - 혼잡도: [{pred_occ[0]:.1f}, {pred_occ[1]:.1f}] "
                    f"(실제: [{true_occ[0]:.1f}, {true_occ[1]:.1f}])"
                )
                self.logger.info(
                    f"예측 범위 - 통행량: [{pred_flow[0]:.1f}, {pred_flow[1]:.1f}] "
                    f"(실제: [{true_flow[0]:.1f}, {true_flow[1]:.1f}])"
            )
            
            # TensorBoard 로깅
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/Train_Occupancy', train_metrics['occupancy_loss'], epoch)
                self.writer.add_scalar('Loss/Val_Occupancy', val_metrics['occupancy_loss'], epoch)
                self.writer.add_scalar('Loss/Train_Flow', train_metrics['flow_loss'], epoch)
                self.writer.add_scalar('Loss/Val_Flow', val_metrics['flow_loss'], epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                self.writer.add_scalar('Teacher_Forcing_Ratio', train_metrics['teacher_forcing_ratio'], epoch)
            
            # 체크포인트 저장
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # 정기 저장: 현재 상태만
            if epoch % training_config['checkpoint']['save_every'] == 0:
                self.save_checkpoint(is_best=False)
            
            # 최고 성능 모델 별도 저장
            if is_best and training_config['checkpoint']['save_best']:
                self.save_checkpoint(is_best=True)
            
            # Early stopping
            if early_stopping_config['enabled']:
                if self.early_stopping_counter >= early_stopping_config['patience']:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        self.logger.info("Training completed!")
        
        # 최종 체크포인트 저장
        self.save_checkpoint()
        
        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train DCRNN model')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 트레이너 생성
    trainer = DCRNNTrainer(config)
    
    # 데이터 로드
    trainer.load_data()
    
    # 모델 구축
    trainer.build_model()
    
    # 학습 설정
    trainer.setup_training()
    
    # 체크포인트 복원 및 시작 에포크 설정
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint['epoch']}, starting at epoch {start_epoch}")
    
    # 학습 시작
    trainer.train(start_epoch)


if __name__ == '__main__':
    main()
