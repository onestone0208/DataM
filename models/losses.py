#!/usr/bin/env python3
"""
DCRNN 손실 함수 구현

멀티태스크 학습을 위한 다양한 손실 함수:
- 혼잡도 예측 (occupancy)
- 통행량 예측 (total_flow)
- 정규화 및 안정성 개선
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class MultiTaskLoss(nn.Module):
    """
    멀티태스크 손실 함수
    
    occupancy와 total_flow를 동시에 예측하는 손실 함수
    """
    
    def __init__(self,
                 occupancy_weight: float = 1.0,
                 flow_weight: float = 1.0,
                 loss_type: str = 'mae',
                 adaptive_weights: bool = True,
                 uncertainty_weighting: bool = False):
        super(MultiTaskLoss, self).__init__()
        
        self.occupancy_weight = occupancy_weight
        self.flow_weight = flow_weight
        self.loss_type = loss_type
        self.adaptive_weights = adaptive_weights
        self.uncertainty_weighting = uncertainty_weighting
        
        # Loss functions
        if loss_type == 'mae':
            self.base_loss = nn.L1Loss(reduction='none')
        elif loss_type == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        elif loss_type == 'huber':
            self.base_loss = nn.SmoothL1Loss(reduction='none', beta=1.0)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        # Uncertainty-based weighting parameters (if enabled)
        if uncertainty_weighting:
            self.log_var_occupancy = nn.Parameter(torch.zeros(1))
            self.log_var_flow = nn.Parameter(torch.zeros(1))
        
        # Adaptive weight tracking
        if adaptive_weights:
            self.register_buffer('occupancy_loss_history', torch.zeros(100))
            self.register_buffer('flow_loss_history', torch.zeros(100))
            self.register_buffer('step_count', torch.zeros(1))
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            predictions: [batch_size, target_length, num_nodes, 2] - (occupancy, flow)
            targets: [batch_size, target_length, num_nodes, 2] - (occupancy, flow)
            mask: [batch_size, target_length, num_nodes] - valid data mask
            
        Returns:
            loss_dict: Dictionary containing individual and total losses
        """
        # 🆕 데이터 스케일링 적용 (통행량을 1000으로 나누어 혼잡도와 비슷한 스케일로)
        scaled_predictions = predictions.clone()
        scaled_targets = targets.clone()
        scaled_predictions[..., 1] = predictions[..., 1] / 1000.0  # 통행량 스케일링
        scaled_targets[..., 1] = targets[..., 1] / 1000.0
        
        # Split predictions and targets
        pred_occupancy = scaled_predictions[..., 0]  # [B, T, N]
        pred_flow = scaled_predictions[..., 1]       # [B, T, N]
        
        target_occupancy = scaled_targets[..., 0]    # [B, T, N]
        target_flow = scaled_targets[..., 1]         # [B, T, N]
        
        # Compute individual losses
        occupancy_loss = self.base_loss(pred_occupancy, target_occupancy)  # [B, T, N]
        flow_loss = self.base_loss(pred_flow, target_flow)                 # [B, T, N]
        
        # Apply mask if provided
        if mask is not None:
            occupancy_loss = occupancy_loss * mask
            flow_loss = flow_loss * mask
            valid_count = mask.sum()
        else:
            valid_count = torch.numel(occupancy_loss)
        
        # Reduce losses
        occupancy_loss_mean = occupancy_loss.sum() / valid_count
        flow_loss_mean = flow_loss.sum() / valid_count
        
        # Update adaptive weights
        if self.adaptive_weights and self.training:
            self._update_adaptive_weights(occupancy_loss_mean, flow_loss_mean)
        
        # Compute weighted loss
        if self.uncertainty_weighting:
            # Uncertainty-based weighting (Kendall & Gal, 2017)
            precision_occupancy = torch.exp(-self.log_var_occupancy)
            precision_flow = torch.exp(-self.log_var_flow)
            
            weighted_occupancy = precision_occupancy * occupancy_loss_mean + self.log_var_occupancy
            weighted_flow = precision_flow * flow_loss_mean + self.log_var_flow
            
            total_loss = weighted_occupancy + weighted_flow
        else:
            # Manual or adaptive weighting
            occ_weight = self.occupancy_weight
            flow_weight = self.flow_weight
            
            if self.adaptive_weights:
                occ_weight, flow_weight = self._get_adaptive_weights()
            
            total_loss = occ_weight * occupancy_loss_mean + flow_weight * flow_loss_mean
        
        return {
            'total_loss': total_loss,
            'occupancy_loss': occupancy_loss_mean,
            'flow_loss': flow_loss_mean,
            'occupancy_weight': occ_weight if not self.uncertainty_weighting else precision_occupancy,
            'flow_weight': flow_weight if not self.uncertainty_weighting else precision_flow
        }
    
    def _update_adaptive_weights(self, occupancy_loss: torch.Tensor, flow_loss: torch.Tensor):
        """적응적 가중치 업데이트"""
        step = int(self.step_count.item()) % 100
        
        self.occupancy_loss_history[step] = occupancy_loss.detach()
        self.flow_loss_history[step] = flow_loss.detach()
        self.step_count += 1
    
    def _get_adaptive_weights(self) -> Tuple[float, float]:
        """적응적 가중치 계산"""
        if self.step_count < 10:  # Not enough history
            return self.occupancy_weight, self.flow_weight
        
        # Use recent loss history
        recent_steps = min(int(self.step_count.item()), 100)
        occ_mean = self.occupancy_loss_history[:recent_steps].mean()
        flow_mean = self.flow_loss_history[:recent_steps].mean()
        
        # Inverse weighting: higher loss gets lower weight
        total_loss = occ_mean + flow_mean
        if total_loss > 0:
            occ_weight = flow_mean / total_loss
            flow_weight = occ_mean / total_loss
        else:
            occ_weight = self.occupancy_weight
            flow_weight = self.flow_weight
        
        return float(occ_weight), float(flow_weight)


class MaskedLoss(nn.Module):
    """
    마스크 기반 손실 함수
    
    결측값이나 특정 조건을 제외한 손실 계산
    """
    
    def __init__(self, loss_fn: nn.Module, null_val: float = 0.0):
        super(MaskedLoss, self).__init__()
        self.loss_fn = loss_fn
        self.null_val = null_val
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic masking
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            masked_loss: Loss computed only on valid (non-null) values
        """
        # Create mask for valid values
        mask = (targets != self.null_val)
        
        if mask.sum() == 0:  # No valid values
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Compute loss only on valid values
        valid_predictions = predictions[mask]
        valid_targets = targets[mask]
        
        return self.loss_fn(valid_predictions, valid_targets)


class QuantileLoss(nn.Module):
    """
    분위수 손실 함수
    
    불확실성 정량화를 위한 분위수 회귀
    """
    
    def __init__(self, quantiles: list = [0.1, 0.5, 0.9]):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            predictions: [batch_size, ..., num_quantiles] - Quantile predictions
            targets: [batch_size, ...] - Ground truth targets
            
        Returns:
            quantile_loss: Combined quantile loss
        """
        targets = targets.unsqueeze(-1)  # Add quantile dimension
        
        errors = targets - predictions  # [B, ..., Q]
        
        total_loss = 0.0
        for i, q in enumerate(self.quantiles):
            quantile_loss = torch.max(q * errors[..., i], (q - 1) * errors[..., i])
            total_loss += quantile_loss.mean()
        
        return total_loss / len(self.quantiles)


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced regression
    
    극값 예측에 더 집중하는 손실 함수
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            focal_loss: Focal loss value
        """
        # L1 loss as base
        l1_loss = F.l1_loss(predictions, targets, reduction='none')
        
        # Focal weight: (1 + |error|)^gamma
        focal_weight = torch.pow(1 + l1_loss, self.gamma)
        
        # Apply focal weight
        focal_loss = self.alpha * focal_weight * l1_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TemporalConsistencyLoss(nn.Module):
    """
    시간적 일관성 손실
    
    연속된 예측 간의 급격한 변화를 억제
    """
    
    def __init__(self, weight: float = 0.1):
        super(TemporalConsistencyLoss, self).__init__()
        self.weight = weight
    
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            predictions: [batch_size, seq_len, num_nodes, features] - Sequential predictions
            
        Returns:
            consistency_loss: Temporal consistency penalty
        """
        if predictions.size(1) < 2:  # Need at least 2 time steps
            return torch.tensor(0.0, device=predictions.device)
        
        # Compute differences between consecutive predictions
        diff = predictions[:, 1:] - predictions[:, :-1]  # [B, T-1, N, F]
        
        # L2 penalty on differences
        consistency_loss = torch.mean(diff ** 2)
        
        return self.weight * consistency_loss


class SpatialConsistencyLoss(nn.Module):
    """
    공간적 일관성 손실
    
    인접한 역들 간의 예측 일관성 유지
    """
    
    def __init__(self, weight: float = 0.1):
        super(SpatialConsistencyLoss, self).__init__()
        self.weight = weight
    
    def forward(self, predictions: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            predictions: [batch_size, ..., num_nodes, features] - Node predictions
            adjacency: [batch_size, num_nodes, num_nodes] - Graph adjacency matrix
            
        Returns:
            spatial_loss: Spatial consistency penalty
        """
        batch_size = predictions.size(0)
        num_nodes = predictions.size(-2)
        
        # Flatten spatial dimensions for easier computation
        pred_flat = predictions.view(batch_size, -1, num_nodes, predictions.size(-1))
        
        spatial_loss = 0.0
        for b in range(batch_size):
            adj = adjacency[b]  # [N, N]
            
            for t in range(pred_flat.size(1)):
                pred_t = pred_flat[b, t]  # [N, F]
                
                # Compute weighted differences with neighbors
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if adj[i, j] > 0:  # Connected nodes
                            weight = adj[i, j]
                            diff = pred_t[i] - pred_t[j]  # [F]
                            spatial_loss += weight * torch.sum(diff ** 2)
        
        # Normalize by batch size and connections
        total_connections = (adjacency > 0).sum().float()
        if total_connections > 0:
            spatial_loss = spatial_loss / total_connections
        
        return self.weight * spatial_loss


def get_loss_function(config: dict) -> nn.Module:
    """
    손실 함수 팩토리
    
    Args:
        config: Loss configuration dictionary
        
    Returns:
        loss_fn: Configured loss function
    """
    loss_type = config.get('type', 'multitask')
    
    if loss_type == 'multitask':
        return MultiTaskLoss(
            occupancy_weight=config.get('occupancy_weight', 1.0),
            flow_weight=config.get('flow_weight', 1.0),
            loss_type=config.get('base_loss', 'mae'),
            adaptive_weights=config.get('adaptive_weights', True),
            uncertainty_weighting=config.get('uncertainty_weighting', False)
        )
    
    elif loss_type == 'masked':
        base_loss = nn.L1Loss() if config.get('base_loss', 'mae') == 'mae' else nn.MSELoss()
        return MaskedLoss(base_loss, null_val=config.get('null_val', 0.0))
    
    elif loss_type == 'quantile':
        return QuantileLoss(quantiles=config.get('quantiles', [0.1, 0.5, 0.9]))
    
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=config.get('alpha', 1.0),
            gamma=config.get('gamma', 2.0),
            reduction=config.get('reduction', 'mean')
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
