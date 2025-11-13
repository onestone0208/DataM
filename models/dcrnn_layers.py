#!/usr/bin/env python3
"""
DCRNN 핵심 레이어 구현

Diffusion Convolutional Recurrent Neural Network의 핵심 컴포넌트들:
- Diffusion Convolution
- DCGRU Cell
- Attention Mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class DiffusionConvolution(nn.Module):
    """
    Diffusion Convolution Layer
    
    논문 구현: X *_G f_θ = Σ(k=0 to K-1) [θ_k,1 * (D_O^-1 * W)^k + θ_k,2 * (D_I^-1 * W^T)^k] * X
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int, 
                 diffusion_steps: int = 3,
                 bias: bool = True):
        super(DiffusionConvolution, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.diffusion_steps = diffusion_steps
        
        # θ_k,1 and θ_k,2 parameters for each diffusion step
        self.weight_forward = nn.Parameter(torch.FloatTensor(diffusion_steps, in_channels, out_channels))
        self.weight_backward = nn.Parameter(torch.FloatTensor(diffusion_steps, in_channels, out_channels))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """파라미터 초기화"""
        std = 1.0 / math.sqrt(self.in_channels)
        self.weight_forward.data.uniform_(-std, std)
        self.weight_backward.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)
    
    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [batch_size, num_nodes, in_channels]
            adjacency: [batch_size, num_nodes, num_nodes] or [num_nodes, num_nodes]
            
        Returns:
            output: [batch_size, num_nodes, out_channels]
        """
        batch_size, num_nodes, _ = x.shape
        
        # Adjacency matrix normalization
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Out-degree normalization: D_O^-1 * W
        out_degree = adjacency.sum(dim=2, keepdim=True)  # [B, N, 1]
        out_degree = torch.clamp(out_degree, min=1e-6)  # Avoid division by zero
        forward_adj = adjacency / out_degree  # [B, N, N]
        
        # In-degree normalization: D_I^-1 * W^T  
        in_degree = adjacency.sum(dim=1, keepdim=True)  # [B, 1, N]
        in_degree = torch.clamp(in_degree, min=1e-6)
        backward_adj = adjacency.transpose(1, 2) / in_degree.transpose(1, 2)  # [B, N, N]
        
        # Diffusion convolution
        output = torch.zeros(batch_size, num_nodes, self.out_channels, device=x.device)
        
        # Forward diffusion: (D_O^-1 * W)^k
        forward_x = x  # [B, N, C_in]
        for k in range(self.diffusion_steps):
            # Apply diffusion: X * θ_k,1
            diffused = torch.matmul(forward_x, self.weight_forward[k])  # [B, N, C_out]
            output += diffused
            
            # Next diffusion step: (D_O^-1 * W) * X
            if k < self.diffusion_steps - 1:
                forward_x = torch.bmm(forward_adj, forward_x)  # [B, N, C_in]
        
        # Backward diffusion: (D_I^-1 * W^T)^k
        backward_x = x  # [B, N, C_in]
        for k in range(self.diffusion_steps):
            # Apply diffusion: X * θ_k,2
            diffused = torch.matmul(backward_x, self.weight_backward[k])  # [B, N, C_out]
            output += diffused
            
            # Next diffusion step: (D_I^-1 * W^T) * X
            if k < self.diffusion_steps - 1:
                backward_x = torch.bmm(backward_adj, backward_x)  # [B, N, C_in]
        
        # Add bias
        if self.bias is not None:
            output += self.bias
        
        return output


class DCGRUCell(nn.Module):
    """
    Diffusion Convolutional GRU Cell
    
    GRU with diffusion convolution instead of linear transformations
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 diffusion_steps: int = 3,
                 bias: bool = True):
        super(DCGRUCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.diffusion_steps = diffusion_steps
        
        # Reset gate: r = σ(W_r * [x, h] + b_r)
        self.reset_gate_x = DiffusionConvolution(input_size, hidden_size, diffusion_steps, bias)
        self.reset_gate_h = DiffusionConvolution(hidden_size, hidden_size, diffusion_steps, bias=False)
        
        # Update gate: z = σ(W_z * [x, h] + b_z)  
        self.update_gate_x = DiffusionConvolution(input_size, hidden_size, diffusion_steps, bias)
        self.update_gate_h = DiffusionConvolution(hidden_size, hidden_size, diffusion_steps, bias=False)
        
        # Candidate state: h_tilde = tanh(W_h * [x, r ⊙ h] + b_h)
        self.candidate_x = DiffusionConvolution(input_size, hidden_size, diffusion_steps, bias)
        self.candidate_h = DiffusionConvolution(hidden_size, hidden_size, diffusion_steps, bias=False)
    
    def forward(self, 
                x: torch.Tensor, 
                hidden: torch.Tensor, 
                adjacency: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [batch_size, num_nodes, input_size]
            hidden: [batch_size, num_nodes, hidden_size]
            adjacency: [batch_size, num_nodes, num_nodes]
            
        Returns:
            new_hidden: [batch_size, num_nodes, hidden_size]
        """
        # Reset gate
        reset_x = self.reset_gate_x(x, adjacency)
        reset_h = self.reset_gate_h(hidden, adjacency)
        reset_gate = torch.sigmoid(reset_x + reset_h)
        
        # Update gate
        update_x = self.update_gate_x(x, adjacency)
        update_h = self.update_gate_h(hidden, adjacency)
        update_gate = torch.sigmoid(update_x + update_h)
        
        # Candidate hidden state
        candidate_x = self.candidate_x(x, adjacency)
        candidate_h = self.candidate_h(reset_gate * hidden, adjacency)
        candidate = torch.tanh(candidate_x + candidate_h)
        
        # New hidden state
        new_hidden = (1 - update_gate) * hidden + update_gate * candidate
        
        return new_hidden


class AttentionMechanism(nn.Module):
    """
    Attention Mechanism for DCRNN
    
    Additive attention to focus on important time steps
    """
    
    def __init__(self, 
                 hidden_size: int,
                 attention_size: int = None):
        super(AttentionMechanism, self).__init__()
        
        if attention_size is None:
            attention_size = hidden_size
        
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        
        # Attention layers
        self.W_h = nn.Linear(hidden_size, attention_size, bias=False)  # Hidden state projection
        self.W_s = nn.Linear(hidden_size, attention_size, bias=False)  # Query projection  
        self.v = nn.Linear(attention_size, 1, bias=False)  # Attention score
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, 
                hidden_states: torch.Tensor, 
                query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            hidden_states: [batch_size, seq_len, num_nodes, hidden_size]
            query: [batch_size, num_nodes, hidden_size] - current decoder state
            
        Returns:
            context: [batch_size, num_nodes, hidden_size] - attended context
            attention_weights: [batch_size, seq_len] - attention weights
        """
        batch_size, seq_len, num_nodes, hidden_size = hidden_states.shape
        
        # Reshape for attention computation
        hidden_flat = hidden_states.view(batch_size * seq_len, num_nodes, hidden_size)
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1, -1)
        query_flat = query_expanded.contiguous().view(batch_size * seq_len, num_nodes, hidden_size)
        
        # Compute attention scores
        h_proj = self.W_h(hidden_flat)  # [B*T, N, A]
        s_proj = self.W_s(query_flat)   # [B*T, N, A]
        
        # Additive attention: v^T * tanh(W_h * h + W_s * s)
        attention_input = self.tanh(h_proj + s_proj)  # [B*T, N, A]
        attention_scores = self.v(attention_input).squeeze(-1)  # [B*T, N]
        
        # Average over nodes to get time-level attention
        attention_scores = attention_scores.mean(dim=1)  # [B*T]
        attention_scores = attention_scores.view(batch_size, seq_len)  # [B, T]
        
        # Softmax normalization
        attention_weights = self.softmax(attention_scores)  # [B, T]
        
        # Compute weighted context
        attention_expanded = attention_weights.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
        context = (hidden_states * attention_expanded).sum(dim=1)  # [B, N, H]
        
        return context, attention_weights


class DCRNN(nn.Module):
    """
    Complete DCRNN Model with Attention
    
    Encoder-Decoder architecture with diffusion convolution and attention
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 2,
                 diffusion_steps: int = 3,
                 use_attention: bool = True,
                 dropout: float = 0.1):
        super(DCRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.encoder_layers.append(
                DCGRUCell(layer_input_size, hidden_size, diffusion_steps)
            )
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(num_layers):
            # 🔥 Decoder 첫 레이어도 input_size를 받도록 변경
            layer_input_size = input_size if i == 0 else hidden_size
            self.decoder_layers.append(
                DCGRUCell(layer_input_size, hidden_size, diffusion_steps)
            )
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionMechanism(hidden_size)
            # Context projection to match decoder input size
            self.context_projection = nn.Linear(hidden_size, input_size)  # 🔥 input_size로 변경
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, output_size)
        
        # 🔥 Output을 input_size 차원으로 확장하는 projection (다음 스텝 입력용)
        self.output_to_input_projection = nn.Linear(output_size, input_size)
        
        # 🔥 Hidden state를 input_size 차원으로 projection (디코더 초기 입력용)
        self.hidden_to_input_projection = nn.Linear(hidden_size, input_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, 
               x: torch.Tensor, 
               adjacency: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Encoder forward pass
        
        Args:
            x: [batch_size, seq_len, num_nodes, input_size]
            adjacency: [batch_size, num_nodes, num_nodes]
            
        Returns:
            all_hidden: List of hidden states for each layer
            encoder_outputs: [batch_size, seq_len, num_nodes, hidden_size]
        """
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Initialize hidden states
        hidden_states = []
        for _ in range(self.num_layers):
            hidden_states.append(torch.zeros(batch_size, num_nodes, self.hidden_size, device=x.device))
        
        # Store all encoder outputs for attention
        encoder_outputs = []
        
        # Process each time step
        for t in range(seq_len):
            layer_input = x[:, t]  # [B, N, input_size]
            
            # Forward through encoder layers
            for i, layer in enumerate(self.encoder_layers):
                hidden_states[i] = layer(layer_input, hidden_states[i], adjacency)
                layer_input = self.dropout(hidden_states[i])
            
            encoder_outputs.append(hidden_states[-1])  # Store top layer output
        
        encoder_outputs = torch.stack(encoder_outputs, dim=1)  # [B, T, N, H]
        
        return hidden_states, encoder_outputs
    
    def decode(self, 
               encoder_hidden: List[torch.Tensor],
               encoder_outputs: torch.Tensor,
               adjacency: torch.Tensor,
               target_length: int,
               teacher_forcing_ratio: float = 0.0,
               targets: Optional[torch.Tensor] = None,
               initial_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decoder forward pass
        
        Args:
            encoder_hidden: Final encoder hidden states
            encoder_outputs: All encoder outputs for attention
            adjacency: Graph adjacency matrix
            target_length: Number of steps to predict
            teacher_forcing_ratio: Probability of using teacher forcing
            targets: Ground truth targets for teacher forcing
            initial_input: Initial decoder input [B, N, input_size] (optional)
            
        Returns:
            outputs: [batch_size, target_length, num_nodes, output_size]
            attention_weights: [batch_size, target_length, seq_len] if using attention
        """
        batch_size, num_nodes = encoder_hidden[0].shape[:2]
        
        # Initialize decoder hidden states with encoder final states
        hidden_states = encoder_hidden.copy()
        
        outputs = []
        attention_weights_list = []
        
        # 🔥 Initial decoder input: Encoder의 마지막 입력 사용 (input_size 차원)
        if initial_input is not None:
            decoder_input = initial_input  # [B, N, input_size]
        else:
            # Fallback: zero tensor with input_size dimension
            decoder_input = torch.zeros(batch_size, num_nodes, self.input_size, device=encoder_outputs.device)
        
        for t in range(target_length):
            # Teacher forcing: targets를 input_size 차원으로 확장
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # targets: [B, T, N, output_size] -> [B, N, output_size]
                target_output = targets[:, t]  # [B, N, output_size=1]
                # 🔥 output_size를 input_size로 확장
                decoder_input = self.output_to_input_projection(target_output)  # [B, N, input_size]
            
            # Attention mechanism
            if self.use_attention:
                context, attention_weights = self.attention(encoder_outputs, hidden_states[-1])
                attention_weights_list.append(attention_weights)
                
                # Context projection to match decoder input size
                projected_context = self.context_projection(context)  # [B, N, input_size]
                combined_input = decoder_input + projected_context  # Residual connection
            else:
                combined_input = decoder_input
                attention_weights = None
            
            # Forward through decoder layers
            layer_input = combined_input
            for i, layer in enumerate(self.decoder_layers):
                hidden_states[i] = layer(layer_input, hidden_states[i], adjacency)
                layer_input = self.dropout(hidden_states[i])
            
            # Output projection
            output = self.output_projection(hidden_states[-1])  # [B, N, output_size]
            outputs.append(output)
            
            # 🔥 Next decoder input: output을 input_size 차원으로 확장
            # Teacher forcing이 아닐 때만 다음 입력으로 사용 (다양성 확보)
            if targets is None or torch.rand(1).item() >= teacher_forcing_ratio:
                decoder_input = self.output_to_input_projection(output)  # [B, N, input_size]
                
                # 🔥 디코더 collapse 방지: 예측값에 작은 노이즈 추가 (평가 시에는 제외)
                if self.training:
                    noise_scale = 0.005
                    noise = torch.randn_like(decoder_input) * noise_scale
                    decoder_input = decoder_input + noise
        
        outputs = torch.stack(outputs, dim=1)  # [B, target_length, N, output_size]
        
        if attention_weights_list:
            attention_weights = torch.stack(attention_weights_list, dim=1)  # [B, target_length, seq_len]
        else:
            attention_weights = None
        
        return outputs, attention_weights
    
    def forward(self, 
                x: torch.Tensor,
                adjacency: torch.Tensor,
                target_length: int = 1,
                teacher_forcing_ratio: float = 0.0,
                targets: Optional[torch.Tensor] = None,
                node_features: Optional[torch.Tensor] = None,
                date_features: Optional[torch.Tensor] = None,
                time_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Complete forward pass
        
        Args:
            x: [batch_size, seq_len, num_nodes, input_size]
            adjacency: [batch_size, num_nodes, num_nodes]
            target_length: Number of prediction steps
            teacher_forcing_ratio: Teacher forcing probability
            targets: Ground truth for teacher forcing
            node_features: [batch_size, num_nodes, node_feat_dim] - 역별 특성
            date_features: [batch_size, date_feat_dim] - 날짜 특성
            time_features: [batch_size, time_feat_dim] - 시간 특성
            
        Returns:
            predictions: [batch_size, target_length, num_nodes, output_size]
            attention_weights: [batch_size, target_length, seq_len] or None
        """
        # 🔥 추가 특성들을 시계열 데이터에 결합
        # date_features와 time_features는 이미 x에 포함되어 있을 수 있으므로
        # node_features만 추가하는 것이 안전함
        if node_features is not None:
            x = self._augment_features(x, node_features, None, None)
        
        # Encode
        encoder_hidden, encoder_outputs = self.encode(x, adjacency)
        
        # 🔥 Encoder의 마지막 timestep 입력을 Decoder의 첫 입력으로 사용
        # 마지막 encoder hidden state를 input_size 차원으로 projection
        last_encoder_hidden = encoder_hidden[-1]  # [B, N, hidden_size]
        initial_decoder_input = self.hidden_to_input_projection(last_encoder_hidden)  # [B, N, input_size]
        
        # 🔥 디코더 collapse 방지: 초기 입력에 작은 노이즈 추가 (평가 시에는 제외)
        if self.training:
            noise_scale = 0.01
            noise = torch.randn_like(initial_decoder_input) * noise_scale
            initial_decoder_input = initial_decoder_input + noise
        
        # Decode
        predictions, attention_weights = self.decode(
            encoder_hidden, encoder_outputs, adjacency, 
            target_length, teacher_forcing_ratio, targets,
            initial_input=initial_decoder_input  # 🔥 Decoder 첫 입력으로 전달
        )
        
        return predictions, attention_weights
    
    def _augment_features(self, 
                         x: torch.Tensor,
                         node_features: Optional[torch.Tensor] = None,
                         date_features: Optional[torch.Tensor] = None,
                         time_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        시계열 데이터에 추가 특성들을 결합
        
        Args:
            x: [batch_size, seq_len, num_nodes, input_size]
            node_features: [batch_size, num_nodes, node_feat_dim]
            date_features: [batch_size, date_feat_dim]
            time_features: [batch_size, time_feat_dim] or [batch_size, seq_len, time_feat_dim]
            
        Returns:
            augmented_x: [batch_size, seq_len, num_nodes, augmented_input_size]
        """
        batch_size, seq_len, num_nodes, input_size = x.shape
        augmented_features = [x]
        
        # 🔥 노드 특성 추가 (각 시점마다 복제)
        if node_features is not None:
            # [B, N, node_dim] → [B, T, N, node_dim]
            node_feat_expanded = node_features.unsqueeze(1).expand(-1, seq_len, -1, -1)
            
            # 🔥 Node features 정규화 (스케일이 너무 커서 다른 입력을 압도하는 문제 해결)
            # Log1p 변환으로 스케일 조정
            node_feat_expanded = torch.log1p(node_feat_expanded + 1e-6)  # 0 값 방지
            
            # 추가 정규화 (평균 0, 표준편차 1 근처로)
            node_mean = node_feat_expanded.mean(dim=(0, 1, 2), keepdim=True)
            node_std = node_feat_expanded.std(dim=(0, 1, 2), keepdim=True) + 1e-6
            node_feat_expanded = (node_feat_expanded - node_mean) / node_std
            
            augmented_features.append(node_feat_expanded)
        
        # 🔥 날짜 특성 추가 (모든 노드, 모든 시점에 복제)
        if date_features is not None:
            # [B, date_dim] → [B, T, N, date_dim]
            date_feat_expanded = date_features.unsqueeze(1).unsqueeze(2).expand(-1, seq_len, num_nodes, -1)
            augmented_features.append(date_feat_expanded)
        
        # 🔥 시간 특성 추가 (시계열로 각 시점마다 다른 시간 특성)
        if time_features is not None:
            if time_features.dim() == 2:
                # 기존 방식: [B, time_dim] → [B, T, N, time_dim] (정적 복제)
                time_feat_expanded = time_features.unsqueeze(1).unsqueeze(2).expand(-1, seq_len, num_nodes, -1)
            elif time_features.dim() == 3:
                # 🔥 새로운 방식: [B, T, time_dim] → [B, T, N, time_dim] (시계열)
                time_feat_expanded = time_features.unsqueeze(2).expand(-1, -1, num_nodes, -1)
            else:
                raise ValueError(f"time_features should be 2D or 3D, got {time_features.dim()}D")
            augmented_features.append(time_feat_expanded)
        
        # 모든 특성 결합
        augmented_x = torch.cat(augmented_features, dim=-1)
        
        return augmented_x
