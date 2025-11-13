#!/usr/bin/env python3
"""
DCRNN 모델 기본 테스트 스크립트

모델 구조와 forward pass가 정상 작동하는지 확인
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN, DiffusionConvolution, DCGRUCell, AttentionMechanism
from models.losses import MultiTaskLoss


def test_diffusion_convolution():
    """Diffusion Convolution 레이어 테스트"""
    print("Testing DiffusionConvolution...")
    
    batch_size, num_nodes, in_channels, out_channels = 4, 22, 2, 64
    diffusion_steps = 3
    
    # 레이어 생성
    layer = DiffusionConvolution(in_channels, out_channels, diffusion_steps)
    
    # 입력 데이터
    x = torch.randn(batch_size, num_nodes, in_channels)
    adjacency = torch.rand(batch_size, num_nodes, num_nodes)
    
    # Forward pass
    output = layer(x, adjacency)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected shape: ({batch_size}, {num_nodes}, {out_channels})")
    
    assert output.shape == (batch_size, num_nodes, out_channels)
    print("  ✅ DiffusionConvolution test passed!")


def test_dcgru_cell():
    """DCGRU Cell 테스트"""
    print("\nTesting DCGRUCell...")
    
    batch_size, num_nodes, input_size, hidden_size = 4, 22, 2, 64
    
    # 셀 생성
    cell = DCGRUCell(input_size, hidden_size)
    
    # 입력 데이터
    x = torch.randn(batch_size, num_nodes, input_size)
    hidden = torch.randn(batch_size, num_nodes, hidden_size)
    adjacency = torch.rand(batch_size, num_nodes, num_nodes)
    
    # Forward pass
    new_hidden = cell(x, hidden, adjacency)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Hidden shape: {hidden.shape}")
    print(f"  Output shape: {new_hidden.shape}")
    
    assert new_hidden.shape == hidden.shape
    print("  ✅ DCGRUCell test passed!")


def test_attention_mechanism():
    """Attention Mechanism 테스트"""
    print("\nTesting AttentionMechanism...")
    
    batch_size, seq_len, num_nodes, hidden_size = 4, 12, 22, 64
    
    # Attention 생성
    attention = AttentionMechanism(hidden_size)
    
    # 입력 데이터
    hidden_states = torch.randn(batch_size, seq_len, num_nodes, hidden_size)
    query = torch.randn(batch_size, num_nodes, hidden_size)
    
    # Forward pass
    context, attention_weights = attention(hidden_states, query)
    
    print(f"  Hidden states shape: {hidden_states.shape}")
    print(f"  Query shape: {query.shape}")
    print(f"  Context shape: {context.shape}")
    print(f"  Attention weights shape: {attention_weights.shape}")
    
    assert context.shape == (batch_size, num_nodes, hidden_size)
    assert attention_weights.shape == (batch_size, seq_len)
    print("  ✅ AttentionMechanism test passed!")


def test_dcrnn_model():
    """전체 DCRNN 모델 테스트"""
    print("\nTesting DCRNN Model...")
    
    # 모델 파라미터
    batch_size, seq_len, num_nodes = 4, 12, 22
    input_size, hidden_size, output_size = 2, 64, 2
    num_layers, diffusion_steps = 2, 3
    target_length = 1
    
    # 모델 생성
    model = DCRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        diffusion_steps=diffusion_steps,
        use_attention=True,
        dropout=0.1
    )
    
    # 입력 데이터
    x = torch.randn(batch_size, seq_len, num_nodes, input_size)
    adjacency = torch.rand(batch_size, num_nodes, num_nodes)
    
    # Forward pass (추론 모드)
    model.eval()
    with torch.no_grad():
        predictions, attention_weights = model(x, adjacency, target_length=target_length)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Adjacency shape: {adjacency.shape}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Attention weights shape: {attention_weights.shape if attention_weights is not None else None}")
    
    expected_pred_shape = (batch_size, target_length, num_nodes, output_size)
    assert predictions.shape == expected_pred_shape
    
    if attention_weights is not None:
        expected_att_shape = (batch_size, target_length, seq_len)
        assert attention_weights.shape == expected_att_shape
    
    print("  ✅ DCRNN Model test passed!")
    
    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")


def test_loss_function():
    """손실 함수 테스트"""
    print("\nTesting MultiTaskLoss...")
    
    batch_size, num_nodes, output_size = 4, 22, 2
    
    # 손실 함수 생성
    loss_fn = MultiTaskLoss(
        occupancy_weight=1.0,
        flow_weight=0.5,
        loss_type='mae',
        adaptive_weights=True
    )
    
    # 데이터 생성
    predictions = torch.randn(batch_size, num_nodes, output_size)
    targets = torch.randn(batch_size, num_nodes, output_size)
    
    # 손실 계산
    loss_dict = loss_fn(predictions, targets)
    
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"  Occupancy loss: {loss_dict['occupancy_loss'].item():.4f}")
    print(f"  Flow loss: {loss_dict['flow_loss'].item():.4f}")
    
    assert 'total_loss' in loss_dict
    assert 'occupancy_loss' in loss_dict
    assert 'flow_loss' in loss_dict
    
    print("  ✅ MultiTaskLoss test passed!")


def test_training_step():
    """학습 스텝 테스트"""
    print("\nTesting Training Step...")
    
    # 모델 및 손실 함수
    model = DCRNN(
        input_size=2, hidden_size=32, output_size=2,
        num_layers=1, diffusion_steps=2, use_attention=True
    )
    
    loss_fn = MultiTaskLoss(loss_type='mae')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 데이터
    batch_size, seq_len, num_nodes = 2, 12, 22
    x = torch.randn(batch_size, seq_len, num_nodes, 2)
    y = torch.randn(batch_size, num_nodes, 2)
    adjacency = torch.rand(batch_size, num_nodes, num_nodes)
    
    # 학습 모드
    model.train()
    
    # Forward pass
    predictions, _ = model(x, adjacency, target_length=1)
    predictions = predictions.squeeze(1)  # [B, N, 2]
    
    # 손실 계산
    loss_dict = loss_fn(predictions, y)
    loss = loss_dict['total_loss']
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"  Training loss: {loss.item():.4f}")
    print("  ✅ Training step test passed!")


def main():
    """모든 테스트 실행"""
    print("🚀 DCRNN Model Testing")
    print("=" * 50)
    
    try:
        # 개별 컴포넌트 테스트
        test_diffusion_convolution()
        test_dcgru_cell()
        test_attention_mechanism()
        
        # 전체 모델 테스트
        test_dcrnn_model()
        
        # 손실 함수 테스트
        test_loss_function()
        
        # 학습 스텝 테스트
        test_training_step()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        print("✅ DCRNN model is ready for training!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
