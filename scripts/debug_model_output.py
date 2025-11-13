#!/usr/bin/env python3
"""
모델 출력 디버깅 스크립트

왜 모델이 고정값을 출력하는지 확인
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN


def test_model_output():
    """모델 출력 테스트"""
    print("🔍 모델 출력 디버깅")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # 모델 생성 (작은 모델)
    model = DCRNN(
        input_size=47,
        hidden_size=64,
        output_size=1,
        num_layers=2,
        diffusion_steps=3,
        use_attention=True,
        dropout=0.0
    ).to(device)
    
    model.eval()
    
    # 테스트 데이터 생성
    batch_size, seq_len, num_nodes = 2, 12, 22
    
    # 다양한 입력 데이터 생성
    X1 = torch.randn(batch_size, seq_len, num_nodes, 47) * 0.5
    X2 = torch.randn(batch_size, seq_len, num_nodes, 47) * 2.0
    X3 = torch.zeros(batch_size, seq_len, num_nodes, 47)
    
    adjacency = torch.eye(num_nodes).unsqueeze(0).expand(batch_size, -1, -1)
    
    print("\n1. 다양한 입력에 대한 출력 테스트:")
    print("-" * 60)
    
    with torch.no_grad():
        # 입력 1
        pred1, _ = model(X1, adjacency, target_length=1, teacher_forcing_ratio=0.0)
        pred1 = pred1.squeeze(1)  # [B, N, 1]
        print(f"입력 1 (작은 값):")
        print(f"  예측 범위: [{pred1.min().item():.3f}, {pred1.max().item():.3f}]")
        print(f"  역별 예측 (첫 배치): {pred1[0, :5, 0].tolist()}")
        
        # 입력 2
        pred2, _ = model(X2, adjacency, target_length=1, teacher_forcing_ratio=0.0)
        pred2 = pred2.squeeze(1)
        print(f"\n입력 2 (큰 값):")
        print(f"  예측 범위: [{pred2.min().item():.3f}, {pred2.max().item():.3f}]")
        print(f"  역별 예측 (첫 배치): {pred2[0, :5, 0].tolist()}")
        
        # 입력 3
        pred3, _ = model(X3, adjacency, target_length=1, teacher_forcing_ratio=0.0)
        pred3 = pred3.squeeze(1)
        print(f"\n입력 3 (영벡터):")
        print(f"  예측 범위: [{pred3.min().item():.3f}, {pred3.max().item():.3f}]")
        print(f"  역별 예측 (첫 배치): {pred3[0, :5, 0].tolist()}")
    
    print("\n2. 같은 입력에 대한 반복 출력 테스트:")
    print("-" * 60)
    
    with torch.no_grad():
        outputs = []
        for i in range(5):
            pred, _ = model(X1, adjacency, target_length=1, teacher_forcing_ratio=0.0)
            pred = pred.squeeze(1)
            outputs.append(pred[0, 0, 0].item())  # 첫 배치, 첫 역
        
        print(f"같은 입력 5번 예측: {outputs}")
        print(f"표준편차: {np.std(outputs):.6f}")
        
        if np.std(outputs) < 1e-6:
            print("⚠️ 완전히 고정된 출력! (결정론적 모델이므로 정상)")
        else:
            print("✅ 출력이 변함 (비정상 - 랜덤성이 있음)")
    
    print("\n3. Encoder 출력 확인:")
    print("-" * 60)
    
    with torch.no_grad():
        # Encoder만 실행
        encoder_hidden, encoder_outputs = model.encode(X1, adjacency)
        
        print(f"Encoder hidden states:")
        for i, h in enumerate(encoder_hidden):
            print(f"  Layer {i}: mean={h.mean().item():.4f}, std={h.std().item():.4f}, "
                  f"range=[{h.min().item():.4f}, {h.max().item():.4f}]")
        
        print(f"\nEncoder outputs:")
        print(f"  Shape: {encoder_outputs.shape}")
        print(f"  Mean: {encoder_outputs.mean().item():.4f}, Std: {encoder_outputs.std().item():.4f}")
        print(f"  Range: [{encoder_outputs.min().item():.4f}, {encoder_outputs.max().item():.4f}]")
    
    print("\n4. 디코더 초기 입력 확인:")
    print("-" * 60)
    
    with torch.no_grad():
        encoder_hidden, encoder_outputs = model.encode(X1, adjacency)
        last_encoder_hidden = encoder_hidden[-1]
        initial_decoder_input = model.hidden_to_input_projection(last_encoder_hidden)
        
        print(f"디코더 초기 입력:")
        print(f"  Shape: {initial_decoder_input.shape}")
        print(f"  Mean: {initial_decoder_input.mean().item():.4f}, Std: {initial_decoder_input.std().item():.4f}")
        print(f"  Range: [{initial_decoder_input.min().item():.4f}, {initial_decoder_input.max().item():.4f}]")
        print(f"  역별 값 (첫 배치): {initial_decoder_input[0, :5, 0].tolist()}")
        
        # 다른 입력으로 테스트
        encoder_hidden2, _ = model.encode(X2, adjacency)
        last_encoder_hidden2 = encoder_hidden2[-1]
        initial_decoder_input2 = model.hidden_to_input_projection(last_encoder_hidden2)
        
        print(f"\n다른 입력의 디코더 초기 입력:")
        print(f"  역별 값 (첫 배치): {initial_decoder_input2[0, :5, 0].tolist()}")
        
        # 차이 확인
        diff = (initial_decoder_input - initial_decoder_input2).abs().mean()
        print(f"\n두 입력의 초기 디코더 입력 차이: {diff.item():.4f}")
        
        if diff.item() < 1e-6:
            print("⚠️ 두 입력이 같은 초기 디코더 입력을 생성! (문제!)")
        else:
            print("✅ 입력에 따라 다른 초기 디코더 입력 생성 (정상)")


if __name__ == '__main__':
    test_model_output()

