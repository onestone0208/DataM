#!/usr/bin/env python3
"""
모델 로딩 문제 해결 스크립트

체크포인트의 실제 구조를 확인하고 올바르게 로드하는 방법 제시
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN


def analyze_checkpoint(checkpoint_path: str):
    """체크포인트 구조 분석"""
    print("=" * 60)
    print("🔍 체크포인트 구조 분석")
    print("=" * 60)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Config 정보
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    print(f"\n📋 체크포인트 Config:")
    print(f"  input_size: {model_config.get('input_size', 'N/A')}")
    print(f"  output_size: {model_config.get('output_size', 'N/A')}")
    print(f"  hidden_size: {model_config.get('hidden_size', 'N/A')}")
    print(f"  num_layers: {model_config.get('num_layers', 'N/A')}")
    print(f"  epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # State dict 분석
    state_dict = checkpoint['model_state_dict']
    
    print(f"\n🔧 State Dict 분석:")
    print(f"  총 파라미터 수: {len(state_dict)}")
    
    # Encoder 첫 레이어의 입력 차원 확인
    encoder_first_layer = 'encoder_layers.0.reset_gate_x.weight_forward'
    if encoder_first_layer in state_dict:
        weight_shape = state_dict[encoder_first_layer].shape
        print(f"  Encoder 첫 레이어 입력 차원: {weight_shape[1]} (확인됨)")
        actual_input_size = weight_shape[1]
    else:
        print(f"  ⚠️ Encoder 첫 레이어를 찾을 수 없음")
        actual_input_size = None
    
    # Output projection 차원 확인
    output_proj = 'output_projection.weight'
    if output_proj in state_dict:
        output_shape = state_dict[output_proj].shape
        print(f"  Output projection 출력 차원: {output_shape[0]} (확인됨)")
        actual_output_size = output_shape[0]
    else:
        print(f"  ⚠️ Output projection을 찾을 수 없음")
        actual_output_size = None
    
    return {
        'config_input_size': model_config.get('input_size'),
        'config_output_size': model_config.get('output_size'),
        'actual_input_size': actual_input_size,
        'actual_output_size': actual_output_size,
        'model_config': model_config
    }


def test_model_loading(checkpoint_path: str, use_config: bool = True):
    """모델 로딩 테스트"""
    print("\n" + "=" * 60)
    print("🧪 모델 로딩 테스트")
    print("=" * 60)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config = checkpoint['config']['model']
    
    if use_config:
        # 방법 1: Config에 있는 값 사용 (옛날 구조)
        print("\n📌 방법 1: Config 값 사용 (옛날 구조)")
        print(f"  input_size={model_config['input_size']}, output_size={model_config['output_size']}")
        
        try:
            model = DCRNN(
                input_size=model_config['input_size'],
                hidden_size=model_config['hidden_size'],
                output_size=model_config['output_size'],
                num_layers=model_config['num_layers'],
                diffusion_steps=model_config['diffusion_steps'],
                use_attention=model_config['use_attention'],
                dropout=model_config['dropout']
            )
            
            # State dict 로드
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            print(f"  ✅ 로드 성공!")
            print(f"  Missing keys: {len(missing_keys)}")
            print(f"  Unexpected keys: {len(unexpected_keys)}")
            
            if missing_keys:
                print(f"  ⚠️ 누락된 키 (처음 5개): {missing_keys[:5]}")
            if unexpected_keys:
                print(f"  ⚠️ 예상치 못한 키 (처음 5개): {unexpected_keys[:5]}")
                
        except Exception as e:
            print(f"  ❌ 로드 실패: {e}")
    
    else:
        # 방법 2: 실제 차원 사용 (새 구조)
        print("\n📌 방법 2: 실제 차원 사용 (새 구조)")
        print(f"  input_size=47, output_size=1")
        
        try:
            model = DCRNN(
                input_size=47,
                hidden_size=model_config['hidden_size'],
                output_size=1,
                num_layers=model_config['num_layers'],
                diffusion_steps=model_config['diffusion_steps'],
                use_attention=model_config['use_attention'],
                dropout=model_config['dropout']
            )
            
            # State dict 로드
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            print(f"  ✅ 로드 성공 (일부만 로드됨)")
            print(f"  Missing keys: {len(missing_keys)}")
            print(f"  Unexpected keys: {len(unexpected_keys)}")
            
            if missing_keys:
                print(f"  ⚠️ 누락된 키 (처음 10개): {missing_keys[:10]}")
            if unexpected_keys:
                print(f"  ⚠️ 예상치 못한 키 (처음 10개): {unexpected_keys[:10]}")
                
        except Exception as e:
            print(f"  ❌ 로드 실패: {e}")


def main():
    checkpoint_path = "checkpoints/occupancy_model/best_occupancy_model.pth"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        return
    
    # 체크포인트 분석
    analysis = analyze_checkpoint(checkpoint_path)
    
    # 모델 로딩 테스트
    test_model_loading(checkpoint_path, use_config=True)  # 옛날 구조
    test_model_loading(checkpoint_path, use_config=False)  # 새 구조
    
    # 결론
    print("\n" + "=" * 60)
    print("💡 결론 및 권장사항")
    print("=" * 60)
    
    if analysis['config_input_size'] != 47 or analysis['config_output_size'] != 1:
        print("\n⚠️ 문제 확인:")
        print(f"  - 체크포인트는 옛날 구조로 학습됨 (input={analysis['config_input_size']}, output={analysis['config_output_size']})")
        print(f"  - 현재 코드는 새 구조를 사용 (input=47, output=1)")
        print(f"  - 이 불일치가 모델 collapse의 원인입니다!")
        
        print("\n✅ 해결 방법:")
        print("  1. 새 구조로 모델을 재학습하세요")
        print("  2. 또는 평가 시 옛날 구조를 사용하세요 (권장하지 않음)")
        print("  3. 체크포인트 디렉토리를 정리하고 새로 학습하세요")
    else:
        print("\n✅ 체크포인트 구조가 현재 코드와 일치합니다!")


if __name__ == '__main__':
    main()

