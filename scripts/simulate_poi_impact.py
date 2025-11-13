#!/usr/bin/env python3
"""
POI 변화에 따른 혼잡도 예측 시뮬레이션

예시: A역에 교육 및 정부기관이 들어온다면 혼잡도가 어떻게 변할까?
- 기존 node features와 변경된 node features로 각각 예측
- 결과 비교 및 분석
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dcrnn_layers import DCRNN
from scripts.train_occupancy_model import OccupancyDataset


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[DCRNN, dict]:
    """모델 로드"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    state_dict = checkpoint['model_state_dict']
    
    # 실제 State Dict에서 차원 확인
    encoder_key = 'encoder_layers.0.reset_gate_x.weight_forward'
    if encoder_key in state_dict:
        actual_input_size = state_dict[encoder_key].shape[1]  # [diffusion_steps, input_size, hidden_size]
        print(f"  실제 State Dict Input Size: {actual_input_size}")
    else:
        actual_input_size = 47  # 기본값 (혼잡도1+승차1+하차1+시간7+날짜21+노드16=47)
        print(f"  ⚠️ State Dict에서 확인 불가, 기본값 사용: {actual_input_size}")
    
    # Output projection에서 실제 output_size 확인
    output_key = 'output_projection.weight'
    if output_key in state_dict:
        actual_output_size = state_dict[output_key].shape[0]  # [output_size, hidden_size]
        print(f"  실제 State Dict Output Size: {actual_output_size}")
    else:
        actual_output_size = 1  # 기본값
        print(f"  ⚠️ State Dict에서 확인 불가, 기본값 사용: {actual_output_size}")
    
    model = DCRNN(
        input_size=actual_input_size,  # 실제 State Dict 차원 사용
        hidden_size=config['model']['hidden_size'],
        output_size=actual_output_size,  # 실제 State Dict 차원 사용
        num_layers=config['model']['num_layers'],
        diffusion_steps=config['model']['diffusion_steps'],
        use_attention=config['model']['use_attention'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # State Dict 로드 (strict=False로 누락된 레이어 허용)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"  ⚠️ 누락된 키 ({len(missing_keys)}개)")
    if unexpected_keys:
        print(f"  ⚠️ 예상치 못한 키 ({len(unexpected_keys)}개)")
    
    model.eval()
    
    print(f"✅ 모델 로드 완료 (Epoch {checkpoint['epoch']})")
    return model, config


def load_node_features(node_features_path: str) -> pd.DataFrame:
    """노드 특성 로드"""
    node_features = pd.read_csv(node_features_path)
    node_features = node_features.set_index('역명')
    return node_features


def modify_node_features(
    node_features: pd.DataFrame,
    station_name: str,
    changes: Dict[str, float]
) -> pd.DataFrame:
    """
    특정 역의 node features 수정
    
    Args:
        node_features: 원본 node features DataFrame
        station_name: 수정할 역 이름
        changes: 변경할 특성과 값 (예: {'교육_밀도': 0.5, '관공서_밀도': 0.4})
    
    Returns:
        수정된 node features DataFrame (복사본)
    """
    modified = node_features.copy()
    
    if station_name not in modified.index:
        raise ValueError(f"역 '{station_name}'을 찾을 수 없습니다.")
    
    print(f"\n📝 {station_name}의 node features 수정:")
    for feature, new_value in changes.items():
        old_value = modified.loc[station_name, feature]
        modified.loc[station_name, feature] = new_value
        print(f"  {feature}: {old_value:.4f} → {new_value:.4f}")
    
    # 총_POI_수와 POI_다양성도 업데이트 필요할 수 있음
    # (간단히 하기 위해 밀도만 변경하고, 총_POI_수는 유지)
    
    return modified


def predict_with_node_features(
    model: DCRNN,
    sample: Dict[str, torch.Tensor],
    modified_node_features: torch.Tensor,
    adjacency: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """수정된 node features로 예측"""
    with torch.no_grad():
        X = sample['X'].unsqueeze(0).to(device)  # [1, T, N, 31]
        
        # 인접행렬 확장
        batch_size = X.size(0)
        adj_batch = adjacency.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 수정된 node features 사용
        node_features_batch = modified_node_features.unsqueeze(0).to(device)  # [1, N, node_dim]
        date_features = sample['date_features'].unsqueeze(0).to(device)
        time_features = sample['time_features'].unsqueeze(0).to(device)
        
        # 예측
        predictions, _ = model(
            X, adj_batch,
            target_length=1,
            node_features=node_features_batch,
            date_features=date_features,
            time_features=time_features
        )
        
        return predictions.squeeze(1)  # [N, 1]


def convert_node_features_to_tensor(
    node_features_df: pd.DataFrame,
    station_to_idx: Dict[str, int],
    feature_names: List[str]
) -> torch.Tensor:
    """
    DataFrame을 Tensor로 변환 (역 순서 맞춤)
    
    Args:
        node_features_df: 역명을 인덱스로 하는 DataFrame
        station_to_idx: 역명 -> 인덱스 매핑
        feature_names: 사용할 특성 이름 리스트
    
    Returns:
        [num_stations, num_features] Tensor
    """
    num_stations = len(station_to_idx)
    num_features = len(feature_names)
    
    # 역 순서대로 정렬
    station_names = [name for name, _ in sorted(station_to_idx.items(), key=lambda x: x[1])]
    
    # 특성 추출
    features_array = np.zeros((num_stations, num_features))
    for idx, station_name in enumerate(station_names):
        if station_name in node_features_df.index:
            for feat_idx, feat_name in enumerate(feature_names):
                if feat_name in node_features_df.columns:
                    features_array[idx, feat_idx] = node_features_df.loc[station_name, feat_name]
    
    return torch.FloatTensor(features_array)


def simulate_poi_impact(
    model: DCRNN,
    test_dataset: OccupancyDataset,
    station_name: str,
    changes: Dict[str, float],
    num_samples: int = 10,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    POI 변화 시뮬레이션 실행
    
    Args:
        model: 학습된 모델
        test_dataset: 테스트 데이터셋
        station_name: 시뮬레이션할 역 이름
        changes: 변경할 특성과 값
        num_samples: 테스트할 샘플 수
        device: 디바이스
    
    Returns:
        시뮬레이션 결과 딕셔너리
    """
    # 원본 node features 로드
    node_features_df = load_node_features('data/node_features.csv')
    
    # 역 인덱스 매핑
    station_to_idx = test_dataset.station_to_idx
    
    if station_name not in station_to_idx:
        raise ValueError(f"역 '{station_name}'을 찾을 수 없습니다. 사용 가능한 역: {list(station_to_idx.keys())}")
    
    # 원본 node features를 Tensor로 변환
    # 실제 사용되는 특성 이름 확인 (node_features_info.json 참고)
    feature_names = [
        '관공서_밀도', '교육_밀도', '교통_밀도', '금융_밀도', '상업_밀도',
        '숙박_밀도', '의료_밀도', '종교_밀도', '주거_밀도', '체육문화_밀도',
        '총_POI_수', 'POI_다양성', '위도', '경도', '노선순서', '연면적'
    ]
    
    original_node_features = convert_node_features_to_tensor(
        node_features_df, station_to_idx, feature_names
    )
    
    # 수정된 node features 생성
    modified_node_features_df = modify_node_features(node_features_df, station_name, changes)
    modified_node_features = convert_node_features_to_tensor(
        modified_node_features_df, station_to_idx, feature_names
    )
    
    # 인접행렬
    adjacency = test_dataset.adjacency_matrix
    
    # 정규화 통계 (예측값 역변환용)
    log_mean = test_dataset.log_mean
    log_std = test_dataset.log_std
    
    def inverse_transform(pred_norm):
        """정규화된 예측값을 원본 스케일로 역변환"""
        if isinstance(pred_norm, torch.Tensor):
            pred_norm = pred_norm.cpu().numpy()
        pred_log = pred_norm * log_std + log_mean
        pred_original = np.expm1(pred_log)
        return pred_original
    
    # 결과 저장
    results = {
        'station_name': station_name,
        'changes': changes,
        'samples': []
    }
    
    station_idx = station_to_idx[station_name]
    
    print(f"\n🔮 {num_samples}개 샘플로 시뮬레이션 시작...")
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            sample = test_dataset[i]
            
            # 원본 node features로 예측
            pred_original = predict_with_node_features(
                model, sample, original_node_features, adjacency, device
            )
            
            # 수정된 node features로 예측
            pred_modified = predict_with_node_features(
                model, sample, modified_node_features, adjacency, device
            )
            
            # 예측 결과 shape 확인 및 처리
            if pred_original.dim() == 2:
                # [N, 1] 형태
                pred_orig_val = inverse_transform(pred_original[station_idx, 0].cpu().numpy())
                pred_mod_val = inverse_transform(pred_modified[station_idx, 0].cpu().numpy())
            elif pred_original.dim() == 1:
                # [N] 형태
                pred_orig_val = inverse_transform(pred_original[station_idx].cpu().numpy())
                pred_mod_val = inverse_transform(pred_modified[station_idx].cpu().numpy())
            else:
                # [1, N, 1] 형태
                pred_original = pred_original.squeeze(0)  # [N, 1]
                pred_modified = pred_modified.squeeze(0)  # [N, 1]
                pred_orig_val = inverse_transform(pred_original[station_idx, 0].cpu().numpy())
                pred_mod_val = inverse_transform(pred_modified[station_idx, 0].cpu().numpy())
            
            # 실제 값 (있는 경우)
            if 'Y_raw' in sample:
                actual_val = sample['Y_raw'][station_idx].item()
            else:
                actual_val = None
            
            # 변화량 계산
            change = pred_mod_val - pred_orig_val
            change_pct = (change / pred_orig_val * 100) if pred_orig_val > 0 else 0
            
            results['samples'].append({
                'sample_idx': i,
                'original_pred': float(pred_orig_val),
                'modified_pred': float(pred_mod_val),
                'change': float(change),
                'change_pct': float(change_pct),
                'actual': float(actual_val) if actual_val is not None else None
            })
            
            if (i + 1) % 5 == 0:
                print(f"  진행: {i+1}/{num_samples}")
    
    # 통계 계산
    changes_list = [s['change'] for s in results['samples']]
    changes_pct_list = [s['change_pct'] for s in results['samples']]
    
    results['statistics'] = {
        'mean_change': float(np.mean(changes_list)),
        'std_change': float(np.std(changes_list)),
        'min_change': float(np.min(changes_list)),
        'max_change': float(np.max(changes_list)),
        'mean_change_pct': float(np.mean(changes_pct_list)),
        'std_change_pct': float(np.std(changes_pct_list))
    }
    
    return results


def print_results(results: Dict):
    """결과 출력"""
    print("\n" + "="*70)
    print(f"📊 시뮬레이션 결과: {results['station_name']}")
    print("="*70)
    
    print(f"\n🔧 변경 사항:")
    for feature, value in results['changes'].items():
        print(f"  - {feature}: {value:.4f}")
    
    print(f"\n📈 통계 요약 ({len(results['samples'])}개 샘플):")
    stats = results['statistics']
    print(f"  평균 변화량: {stats['mean_change']:+.1f}명 ({stats['mean_change_pct']:+.1f}%)")
    print(f"  표준편차: {stats['std_change']:.1f}명 ({stats['std_change_pct']:.1f}%)")
    print(f"  최소 변화: {stats['min_change']:+.1f}명")
    print(f"  최대 변화: {stats['max_change']:+.1f}명")
    
    print(f"\n📋 샘플별 상세 결과 (처음 5개):")
    print(f"{'샘플':<8} {'기존 예측':<12} {'변경 후 예측':<14} {'변화량':<12} {'변화율':<10}")
    print("-" * 70)
    for sample in results['samples'][:5]:
        print(f"{sample['sample_idx']:<8} "
              f"{sample['original_pred']:>10.1f}명  "
              f"{sample['modified_pred']:>12.1f}명  "
              f"{sample['change']:>+10.1f}명  "
              f"{sample['change_pct']:>+8.1f}%")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="POI 변화에 따른 혼잡도 예측 시뮬레이션")
    parser.add_argument('--station', type=str, default='정부청사역',
                       help='시뮬레이션할 역 이름')
    parser.add_argument('--education', type=float, default=0.5,
                       help='교육_밀도 변경값 (0.0~1.0)')
    parser.add_argument('--government', type=float, default=0.6,
                       help='관공서_밀도 변경값 (0.0~1.0)')
    parser.add_argument('--samples', type=int, default=20,
                       help='테스트할 샘플 수')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/occupancy_model/best_occupancy_model.pth',
                       help='모델 체크포인트 경로')
    
    args = parser.parse_args()
    
    print("🚇 POI 변화에 따른 혼잡도 예측 시뮬레이션")
    print("="*70)
    
    # 디바이스 설정
    device = torch.device('cpu')
    
    # 모델 로드
    print(f"\n📦 모델 로드 중: {args.checkpoint}")
    if not Path(args.checkpoint).exists():
        print(f"❌ 체크포인트 파일이 없습니다: {args.checkpoint}")
        return
    
    model, config = load_model(args.checkpoint, device)
    
    # 테스트 데이터셋 생성
    print("\n📂 테스트 데이터셋 생성 중...")
    congestion_data = pd.read_csv('data/혼잡도.csv')
    node_features = pd.read_csv('data/node_features.csv')
    date_features = pd.read_csv('data/date_features.csv')
    time_features = pd.read_csv('data/time_features.csv')
    adjacency_matrix = np.load('data/adjacency_matrix.npy')
    
    # 역 인덱스 매핑
    stations = sorted(congestion_data['station'].unique())
    station_to_idx = {station: idx for idx, station in enumerate(stations)}
    
    # 테스트 데이터 분할 (85% 이후)
    dates = sorted(congestion_data['date'].unique())
    n_dates = len(dates)
    test_start = int(n_dates * 0.85)
    test_dates = dates[test_start:]
    test_data = congestion_data[congestion_data['date'].isin(test_dates)]
    
    test_dataset = OccupancyDataset(
        test_data, node_features, date_features, time_features,
        adjacency_matrix, station_to_idx,
        sequence_length=12, prediction_length=1,
        target_columns=['occupancy']
    )
    
    print(f"✅ 테스트 데이터셋 준비 완료 (샘플 수: {len(test_dataset)})")
    
    # 시뮬레이션 실행
    changes = {
        '교육_밀도': args.education,
        '관공서_밀도': args.government
    }
    
    results = simulate_poi_impact(
        model, test_dataset, args.station, changes,
        num_samples=args.samples, device=device
    )
    
    # 결과 출력
    print_results(results)
    
    # 결과 저장
    output_path = f"results/poi_simulation_{args.station}.json"
    os.makedirs("results", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 결과 저장: {output_path}")


if __name__ == '__main__':
    main()

