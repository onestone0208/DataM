#!/usr/bin/env python3
"""
Dataset 디버깅 스크립트
"""

import sys
import pickle
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dataset import SubwayGraphDataset

def main():
    print("Dataset 디버깅 시작...")
    
    # 데이터 로드
    with open('data/train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    
    print(f"Dataset 타입: {type(train_dataset)}")
    print(f"Dataset 길이: {len(train_dataset)}")
    
    # 첫 번째 샘플 확인
    try:
        sample = train_dataset[0]
        print(f"샘플 키: {sample.keys()}")
        
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: {value}")
                
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
