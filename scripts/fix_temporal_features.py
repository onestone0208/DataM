#!/usr/bin/env python3
"""
시간 특징에서 기온 정규화를 제거하고 원본 기온 값을 사용하도록 수정
"""

import pandas as pd
from pathlib import Path
import argparse


def remove_temperature_normalization(input_file: Path, output_file: Path) -> None:
    """기온 정규화 컬럼 제거"""
    
    # 데이터 로드
    df = pd.read_csv(input_file)
    
    # 제거할 컬럼들
    cols_to_remove = ['avg_temp_c_정규화', 'min_temp_c_정규화', 'max_temp_c_정규화']
    
    # 컬럼 제거
    df_cleaned = df.drop(columns=cols_to_remove, errors='ignore')
    
    # 저장
    df_cleaned.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"=== 기온 정규화 제거 완료 ===")
    print(f"원본 특징 수: {len(df.columns)}개")
    print(f"수정 후 특징 수: {len(df_cleaned.columns)}개")
    print(f"제거된 특징: {cols_to_remove}")
    
    # 기온 범위 확인
    temp_cols = ['avg_temp_c', 'min_temp_c', 'max_temp_c']
    print(f"\n=== 기온 데이터 범위 ===")
    for col in temp_cols:
        if col in df_cleaned.columns:
            min_val, max_val = df_cleaned[col].min(), df_cleaned[col].max()
            print(f"{col:15s}: {min_val:6.1f}°C ~ {max_val:6.1f}°C")
    
    # 샘플 출력
    print(f"\n=== 샘플 (2025-01-01) ===")
    sample = df_cleaned[df_cleaned['date'] == '2025-01-01'].iloc[0]
    for col in ['date', 'avg_temp_c', 'min_temp_c', 'max_temp_c', '요일_수', '휴일여부', '겨울']:
        if col in sample.index:
            print(f"{col:15s}: {sample[col]}")


def main():
    parser = argparse.ArgumentParser(description="기온 정규화 제거")
    parser.add_argument(
        "--input", 
        type=Path, 
        default=Path("data/temporal_features.csv"),
        help="입력 시간 특징 CSV 파일"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=Path("data/temporal_features_fixed.csv"),
        help="출력 수정된 CSV 파일"
    )
    
    args = parser.parse_args()
    
    remove_temperature_normalization(args.input, args.output)
    print(f"\n수정된 시간 특징 저장: {args.output}")


if __name__ == "__main__":
    main()
