#!/usr/bin/env python3
"""
노드 특징에서 불필요한 컬럼들을 제거하고 정리하는 스크립트
"""

import pandas as pd
from pathlib import Path
import argparse


def clean_node_features(input_file: Path, output_file: Path) -> None:
    """노드 특징에서 불필요한 컬럼 제거"""
    
    # 데이터 로드
    df = pd.read_csv(input_file, index_col=0)  # 역명을 인덱스로
    
    # 유지할 컬럼 정의
    keep_columns = [
        # POI 밀도 특징 (10개)
        '관공서_밀도', '교육_밀도', '교통_밀도', '금융_밀도', '상업_밀도',
        '숙박_밀도', '의료_밀도', '종교_밀도', '주거_밀도', '체육문화_밀도',
        
        # POI 통계 (2개)
        '총_POI_수', 'POI_다양성',
        
        # 물리적 특징 (4개) - 원본 값만 유지
        '위도', '경도', '노선순서', '연면적'
    ]
    
    # 필요한 컬럼만 선택
    cleaned_df = df[keep_columns].copy()
    
    # 저장
    cleaned_df.to_csv(output_file, encoding='utf-8-sig')
    
    print(f"=== 노드 특징 정리 완료 ===")
    print(f"원본 특징 수: {len(df.columns)}개")
    print(f"정리 후 특징 수: {len(cleaned_df.columns)}개")
    print(f"제거된 특징 수: {len(df.columns) - len(cleaned_df.columns)}개")
    
    print(f"\n=== 제거된 특징들 ===")
    removed_cols = set(df.columns) - set(keep_columns)
    for col in sorted(removed_cols):
        print(f"- {col}")
    
    print(f"\n=== 최종 특징 구성 ===")
    poi_cols = [col for col in keep_columns if '_밀도' in col]
    stat_cols = ['총_POI_수', 'POI_다양성']
    physical_cols = ['위도', '경도', '노선순서', '연면적']
    
    print(f"POI 밀도 특징: {len(poi_cols)}개")
    print(f"POI 통계 특징: {len(stat_cols)}개")
    print(f"물리적 특징: {len(physical_cols)}개")
    print(f"총 특징 수: {len(keep_columns)}개")
    
    # 샘플 출력
    print(f"\n=== 샘플 (갈마역) ===")
    if '갈마역' in cleaned_df.index:
        sample = cleaned_df.loc['갈마역']
        for col in cleaned_df.columns:
            print(f"{col:15s}: {sample[col]:.3f}")


def main():
    parser = argparse.ArgumentParser(description="노드 특징 정리")
    parser.add_argument(
        "--input", 
        type=Path, 
        default=Path("data/node_features.csv"),
        help="입력 노드 특징 CSV 파일"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=Path("data/node_features_clean.csv"),
        help="출력 정리된 CSV 파일"
    )
    
    args = parser.parse_args()
    
    clean_node_features(args.input, args.output)
    print(f"\n정리된 노드 특징 저장: {args.output}")


if __name__ == "__main__":
    main()
