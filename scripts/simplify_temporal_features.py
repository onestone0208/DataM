#!/usr/bin/env python3
"""
시간 특징을 더 단순하게 만들기 - 순환 인코딩 제거
"""

import pandas as pd
from pathlib import Path
import argparse


def simplify_temporal_features(input_file: Path, output_file: Path) -> None:
    """복잡한 순환 인코딩 제거하고 단순한 특징만 유지"""
    
    # 데이터 로드
    df = pd.read_csv(input_file)
    
    # 제거할 컬럼들 (순환 인코딩)
    cols_to_remove = ['월_sin', '월_cos', '일_sin', '일_cos']
    
    # 컬럼 제거
    df_simplified = df.drop(columns=cols_to_remove, errors='ignore')
    
    # 날짜에서 월, 일 추출 (단순한 숫자로)
    df_simplified['date_parsed'] = pd.to_datetime(df_simplified['date'])
    df_simplified['월'] = df_simplified['date_parsed'].dt.month
    df_simplified['일'] = df_simplified['date_parsed'].dt.day
    
    # 임시 컬럼 제거
    df_simplified = df_simplified.drop('date_parsed', axis=1)
    
    # 저장
    df_simplified.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"=== 시간 특징 단순화 완료 ===")
    print(f"원본 특징 수: {len(df.columns)}개")
    print(f"단순화 후 특징 수: {len(df_simplified.columns)}개")
    print(f"제거된 특징: {cols_to_remove}")
    print(f"추가된 특징: ['월', '일']")
    
    # 최종 특징 구성
    feature_categories = {
        '요일': [col for col in df_simplified.columns if col.startswith('요일_')],
        '기온': ['avg_temp_c', 'min_temp_c', 'max_temp_c'],
        '날짜': ['월', '일'],
        '휴일': ['주말여부', '휴일여부', '평일여부', 'is_holiday'],
        '계절': ['봄', '여름', '가을', '겨울'],
        '기타': ['date', 'weekday', 'weekday_name', 'holiday_name']
    }
    
    print(f"\n=== 최종 특징 구성 ===")
    total_features = 0
    for category, features in feature_categories.items():
        existing_features = [f for f in features if f in df_simplified.columns]
        print(f"{category:10s}: {len(existing_features)}개 - {existing_features}")
        total_features += len(existing_features)
    
    print(f"총 특징 수: {total_features}개")
    
    # 샘플 출력
    print(f"\n=== 샘플 (2025-01-01) ===")
    sample = df_simplified[df_simplified['date'] == '2025-01-01'].iloc[0]
    key_features = ['date', '월', '일', 'avg_temp_c', '요일_수', '휴일여부', '겨울']
    for col in key_features:
        if col in sample.index:
            print(f"{col:15s}: {sample[col]}")


def main():
    parser = argparse.ArgumentParser(description="시간 특징 단순화")
    parser.add_argument(
        "--input", 
        type=Path, 
        default=Path("data/temporal_features.csv"),
        help="입력 시간 특징 CSV 파일"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=Path("data/temporal_features_simple.csv"),
        help="출력 단순화된 CSV 파일"
    )
    
    args = parser.parse_args()
    
    simplify_temporal_features(args.input, args.output)
    print(f"\n단순화된 시간 특징 저장: {args.output}")


if __name__ == "__main__":
    main()
