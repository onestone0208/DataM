#!/usr/bin/env python3
"""
시간/외생 특징 확장 스크립트

날짜정보.csv를 확장해서 모델 입력용 시간 특징을 생성합니다:
- 요일 one-hot 인코딩
- 공휴일 플래그
- 기온 정규화
- sin/cos 시간대 인코딩 (24시간)
- 월/일 순환 인코딩
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse
import json
from datetime import datetime


def add_weekday_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """요일 one-hot 인코딩 추가"""
    weekday_names = ['월', '화', '수', '목', '금', '토', '일']
    
    # 요일 one-hot (0=월요일, 6=일요일)
    for i, day_name in enumerate(weekday_names):
        df[f'요일_{day_name}'] = (df['weekday'] == i).astype(int)
    
    return df


def add_time_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """시간 순환 특징 추가 (월, 일)"""
    # 날짜 파싱
    df['date_parsed'] = pd.to_datetime(df['date'])
    
    # 월 순환 인코딩 (1-12월)
    month = df['date_parsed'].dt.month
    df['월_sin'] = np.sin(2 * np.pi * month / 12)
    df['월_cos'] = np.cos(2 * np.pi * month / 12)
    
    # 일 순환 인코딩 (1-31일, 월별로 다름)
    day = df['date_parsed'].dt.day
    days_in_month = df['date_parsed'].dt.days_in_month
    df['일_sin'] = np.sin(2 * np.pi * day / days_in_month)
    df['일_cos'] = np.cos(2 * np.pi * day / days_in_month)
    
    # 임시 컬럼 제거
    df = df.drop('date_parsed', axis=1)
    
    return df


def add_hour_cyclical_features() -> pd.DataFrame:
    """24시간 순환 특징 생성 (별도 테이블)"""
    hours = list(range(24))
    
    hour_features = pd.DataFrame({
        'hour': hours,
        'time_slot': [f'{h:02d}-{(h+1)%24:02d}시' for h in hours],
        '시간_sin': np.sin(2 * np.pi * np.array(hours) / 24),
        '시간_cos': np.cos(2 * np.pi * np.array(hours) / 24)
    })
    
    # 시간대별 특성 추가
    hour_features['평일_출근시간'] = hour_features['hour'].isin([7, 8, 9]).astype(int)
    hour_features['평일_퇴근시간'] = hour_features['hour'].isin([17, 18, 19]).astype(int)
    hour_features['점심시간'] = hour_features['hour'].isin([11, 12, 13]).astype(int)
    hour_features['심야시간'] = hour_features['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
    hour_features['주간시간'] = hour_features['hour'].isin(list(range(6, 22))).astype(int)
    
    return hour_features


def normalize_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """기온 정규화"""
    temp_cols = ['avg_temp_c', 'min_temp_c', 'max_temp_c']
    
    for col in temp_cols:
        # Min-Max 정규화 (-1 ~ 1 범위)
        temp_min, temp_max = df[col].min(), df[col].max()
        df[f'{col}_정규화'] = 2 * (df[col] - temp_min) / (temp_max - temp_min) - 1
    
    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """공휴일 관련 특징 확장"""
    # 기존 is_holiday는 유지
    
    # 주말 여부
    df['주말여부'] = df['weekday'].isin([5, 6]).astype(int)  # 토, 일
    
    # 공휴일 + 주말 = 휴일
    df['휴일여부'] = ((df['is_holiday'] == 1) | (df['주말여부'] == 1)).astype(int)
    
    # 평일 여부
    df['평일여부'] = (df['휴일여부'] == 0).astype(int)
    
    return df


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """계절 특징 추가"""
    df['date_parsed'] = pd.to_datetime(df['date'])
    month = df['date_parsed'].dt.month
    
    # 계절 분류 (한국 기준)
    df['봄'] = month.isin([3, 4, 5]).astype(int)
    df['여름'] = month.isin([6, 7, 8]).astype(int)
    df['가을'] = month.isin([9, 10, 11]).astype(int)
    df['겨울'] = month.isin([12, 1, 2]).astype(int)
    
    df = df.drop('date_parsed', axis=1)
    return df


def build_temporal_features(date_info_file: Path, output_file: Path, 
                          hour_features_file: Path = None) -> None:
    """시간/외생 특징 통합 생성"""
    
    # 날짜 정보 로드
    df = pd.read_csv(date_info_file)
    print(f"원본 날짜 데이터: {len(df)}일, {len(df.columns)}개 특징")
    
    # 특징 확장
    print("요일 one-hot 인코딩 추가...")
    df = add_weekday_onehot(df)
    
    print("시간 순환 특징 추가...")
    df = add_time_cyclical_features(df)
    
    print("기온 정규화...")
    df = normalize_temperature(df)
    
    print("공휴일 특징 확장...")
    df = add_holiday_features(df)
    
    print("계절 특징 추가...")
    df = add_seasonal_features(df)
    
    # 저장
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"확장된 날짜 특징 저장: {output_file}")
    print(f"최종 특징 수: {len(df.columns)}개")
    
    # 시간대별 특징 생성 및 저장
    if hour_features_file:
        hour_df = add_hour_cyclical_features()
        hour_df.to_csv(hour_features_file, index=False, encoding='utf-8-sig')
        print(f"시간대별 특징 저장: {hour_features_file}")
        print(f"시간대별 특징 수: {len(hour_df.columns)}개")
    
    # 특징 요약 출력
    print_feature_summary(df, hour_df if hour_features_file else None)


def print_feature_summary(date_df: pd.DataFrame, hour_df: pd.DataFrame = None) -> None:
    """특징 요약 출력"""
    print(f"\n=== 시간/외생 특징 요약 ===")
    
    # 날짜별 특징
    weekday_cols = [col for col in date_df.columns if col.startswith('요일_')]
    temp_cols = [col for col in date_df.columns if '_정규화' in col]
    cyclical_cols = [col for col in date_df.columns if '_sin' in col or '_cos' in col]
    holiday_cols = [col for col in date_df.columns if '여부' in col or col == 'is_holiday']
    season_cols = ['봄', '여름', '가을', '겨울']
    
    print(f"날짜별 특징:")
    print(f"  - 요일 one-hot: {len(weekday_cols)}개")
    print(f"  - 기온 정규화: {len(temp_cols)}개")
    print(f"  - 순환 특징: {len(cyclical_cols)}개")
    print(f"  - 공휴일/휴일: {len(holiday_cols)}개")
    print(f"  - 계절: {len(season_cols)}개")
    print(f"  - 총 날짜 특징: {len(date_df.columns)}개")
    
    # 시간대별 특징
    if hour_df is not None:
        time_cyclical = [col for col in hour_df.columns if '_sin' in col or '_cos' in col]
        time_binary = [col for col in hour_df.columns if '시간' in col and col not in time_cyclical]
        
        print(f"\n시간대별 특징:")
        print(f"  - 시간 순환: {len(time_cyclical)}개")
        print(f"  - 시간대 구분: {len(time_binary)}개")
        print(f"  - 총 시간 특징: {len(hour_df.columns)}개")
    
    # 샘플 출력
    print(f"\n=== 샘플 (2025-01-01) ===")
    if '2025-01-01' in date_df['date'].values:
        sample = date_df[date_df['date'] == '2025-01-01'].iloc[0]
        for col in date_df.columns:
            if col != 'date':
                print(f"{col:20s}: {sample[col]}")


def main():
    parser = argparse.ArgumentParser(description="시간/외생 특징 확장")
    parser.add_argument(
        "--input", 
        type=Path, 
        default=Path("data/날짜정보.csv"),
        help="입력 날짜정보 CSV 파일"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=Path("data/temporal_features.csv"),
        help="출력 확장된 시간 특징 CSV 파일"
    )
    parser.add_argument(
        "--hour-features", 
        type=Path, 
        default=Path("data/hour_features.csv"),
        help="시간대별 특징 CSV 파일"
    )
    
    args = parser.parse_args()
    
    build_temporal_features(args.input, args.output, args.hour_features)
    print(f"\n시간/외생 특징 확장 완료!")


if __name__ == "__main__":
    main()
