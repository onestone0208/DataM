#!/usr/bin/env python3
"""
역별 노드 특징 벡터 생성 스크립트

POI 데이터와 지도 데이터를 결합해서 각 역의 특징 벡터를 생성합니다.
- POI 카테고리별 밀도
- 물리적 특성 (연면적, 층수, 좌표)
- 네트워크 특성 (노선 순서)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import json
import re


def load_station_info(station_file: Path) -> pd.DataFrame:
    """역 기본 정보 로드"""
    try:
        df = pd.read_csv(station_file, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(station_file, encoding='cp949')
    return df


def load_poi_tagged(poi_file: Path) -> pd.DataFrame:
    """태깅된 POI 데이터 로드"""
    try:
        df = pd.read_csv(poi_file, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(poi_file, encoding='cp949')
    return df


def calculate_poi_features(poi_df: pd.DataFrame) -> pd.DataFrame:
    """역별 POI 카테고리 특징 계산"""
    
    # 방향지시와 기타는 제외 (실질적인 POI가 아님)
    meaningful_categories = [
        '교육', '주거', '관공서', '상업', '의료', '교통', 
        '체육문화', '종교', '금융', '숙박'
    ]
    
    poi_meaningful = poi_df[poi_df['category'].isin(meaningful_categories)]
    
    # 역별 카테고리 집계
    poi_counts = poi_meaningful.groupby(['역명', 'category']).size().unstack(fill_value=0)
    
    # 역별 총 POI 수
    total_poi_per_station = poi_counts.sum(axis=1)
    
    # 밀도 계산 (비율)
    poi_density = poi_counts.div(total_poi_per_station, axis=0).fillna(0)
    
    # 컬럼명 변경 (밀도 표시)
    poi_density.columns = [f'{cat}_밀도' for cat in poi_density.columns]
    
    # 총 POI 수와 다양성 지수 추가
    poi_features = poi_density.copy()
    poi_features['총_POI_수'] = total_poi_per_station
    
    # Shannon 다양성 지수 계산
    def shannon_diversity(row):
        proportions = row[row > 0]  # 0이 아닌 값만
        if len(proportions) == 0:
            return 0
        return -np.sum(proportions * np.log(proportions))
    
    poi_features['POI_다양성'] = poi_density.apply(shannon_diversity, axis=1)
    
    return poi_features


def calculate_physical_features(station_df: pd.DataFrame) -> pd.DataFrame:
    """역별 물리적 특징 계산"""
    
    # 필요한 컬럼 선택 (층수 제외)
    physical_cols = ['역명', 'Latitude', 'Longitude', '역구성순서', '연면적(제곱미터)']
    physical_df = station_df[physical_cols].copy()
    
    # 컬럼명 정리
    physical_df = physical_df.rename(columns={
        'Latitude': '위도',
        'Longitude': '경도', 
        '역구성순서': '노선순서',
        '연면적(제곱미터)': '연면적'
    })
    
    # 정규화를 위한 통계 계산
    lat_min, lat_max = physical_df['위도'].min(), physical_df['위도'].max()
    lon_min, lon_max = physical_df['경도'].min(), physical_df['경도'].max()
    area_min, area_max = physical_df['연면적'].min(), physical_df['연면적'].max()
    order_min, order_max = physical_df['노선순서'].min(), physical_df['노선순서'].max()
    
    # Min-Max 정규화
    physical_df['위도_정규화'] = (physical_df['위도'] - lat_min) / (lat_max - lat_min)
    physical_df['경도_정규화'] = (physical_df['경도'] - lon_min) / (lon_max - lon_min)
    physical_df['연면적_정규화'] = (physical_df['연면적'] - area_min) / (area_max - area_min)
    physical_df['노선순서_정규화'] = (physical_df['노선순서'] - order_min) / (order_max - order_min)
    
    return physical_df.set_index('역명')


def combine_features(poi_features: pd.DataFrame, physical_features: pd.DataFrame) -> pd.DataFrame:
    """POI 특징과 물리적 특징 결합"""
    
    # 인덱스 기준으로 결합
    combined = poi_features.join(physical_features, how='outer')
    
    # 결측값 처리
    combined = combined.fillna(0)
    
    return combined


def save_features(features_df: pd.DataFrame, output_file: Path, 
                 feature_info_file: Path = None) -> None:
    """특징 벡터 저장"""
    
    # CSV로 저장
    features_df.to_csv(output_file, encoding='utf-8-sig')
    
    # 특징 정보 저장 (선택적)
    if feature_info_file:
        feature_info = {
            'feature_names': list(features_df.columns),
            'num_features': len(features_df.columns),
            'num_stations': len(features_df),
            'feature_statistics': {
                col: {
                    'mean': float(features_df[col].mean()),
                    'std': float(features_df[col].std()),
                    'min': float(features_df[col].min()),
                    'max': float(features_df[col].max())
                }
                for col in features_df.columns
            }
        }
        
        with open(feature_info_file, 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, ensure_ascii=False, indent=2)


def print_feature_summary(features_df: pd.DataFrame) -> None:
    """특징 요약 출력"""
    print(f"\n=== 노드 특징 벡터 요약 ===")
    print(f"역 수: {len(features_df)}개")
    print(f"특징 차원: {len(features_df.columns)}개")
    
    print(f"\n=== 특징 카테고리 ===")
    poi_cols = [col for col in features_df.columns if '_밀도' in col]
    physical_cols = [col for col in features_df.columns if '_정규화' in col]
    other_cols = [col for col in features_df.columns if col not in poi_cols + physical_cols]
    
    print(f"POI 밀도 특징: {len(poi_cols)}개")
    print(f"물리적 특징: {len(physical_cols)}개") 
    print(f"기타 특징: {len(other_cols)}개")
    
    print(f"\n=== 샘플 (갈마역) ===")
    if '갈마역' in features_df.index:
        sample = features_df.loc['갈마역']
        for col in features_df.columns:
            print(f"{col:15s}: {sample[col]:.3f}")


def main():
    parser = argparse.ArgumentParser(description="역별 노드 특징 벡터 생성")
    parser.add_argument(
        "--station-info", 
        type=Path, 
        default=Path("data/지도.csv"),
        help="역 기본 정보 CSV 파일"
    )
    parser.add_argument(
        "--poi-tagged", 
        type=Path, 
        default=Path("data/주요장소_tagged_v2.csv"),
        help="태깅된 POI CSV 파일"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=Path("data/node_features.csv"),
        help="출력 특징 벡터 CSV 파일"
    )
    parser.add_argument(
        "--feature-info", 
        type=Path, 
        help="특징 정보 JSON 파일 (선택적)"
    )
    
    args = parser.parse_args()
    
    print("데이터 로딩 중...")
    station_df = load_station_info(args.station_info)
    poi_df = load_poi_tagged(args.poi_tagged)
    
    print("POI 특징 계산 중...")
    poi_features = calculate_poi_features(poi_df)
    
    print("물리적 특징 계산 중...")
    physical_features = calculate_physical_features(station_df)
    
    print("특징 결합 중...")
    combined_features = combine_features(poi_features, physical_features)
    
    print("결과 저장 중...")
    save_features(combined_features, args.output, args.feature_info)
    
    print_feature_summary(combined_features)
    
    print(f"\n노드 특징 벡터 생성 완료!")
    print(f"저장 위치: {args.output}")


if __name__ == "__main__":
    main()
