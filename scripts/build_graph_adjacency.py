#!/usr/bin/env python3
"""
지하철 역간 그래프 인접행렬 생성 스크립트

지도.csv의 역 정보를 기반으로 거리 기반 가중 인접행렬을 생성합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import json
from geopy.distance import geodesic


def load_station_info(station_file: Path) -> pd.DataFrame:
    """역 정보 로드"""
    try:
        df = pd.read_csv(station_file, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(station_file, encoding='cp949')
    return df


def calculate_distance_matrix(stations_df: pd.DataFrame) -> np.ndarray:
    """역간 거리 행렬 계산 (km 단위)"""
    n_stations = len(stations_df)
    distance_matrix = np.zeros((n_stations, n_stations))
    
    for i in range(n_stations):
        for j in range(n_stations):
            if i != j:
                coord1 = (stations_df.iloc[i]['Latitude'], stations_df.iloc[i]['Longitude'])
                coord2 = (stations_df.iloc[j]['Latitude'], stations_df.iloc[j]['Longitude'])
                distance_matrix[i][j] = geodesic(coord1, coord2).kilometers
    
    return distance_matrix


def create_adjacency_matrix(stations_df: pd.DataFrame, 
                          distance_threshold: float = 5.0,
                          sigma: float = 2.0) -> np.ndarray:
    """거리 기반 가중 인접행렬 생성"""
    
    # 거리 행렬 계산
    distance_matrix = calculate_distance_matrix(stations_df)
    n_stations = len(stations_df)
    
    # 인접행렬 초기화
    adjacency_matrix = np.zeros((n_stations, n_stations))
    
    # 노선 순서 기반 직접 연결 (가중치 1.0)
    stations_df_sorted = stations_df.sort_values('역구성순서')
    for i in range(len(stations_df_sorted) - 1):
        curr_idx = stations_df_sorted.index[i]
        next_idx = stations_df_sorted.index[i + 1]
        
        # 양방향 연결
        adjacency_matrix[curr_idx][next_idx] = 1.0
        adjacency_matrix[next_idx][curr_idx] = 1.0
    
    # 거리 기반 추가 연결 (Gaussian kernel)
    for i in range(n_stations):
        for j in range(n_stations):
            if i != j and adjacency_matrix[i][j] == 0:  # 직접 연결되지 않은 경우
                distance = distance_matrix[i][j]
                if distance <= distance_threshold:
                    # Gaussian kernel 가중치
                    weight = np.exp(-(distance ** 2) / (2 * sigma ** 2))
                    if weight > 0.1:  # 임계값 이상만 연결
                        adjacency_matrix[i][j] = weight
    
    return adjacency_matrix, distance_matrix


def save_graph_data(stations_df: pd.DataFrame, 
                   adjacency_matrix: np.ndarray,
                   distance_matrix: np.ndarray,
                   output_dir: Path) -> None:
    """그래프 데이터 저장"""
    
    # 역 이름 매핑
    station_names = stations_df['역명'].tolist()
    station_to_idx = {name: idx for idx, name in enumerate(station_names)}
    
    # NumPy 배열로 저장
    np.save(output_dir / "adjacency_matrix.npy", adjacency_matrix)
    np.save(output_dir / "distance_matrix.npy", distance_matrix)
    
    # 역 매핑 정보 저장
    mapping_info = {
        "station_names": station_names,
        "station_to_idx": station_to_idx,
        "n_stations": len(station_names),
        "adjacency_shape": adjacency_matrix.shape,
        "distance_shape": distance_matrix.shape
    }
    
    with open(output_dir / "graph_info.json", 'w', encoding='utf-8') as f:
        json.dump(mapping_info, f, ensure_ascii=False, indent=2)
    
    # 인접행렬 통계
    print(f"=== 그래프 인접행렬 생성 완료 ===")
    print(f"역 수: {len(station_names)}개")
    print(f"인접행렬 크기: {adjacency_matrix.shape}")
    print(f"총 연결 수: {np.count_nonzero(adjacency_matrix)}개")
    print(f"연결 밀도: {np.count_nonzero(adjacency_matrix) / (len(station_names) ** 2):.3f}")
    
    # 연결 통계
    degrees = np.sum(adjacency_matrix > 0, axis=1)
    print(f"평균 연결도: {degrees.mean():.2f}")
    print(f"최대 연결도: {degrees.max()}")
    print(f"최소 연결도: {degrees.min()}")
    
    # 샘플 연결 출력
    print(f"\n=== 샘플 연결 (갈마역) ===")
    if '갈마역' in station_to_idx:
        galma_idx = station_to_idx['갈마역']
        connected_indices = np.where(adjacency_matrix[galma_idx] > 0)[0]
        for idx in connected_indices:
            weight = adjacency_matrix[galma_idx][idx]
            distance = distance_matrix[galma_idx][idx]
            print(f"  → {station_names[idx]}: 가중치={weight:.3f}, 거리={distance:.2f}km")


def main():
    parser = argparse.ArgumentParser(description="그래프 인접행렬 생성")
    parser.add_argument(
        "--station-info", 
        type=Path, 
        default=Path("data/지도.csv"),
        help="역 정보 CSV 파일"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("data"),
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--distance-threshold", 
        type=float, 
        default=5.0,
        help="거리 임계값 (km)"
    )
    parser.add_argument(
        "--sigma", 
        type=float, 
        default=2.0,
        help="Gaussian kernel sigma"
    )
    
    args = parser.parse_args()
    
    # 역 정보 로드
    stations_df = load_station_info(args.station_info)
    print(f"역 정보 로드: {len(stations_df)}개 역")
    
    # 인접행렬 생성
    adjacency_matrix, distance_matrix = create_adjacency_matrix(
        stations_df, args.distance_threshold, args.sigma
    )
    
    # 저장
    save_graph_data(stations_df, adjacency_matrix, distance_matrix, args.output_dir)
    
    print(f"\n그래프 데이터 저장 완료: {args.output_dir}")


if __name__ == "__main__":
    main()
