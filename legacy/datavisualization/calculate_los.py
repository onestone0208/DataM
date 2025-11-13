"""
LOS 등급 계산 및 CSV 저장

통계.csv 파일을 읽어서 LOS 등급을 계산하고 CSV로 저장합니다.
통계.csv 구조: 역명, 월, 요일, 구분, 03-04시, 04-05시, ..., 02-03시
"""

import os
import re
import pandas as pd
from typing import List
import argparse


def read_csv_safe(path: str, encodings: List[str] = None) -> pd.DataFrame:
    """여러 인코딩을 시도하여 CSV 읽기"""
    if encodings is None:
        encodings = ["cp949", "utf-8-sig", "euc-kr", "utf-8", "latin-1"]
    
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    
    raise ValueError(f"CSV 파일을 읽을 수 없습니다: {path}")


def calculate_los_waiting(area_per_person: float) -> str:
    """대기공간 LOS 등급 계산 (점유면적 기준, ㎡/인)"""
    if not pd.notna(area_per_person) or area_per_person <= 0:
        return 'A'
    if area_per_person >= 1.18:
        return 'A'
    elif area_per_person >= 0.78:
        return 'B'
    elif area_per_person >= 0.54:
        return 'C'
    elif area_per_person >= 0.34:
        return 'D'
    elif area_per_person >= 0.23:
        return 'E'
    else:
        return 'F'


def calculate_los_walking(density: float) -> str:
    """보행공간 LOS 등급 계산 (밀도 기준, 인/㎡)"""
    if not pd.notna(density) or density < 0:
        return 'A'
    if density <= 0.31:
        return 'A'
    elif density <= 0.43:
        return 'B'
    elif density <= 0.71:
        return 'C'
    elif density <= 1.06:
        return 'D'
    elif density <= 1.79:
        return 'E'
    else:
        return 'F'


def calculate_los_from_stats(
    stats_df: pd.DataFrame,
    map_df: pd.DataFrame,
    mode: str = 'waiting',
    platform_ratio: float = 0.35,
    corridor_ratio: float = 0.20,
    wait_min: float = 5.0,
    transit_sec: float = 40.0
) -> pd.DataFrame:
    """
    통계.csv와 지도.csv를 결합하여 LOS 등급을 계산합니다.
    
    통계.csv 구조:
    - 역명: 역 이름
    - 월: 월 정보
    - 요일: 요일 (월요일, 화요일, ...)
    - 구분: 승차, 하차, 합계
    - 03-04시, 04-05시, ...: 시간대별 통행량
    """
    
    print("=== 데이터 구조 확인 ===")
    print(f"통계 데이터 컬럼: {list(stats_df.columns[:10])}")
    print(f"통계 데이터 행 수: {len(stats_df)}")
    
    # 통계.csv 구조 확인: 역명, 월, 요일, 구분, 시간대...
    station_col = '역명'
    month_col = '월'
    weekday_col = '요일'
    direction_col = '구분'
    
    # 시간대 컬럼 찾기 (XX-YY시 형식)
    hour_cols = {}
    for col in stats_df.columns:
        match = re.match(r'(\d{2})-(\d{2})시', str(col))
        if match:
            hour = int(match.group(1))
            if 0 <= hour < 24:
                hour_cols[hour] = col
    
    print(f"시간대 컬럼 {len(hour_cols)}개 찾음: {list(hour_cols.keys())[:5]}...")
    
    # 지도 데이터 면적 정보 준비
    total_area_col = None
    for col in map_df.columns:
        if '연면적' in str(col):
            total_area_col = col
            break
    
    if total_area_col:
        map_df['total_area'] = pd.to_numeric(map_df[total_area_col], errors='coerce')
        if mode == 'waiting':
            map_df['area'] = map_df['total_area'] * platform_ratio
        else:
            map_df['area'] = map_df['total_area'] * corridor_ratio
        
        print(f"\n면적 정보:")
        print(f"  연면적 범위: {map_df['total_area'].min():.1f} ~ {map_df['total_area'].max():.1f} ㎡")
        print(f"  사용 면적 범위: {map_df['area'].min():.1f} ~ {map_df['area'].max():.1f} ㎡ (비율: {platform_ratio if mode == 'waiting' else corridor_ratio})")
    else:
        print("경고: 연면적 컬럼을 찾을 수 없습니다. 기본값 사용.")
        map_df['area'] = 1000.0
    
    # 역명 정규화 (지도 데이터)
    map_df['역명_정규화'] = map_df['역명'].str.replace('역', '', regex=False).str.strip()
    map_dict = map_df.set_index('역명')[['area', 'Latitude', 'Longitude', '역구성순서']].to_dict('index')
    
    # 요일 매핑
    weekday_map = {
        '월요일': ('월', 0),
        '화요일': ('화', 1),
        '수요일': ('수', 2),
        '목요일': ('목', 3),
        '금요일': ('금', 4),
        '토요일': ('토', 5),
        '일요일': ('일', 6),
    }
    
    # 결과 저장용
    results = []
    
    print("\n=== LOS 등급 계산 중 ===")
    processed = 0
    
    # 통계 데이터를 순회하며 계산
    for idx, row in stats_df.iterrows():
        station_name = str(row[station_col]).strip()
        if not station_name or station_name == 'nan':
            continue
        
        # 지도 데이터에서 역 찾기
        station_info = None
        if station_name in map_dict:
            station_info = map_dict[station_name]
        else:
            # 정규화된 이름으로 찾기
            for map_station, info in map_dict.items():
                if map_df[map_df['역명'] == map_station]['역명_정규화'].iloc[0] == station_name.replace('역', '').strip():
                    station_info = info
                    break
        
        if station_info is None or pd.isna(station_info['area']) or station_info['area'] <= 0:
            continue
        
        area = station_info['area']
        direction = str(row[direction_col]).strip()
        month = int(row[month_col]) if pd.notna(row[month_col]) else None
        weekday_str = str(row[weekday_col]).strip()
        
        # 요일 정보
        weekday_name, weekday_num = weekday_map.get(weekday_str, ('알수없음', None))
        
        # 각 시간대별 계산
        for hour, col_name in hour_cols.items():
            try:
                traffic = float(row[col_name]) if pd.notna(row[col_name]) else 0.0
            except:
                traffic = 0.0
            
            if mode == 'waiting':
                # 대기공간: 승차 인원만 사용
                if direction == '승차':
                    # 통행량은 1시간당 평균값이므로, 그대로 사용
                    # 승강장 체류 인원 = 1시간 평균 승차 인원 * 평균 대기 시간 비율
                    # 예: 1시간에 100명 승차, 평균 5분 대기 → 100 * (5/60) = 8.3명이 평균적으로 승강장에 있음
                    boarding = traffic  # 시간당 승차 인원
                    occupancy = boarding * (wait_min / 60.0)  # 평균 체류 인원
                    
                    if occupancy > 0:
                        area_per_person = area / occupancy
                    else:
                        area_per_person = float('inf')
                    
                    los_grade = calculate_los_waiting(area_per_person)
                    
                    # 디버깅 출력 (처음 몇 개만, 피크 시간대)
                    if processed < 3 and hour in [7, 8, 9, 17, 18]:
                        print(f"  샘플: {station_name} {hour}시 - 통행량={traffic:.1f}명/시간, 면적={area:.1f}㎡, 체류={occupancy:.2f}명, ㎡/인={area_per_person:.2f}, LOS={los_grade}")
                    density = occupancy / area if area > 0 else 0
                else:
                    continue  # 승차만 계산
            else:
                # 보행공간: 합계 또는 승차+하차
                if direction == '합계':
                    total_flow = traffic
                elif direction in ['승차', '하차']:
                    # 같은 역, 같은 월, 같은 요일, 같은 시간대의 다른 방향 찾기
                    same_time = stats_df[
                        (stats_df[station_col] == station_name) &
                        (stats_df[month_col] == month) &
                        (stats_df[weekday_col] == weekday_str) &
                        (stats_df[direction_col] != direction)
                    ]
                    if len(same_time) > 0 and col_name in same_time.columns:
                        other_flow = same_time[col_name].sum()
                        total_flow = traffic + other_flow
                    else:
                        total_flow = traffic
                else:
                    total_flow = traffic
                
                occupancy = total_flow * (transit_sec / 3600.0)
                density = occupancy / area if area > 0 else 0
                los_grade = calculate_los_walking(density)
                area_per_person = area / max(occupancy, 1e-9) if occupancy > 0 else float('inf')
            
            results.append({
                'month': month,
                'weekday': weekday_num,
                'weekday_name': weekday_name,
                'station': station_name,
                'hour': hour,
                'direction': direction,
                'traffic': traffic,
                'area': area,
                'density': density if mode == 'walking' else 0,
                'area_per_person': area_per_person if mode == 'waiting' and pd.notna(area_per_person) and area_per_person != float('inf') else None,
                'los_grade': los_grade,
                'latitude': station_info['Latitude'],
                'longitude': station_info['Longitude'],
                'order': station_info['역구성순서']
            })
        
        processed += 1
        if processed % 100 == 0:
            print(f"처리 중: {processed}/{len(stats_df)}")
    
    result_df = pd.DataFrame(results)
    
    if len(result_df) == 0:
        print("경고: 계산된 결과가 없습니다.")
        return result_df
    
    # 등급별 우선순위 추가
    los_priority = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
    result_df['los_priority'] = result_df['los_grade'].map(los_priority)
    
    print(f"\n계산 완료: {len(result_df)}개 레코드")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(description='LOS 등급 계산 및 CSV 저장')
    parser.add_argument('--map', default='data/지도.csv', help='지도 CSV 파일 경로')
    parser.add_argument('--stats', default='data/통계.csv', help='통계 CSV 파일 경로')
    parser.add_argument('--output', default='data/los_calculated.csv', help='출력 CSV 파일 경로')
    parser.add_argument('--mode', choices=['waiting', 'walking'], default='waiting', help='분석 모드')
    parser.add_argument('--platform-ratio', type=float, default=0.35, help='총면적 대비 플랫폼 비율')
    parser.add_argument('--corridor-ratio', type=float, default=0.20, help='총면적 대비 통로 비율')
    parser.add_argument('--wait-min', type=float, default=5.0, help='승차 대기 시간(분)')
    parser.add_argument('--transit-sec', type=float, default=40.0, help='통로 체류 시간(초)')
    
    args = parser.parse_args()
    
    print("=== 데이터 로드 ===")
    map_df = read_csv_safe(args.map)
    print(f"지도 데이터: {len(map_df)}개 역")
    
    stats_df = read_csv_safe(args.stats)
    print(f"통계 데이터: {len(stats_df)}개 레코드")
    
    print("\n=== LOS 등급 계산 ===")
    los_df = calculate_los_from_stats(
        stats_df=stats_df,
        map_df=map_df,
        mode=args.mode,
        platform_ratio=args.platform_ratio,
        corridor_ratio=args.corridor_ratio,
        wait_min=args.wait_min,
        transit_sec=args.transit_sec
    )
    
    # 통계 출력
    if len(los_df) > 0:
        print("\n=== LOS 등급 분포 ===")
        los_counts = los_df['los_grade'].value_counts().sort_index()
        for grade in ['A', 'B', 'C', 'D', 'E', 'F']:
            count = los_counts.get(grade, 0)
            pct = (count / len(los_df) * 100) if len(los_df) > 0 else 0
            print(f"  LOS {grade}: {count:,}개 ({pct:.1f}%)")
        
        # 상세 분석: 가장 나쁜 LOS 예시
        if 'area_per_person' in los_df.columns:
            worst = los_df[los_df['area_per_person'].notna()].nsmallest(5, 'area_per_person')
            if len(worst) > 0:
                print(f"\n가장 혼잡한 구간 (상위 5개):")
                for _, row in worst.iterrows():
                    print(f"  {row['station']} {row['hour']}시 - ㎡/인={row['area_per_person']:.2f}, LOS={row['los_grade']}, 통행량={row['traffic']:.1f}")
        
        print("\n=== 요일별 통계 ===")
        if 'weekday_name' in los_df.columns:
            print(los_df['weekday_name'].value_counts().sort_index())
        
        print("\n=== 시간대별 통계 ===")
        print(los_df['hour'].value_counts().sort_index())
        
        print(f"\n총 {len(los_df)}개 레코드 계산 완료")
    
    # CSV 저장
    output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else '.'
    os.makedirs(output_dir, exist_ok=True)
    
    los_df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n결과 저장: {args.output}")


if __name__ == '__main__':
    main()
