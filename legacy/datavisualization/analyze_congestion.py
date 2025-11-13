"""
LOS 등급 기반 과밀 시간대 분석 및 지도 시각화

과밀(LOS E/F)인 시간대와 요일을 찾아서 지리데이터로 시각화하고 이미지로 저장합니다.
"""

import os
import re
import pandas as pd
import folium
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import argparse


def read_csv_safe(path: str, encodings: List[str] = None) -> pd.DataFrame:
    """여러 인코딩을 시도하여 CSV 읽기"""
    if encodings is None:
        encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8", "latin-1"]
    
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    
    raise ValueError(f"CSV 파일을 읽을 수 없습니다: {path}")


def parse_traffic_data(traffic_df: pd.DataFrame) -> pd.DataFrame:
    """승하차 데이터를 파싱하여 구조화된 DataFrame으로 변환"""
    
    # 컬럼 찾기
    date_col = None
    name_col = None
    dir_col = None
    
    # 날짜 컬럼 찾기
    for col in traffic_df.columns:
        if any(k in str(col) for k in ['날짜', '일자', 'DATE', 'date', '영업월']):
            date_col = col
            break
    if date_col is None:
        date_col = traffic_df.columns[0]
    
    # 역명 컬럼 찾기
    for col in traffic_df.columns:
        if any(k in str(col) for k in ['역명', '역', 'station', 'name']):
            name_col = col
            break
    if name_col is None:
        # 시간대 컬럼이 아닌 것을 찾기
        time_cols = [c for c in traffic_df.columns if '시' in str(c) or 'hour' in str(c).lower()]
        for i, col in enumerate(traffic_df.columns):
            if col not in time_cols and col != date_col:
                name_col = col
                break
        if name_col is None:
            name_col = traffic_df.columns[1] if len(traffic_df.columns) > 1 else None
    
    # 승하차 구분 컬럼 찾기
    for col in traffic_df.columns:
        if any(k in str(col) for k in ['승하차', '구분', 'direction', 'type']):
            dir_col = col
            break
    if dir_col is None:
        # 4번째 컬럼을 시도
        if len(traffic_df.columns) > 3:
            candidate_cols = [c for c in traffic_df.columns if c not in [date_col, name_col]]
            time_cols = [c for c in candidate_cols if '시' in str(c)]
            for col in candidate_cols:
                if col not in time_cols:
                    dir_col = col
                    break
    
    print(f"컬럼 매핑: 날짜={date_col}, 역명={name_col}, 구분={dir_col}")
    
    # 시간대 컬럼 찾기 (XX-YY시 형식)
    hour_columns = {}
    for col in traffic_df.columns:
        # "03-04시" 형식 파싱
        match = re.match(r'(\d{2})-(\d{2})시', str(col))
        if match:
            hour = int(match.group(1))
            if 0 <= hour < 24:
                hour_columns[hour] = col
        # "00시", "01시" 형식
        elif re.match(r'^\d{1,2}시$', str(col)):
            hour = int(re.match(r'(\d{1,2})시', str(col)).group(1))
            if 0 <= hour < 24:
                hour_columns[hour] = col
    
    print(f"시간대 컬럼 {len(hour_columns)}개 찾음")
    
    # 데이터 변환
    records = []
    for _, row in traffic_df.iterrows():
        date = str(row[date_col]).strip()
        station = str(row[name_col]).strip() if name_col else ''
        direction = str(row[dir_col]).strip() if dir_col else '합계'
        
        # 시간대별 데이터 추출
        for hour, col_name in hour_columns.items():
            try:
                value = int(float(row[col_name])) if pd.notna(row[col_name]) else 0
            except:
                value = 0
            
            records.append({
                'date': date,
                'station': station,
                'direction': direction,
                'hour': hour,
                'traffic': value
            })
    
    result_df = pd.DataFrame(records)
    print(f"총 {len(result_df)}개 레코드 생성")
    return result_df


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


def analyze_congestion(
    traffic_df: pd.DataFrame,
    map_df: pd.DataFrame,
    mode: str = 'waiting',  # 'waiting' or 'walking'
    platform_ratio: float = 0.35,
    corridor_ratio: float = 0.20,
    wait_min: float = 5.0,
    transit_sec: float = 40.0,
    min_los_grade: str = 'E'  # 'E' or 'F' (E 이상을 과밀으로 간주)
) -> pd.DataFrame:
    """
    통행량과 면적 정보를 결합하여 LOS 등급을 계산하고 과밀 구간을 찾습니다.
    
    Returns:
        DataFrame with columns: date, station, hour, direction, los_grade, traffic, etc.
    """
    
    # 면적 정보 준비
    total_area_col = None
    for col in map_df.columns:
        if '연면적' in str(col):
            total_area_col = col
            break
    
    # 면적 계산
    if total_area_col:
        map_df['total_area'] = pd.to_numeric(map_df[total_area_col], errors='coerce')
        if mode == 'waiting':
            map_df['area'] = map_df['total_area'] * platform_ratio
        else:
            map_df['area'] = map_df['total_area'] * corridor_ratio
    else:
        print("경고: 연면적 컬럼을 찾을 수 없습니다. 기본값 사용.")
        map_df['area'] = 1000.0  # 기본값
    
    # 역명 정규화 (역명 뒤 '역' 제거 또는 추가)
    map_df['station_normalized'] = map_df['역명'].str.replace('역', '', regex=False).str.strip()
    traffic_df['station_normalized'] = traffic_df['station'].str.replace('역', '', regex=False).str.strip()
    
    # 정확히 일치하는 경우와 정규화된 경우 모두 시도
    # 먼저 정확히 일치하는 경우
    merged = traffic_df.merge(
        map_df[['역명', 'station_normalized', 'area', 'Latitude', 'Longitude', '역구성순서']],
        left_on='station',
        right_on='역명',
        how='left',
        suffixes=('', '_map')
    )
    
    # 매칭되지 않은 경우 정규화된 이름으로 재시도
    unmatched = merged[merged['역명'].isna()]
    if len(unmatched) > 0:
        for idx, row in unmatched.iterrows():
            station_name = row['station_normalized']
            match = map_df[map_df['station_normalized'] == station_name]
            if len(match) > 0:
                matched_row = match.iloc[0]
                merged.loc[idx, '역명'] = matched_row['역명']
                merged.loc[idx, 'area'] = matched_row['area']
                merged.loc[idx, 'Latitude'] = matched_row['Latitude']
                merged.loc[idx, 'Longitude'] = matched_row['Longitude']
                merged.loc[idx, '역구성순서'] = matched_row['역구성순서']
    
    # LOS 등급 계산
    results = []
    for _, row in merged.iterrows():
        if pd.isna(row['area']) or row['area'] <= 0:
            continue
        
        traffic_value = row['traffic']
        area = row['area']
        
        if mode == 'waiting':
            # 승차 인원 기준으로 대기공간 계산
            boarding = row['traffic'] if row['direction'] == '승차' else 0
            # 평균 체류 인원 = 승차 인원 * (대기 시간 / 60)
            occupancy = boarding * (wait_min / 60.0)
            # 점유면적 (㎡/인)
            if occupancy > 0:
                area_per_person = area / occupancy
            else:
                area_per_person = float('inf')
            
            los_grade = calculate_los_waiting(area_per_person)
            density = occupancy / area if area > 0 else 0
        else:  # walking
            # 보행공간: 승차+하차 합계를 사용
            # 현재 행이 승차/하차 중 하나면 다른 방향 데이터를 찾아야 함
            if row['direction'] == '합계':
                total_flow = row['traffic']
            elif row['direction'] in ['승차', '하차']:
                # 같은 역, 같은 날짜, 같은 시간대의 다른 방향 데이터 찾기
                # merged DataFrame에서 찾기 (인덱스 기준으로 빠르게)
                station_key = row['station']
                date_key = row['date']
                hour_key = row['hour']
                
                # 같은 조건의 모든 행 찾기
                matching = merged[
                    (merged['station'] == station_key) &
                    (merged['date'] == date_key) &
                    (merged['hour'] == hour_key)
                ]
                
                boarding = matching[matching['direction'] == '승차']['traffic'].sum()
                alighting = matching[matching['direction'] == '하차']['traffic'].sum()
                total_flow = boarding + alighting
                
                # 찾지 못한 경우 현재 값 사용
                if total_flow == 0 and row['traffic'] > 0:
                    total_flow = row['traffic']
            else:
                total_flow = row['traffic']
            
            # 통로 평균 체류 인원 = 통과 인원 * (통과 시간 / 3600)
            occupancy = total_flow * (transit_sec / 3600.0)
            # 밀도 (인/㎡)
            density = occupancy / area if area > 0 else 0
            
            los_grade = calculate_los_walking(density)
            area_per_person = area / max(occupancy, 1e-9) if occupancy > 0 else float('inf')
        
        # 요일 계산
        try:
            date_obj = pd.to_datetime(row['date'])
            weekday = date_obj.weekday()  # 0=월요일, 6=일요일
            weekday_name = ['월', '화', '수', '목', '금', '토', '일'][weekday]
        except:
            weekday = None
            weekday_name = '알수없음'
        
        results.append({
            'date': row['date'],
            'weekday': weekday,
            'weekday_name': weekday_name,
            'station': row['역명'] if pd.notna(row['역명']) else row['station'],
            'hour': row['hour'],
            'direction': row['direction'],
            'traffic': traffic_value,
            'area': area,
            'density': density,
            'area_per_person': area_per_person,
            'los_grade': los_grade,
            'latitude': row['Latitude'],
            'longitude': row['Longitude'],
            'order': row['역구성순서']
        })
    
    result_df = pd.DataFrame(results)
    
    # LOS 등급 필터링 (E, F만)
    los_priority = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
    if min_los_grade == 'E':
        result_df = result_df[result_df['los_grade'].isin(['E', 'F'])]
    elif min_los_grade == 'F':
        result_df = result_df[result_df['los_grade'] == 'F']
    
    # 등급별 우선순위 추가
    result_df['los_priority'] = result_df['los_grade'].map(los_priority)
    
    return result_df


def create_congestion_map(
    congestion_df: pd.DataFrame,
    map_df: pd.DataFrame,
    output_html: str,
    title: str = "과밀 구간 분석"
) -> folium.Map:
    """과밀 구간을 지도에 시각화"""
    
    # 지도 중심 계산
    center_lat = congestion_df['latitude'].mean()
    center_lon = congestion_df['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron'
    )
    
    # LOS 색상
    los_colors = {
        'A': '#2ECC71',  # 초록
        'B': '#8BC34A',  # 연두
        'C': '#FFC107',  # 노랑
        'D': '#FF9800',  # 주황
        'E': '#FF5722',  # 빨강
        'F': '#8B0000'   # 진한빨강
    }
    
    # 노선 경로 그리기
    route_coords = map_df.sort_values('역구성순서')[['Latitude', 'Longitude']].values.tolist()
    folium.PolyLine(
        locations=route_coords,
        color='#555555',
        weight=4,
        opacity=0.7
    ).add_to(m)
    
    # 각 역별로 가장 심한 LOS 등급 표시
    station_los = congestion_df.groupby('station').agg({
        'los_grade': lambda x: max(x, key=lambda g: {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6}[g]),
        'density': 'max',
        'latitude': 'first',
        'longitude': 'first',
        'order': 'first'
    }).reset_index()
    
    for _, row in station_los.iterrows():
        los = row['los_grade']
        color = los_colors.get(los, '#CCCCCC')
        
        # 역별 과밀 발생 횟수
        station_congestion = congestion_df[congestion_df['station'] == row['station']]
        count = len(station_congestion)
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=15 + count * 2,  # 과밀 횟수에 비례하여 크기 증가
            popup=f"{row['station']}<br>최고 LOS: {los}<br>과밀 발생: {count}회",
            tooltip=f"{int(row['order'])}. {row['station']} (LOS {los})",
            color=color,
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
        
        # 역명 레이블
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            icon=folium.DivIcon(
                html=f"<div style='font-size:12px; font-weight:bold; color:#000;'>{row['station'].replace('역', '')}</div>",
                icon_size=(100, 20),
                icon_anchor=(50, 10),
                class_name='station-label'
            )
        ).add_to(m)
    
    # 범례 추가
    legend_html = f"""
    <div style="position: fixed; bottom: 50px; left: 50px; width: 200px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
    <h4 style="margin: 0 0 10px 0;">{title}</h4>
    <p style="margin: 2px 0; font-size: 11px;">원 크기: 과밀 발생 횟수</p>
    <div style="margin-top: 5px;">
    """
    
    for grade in ['E', 'F']:
        color = los_colors.get(grade, '#CCCCCC')
        legend_html += f"""
        <div style="display:flex; align-items:center; margin:3px 0;">
          <span style="display:inline-block; width:16px; height:16px; background:{color}; border:1px solid #999; margin-right:5px;"></span>
          <span>LOS {grade}</span>
        </div>
        """
    
    legend_html += """
    </div>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # 통계 정보 추가
    total_incidents = len(congestion_df)
    unique_stations = congestion_df['station'].nunique()
    stats_html = f"""
    <div style="position: fixed; top: 10px; right: 10px; width: 250px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:12px; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
    <h4 style="margin: 0 0 10px 0;">과밀 통계</h4>
    <p style="margin: 2px 0;">총 과밀 발생: {total_incidents}회</p>
    <p style="margin: 2px 0;">과밀 역 수: {unique_stations}개</p>
    <p style="margin: 2px 0;">LOS E: {len(congestion_df[congestion_df['los_grade']=='E'])}회</p>
    <p style="margin: 2px 0;">LOS F: {len(congestion_df[congestion_df['los_grade']=='F'])}회</p>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(stats_html))
    
    m.save(output_html)
    print(f"지도 저장 완료: {output_html}")
    
    return m


def save_map_as_image(html_path: str, output_png: str):
    """HTML 지도를 PNG 이미지로 변환"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            service = Service(ChromeDriverManager().install())
        except:
            service = Service()  # 시스템 PATH의 chromedriver 사용
        
        options = Options()
        options.add_argument('--headless=new')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-gpu')
        
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(f'file://{os.path.abspath(html_path)}')
        
        import time
        time.sleep(2)  # 지도 로딩 대기
        
        driver.save_screenshot(output_png)
        driver.quit()
        
        print(f"이미지 저장 완료: {output_png}")
    except Exception as e:
        print(f"이미지 저장 실패 (Selenium 필요): {e}")
        print(f"HTML 파일은 생성되었습니다: {html_path}")


def main():
    parser = argparse.ArgumentParser(description='LOS 등급 기반 과밀 시간대 분석 및 지도 시각화')
    parser.add_argument('--map', default='data/지도.csv', help='지도 CSV 파일 경로')
    parser.add_argument('--traffic', default='data/승하차.csv', help='승하차 데이터 CSV 파일 경로')
    parser.add_argument('--output-dir', default='output', help='출력 디렉토리')
    parser.add_argument('--mode', choices=['waiting', 'walking'], default='waiting', help='분석 모드')
    parser.add_argument('--min-los', choices=['E', 'F'], default='E', help='최소 LOS 등급 (E 또는 F)')
    parser.add_argument('--save-png', action='store_true', help='PNG 이미지로도 저장')
    parser.add_argument('--platform-ratio', type=float, default=0.35, help='총면적 대비 플랫폼 비율')
    parser.add_argument('--corridor-ratio', type=float, default=0.20, help='총면적 대비 통로 비율')
    parser.add_argument('--wait-min', type=float, default=5.0, help='승차 대기 시간(분)')
    parser.add_argument('--transit-sec', type=float, default=40.0, help='통로 체류 시간(초)')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== 데이터 로드 ===")
    map_df = read_csv_safe(args.map)
    print(f"지도 데이터: {len(map_df)}개 역")
    
    traffic_raw = read_csv_safe(args.traffic)
    print(f"승하차 데이터: {len(traffic_raw)}개 레코드")
    
    print("\n=== 데이터 파싱 ===")
    traffic_df = parse_traffic_data(traffic_raw)
    
    print("\n=== LOS 등급 분석 ===")
    congestion_df = analyze_congestion(
        traffic_df=traffic_df,
        map_df=map_df,
        mode=args.mode,
        platform_ratio=args.platform_ratio,
        corridor_ratio=args.corridor_ratio,
        wait_min=args.wait_min,
        transit_sec=args.transit_sec,
        min_los_grade=args.min_los
    )
    
    print(f"과밀 발생: {len(congestion_df)}회")
    print(f"과밀 역: {congestion_df['station'].nunique()}개")
    
    # 요일/시간대별 통계
    print("\n=== 요일별 통계 ===")
    weekday_stats = congestion_df.groupby('weekday_name').size()
    print(weekday_stats)
    
    print("\n=== 시간대별 통계 ===")
    hour_stats = congestion_df.groupby('hour').size()
    print(hour_stats)
    
    # 결과를 CSV로 저장
    output_csv = os.path.join(args.output_dir, f'congestion_analysis_{args.mode}.csv')
    congestion_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n분석 결과 저장: {output_csv}")
    
    # 지도 생성
    print("\n=== 지도 생성 ===")
    output_html = os.path.join(args.output_dir, f'congestion_map_{args.mode}.html')
    create_congestion_map(
        congestion_df=congestion_df,
        map_df=map_df,
        output_html=output_html,
        title=f"과밀 구간 분석 ({args.mode} 모드, LOS {args.min_los} 이상)"
    )
    
    # PNG로 저장
    if args.save_png:
        print("\n=== 이미지 저장 ===")
        output_png = os.path.join(args.output_dir, f'congestion_map_{args.mode}.png')
        save_map_as_image(output_html, output_png)
    
    print("\n=== 완료 ===")


if __name__ == '__main__':
    main()
