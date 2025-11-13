"""
과밀 구간 지도 시각화

계산된 LOS 데이터를 로드하여 과밀 구간을 지도에 시각화하고 이미지로 저장합니다.
"""

import os
import pandas as pd
import folium
from typing import Optional
import argparse


def read_csv_safe(path: str, encodings: list = None) -> pd.DataFrame:
    """여러 인코딩을 시도하여 CSV 읽기"""
    if encodings is None:
        encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8", "latin-1"]
    
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    
    raise ValueError(f"CSV 파일을 읽을 수 없습니다: {path}")


def create_congestion_map(
    los_df: pd.DataFrame,
    map_df: pd.DataFrame,
    output_html: str,
    min_los_grade: str = 'E',
    title: str = "과밀 구간 분석"
) -> folium.Map:
    """과밀 구간을 지도에 시각화"""
    
    # date 컬럼이 없으면 month/weekday 기반으로 처리
    if 'date' not in los_df.columns and 'month' in los_df.columns:
        # month, weekday 기반 데이터
        if min_los_grade == 'E':
            congestion_df = los_df[los_df['los_grade'].isin(['E', 'F'])].copy()
        elif min_los_grade == 'F':
            congestion_df = los_df[los_df['los_grade'] == 'F'].copy()
        else:
            congestion_df = los_df.copy()
    else:
        # date 기반 데이터 (기존 방식)
        if min_los_grade == 'E':
            congestion_df = los_df[los_df['los_grade'].isin(['E', 'F'])].copy()
        elif min_los_grade == 'F':
            congestion_df = los_df[los_df['los_grade'] == 'F'].copy()
        else:
            congestion_df = los_df.copy()
    
    if len(congestion_df) == 0:
        print(f"\n경고: LOS {min_los_grade} 이상인 데이터가 없습니다.")
        if 'los_grade' in los_df.columns:
            print("현재 데이터의 LOS 등급 분포:")
            los_counts = los_df['los_grade'].value_counts().sort_index()
            for grade in ['A', 'B', 'C', 'D', 'E', 'F']:
                count = los_counts.get(grade, 0)
                if count > 0:
                    print(f"  LOS {grade}: {count}개")
        return None
    
    print(f"과밀 발생: {len(congestion_df)}회")
    print(f"과밀 역: {congestion_df['station'].nunique()}개")
    
    # 좌표가 유효한 데이터만 사용
    congestion_df = congestion_df[
        congestion_df['latitude'].notna() & 
        congestion_df['longitude'].notna()
    ].copy()
    
    if len(congestion_df) == 0:
        print("경고: 좌표 정보가 있는 데이터가 없습니다.")
        return None
    
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
    map_df = map_df.sort_values('역구성순서')
    route_coords = map_df[['Latitude', 'Longitude']].values.tolist()
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
        if pd.isna(row['latitude']) or pd.isna(row['longitude']):
            continue
            
        los = row['los_grade']
        color = los_colors.get(los, '#CCCCCC')
        
        # 역별 과밀 발생 횟수
        station_congestion = congestion_df[congestion_df['station'] == row['station']]
        count = len(station_congestion)
        
        # 역 정보 가져오기
        station_info = map_df[map_df['역명'] == row['station']]
        if len(station_info) == 0:
            station_info = map_df[map_df['역명'].str.replace('역', '', regex=False) == row['station'].replace('역', '')]
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=15 + count * 2,  # 과밀 횟수에 비례하여 크기 증가
            popup=f"{row['station']}<br>최고 LOS: {los}<br>과밀 발생: {count}회",
            tooltip=f"{int(row['order']) if pd.notna(row['order']) else ''}. {row['station']} (LOS {los})",
            color=color,
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
        
        # 역명 레이블
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            icon=folium.DivIcon(
                html=f"<div style='font-size:12px; font-weight:bold; color:#000; background:rgba(255,255,255,0.7); padding:2px; border-radius:3px;'>{row['station'].replace('역', '')}</div>",
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
        count = len(congestion_df[congestion_df['los_grade'] == grade])
        legend_html += f"""
        <div style="display:flex; align-items:center; margin:3px 0;">
          <span style="display:inline-block; width:16px; height:16px; background:{color}; border:1px solid #999; margin-right:5px;"></span>
          <span>LOS {grade} ({count}회)</span>
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
    los_e_count = len(congestion_df[congestion_df['los_grade'] == 'E'])
    los_f_count = len(congestion_df[congestion_df['los_grade'] == 'F'])
    
    stats_html = f"""
    <div style="position: fixed; top: 10px; right: 10px; width: 250px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:12px; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
    <h4 style="margin: 0 0 10px 0;">과밀 통계</h4>
    <p style="margin: 2px 0;">총 과밀 발생: {total_incidents}회</p>
    <p style="margin: 2px 0;">과밀 역 수: {unique_stations}개</p>
    <p style="margin: 2px 0;">LOS E: {los_e_count}회</p>
    <p style="margin: 2px 0;">LOS F: {los_f_count}회</p>
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
    parser = argparse.ArgumentParser(description='과밀 구간 지도 시각화')
    parser.add_argument('--los-data', default='data/los_calculated.csv', help='LOS 계산 결과 CSV 파일')
    parser.add_argument('--map', default='data/지도.csv', help='지도 CSV 파일 경로')
    parser.add_argument('--output-dir', default='output', help='출력 디렉토리')
    parser.add_argument('--min-los', choices=['E', 'F'], default='E', help='최소 LOS 등급 (E 또는 F)')
    parser.add_argument('--save-png', action='store_true', help='PNG 이미지로도 저장')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== 데이터 로드 ===")
    los_df = read_csv_safe(args.los_data)
    print(f"LOS 데이터: {len(los_df)}개 레코드")
    
    map_df = read_csv_safe(args.map)
    print(f"지도 데이터: {len(map_df)}개 역")
    
    # LOS 등급별 상세 통계 출력
    if len(los_df) > 0:
        print("\n=== LOS 등급별 통계 ===")
        if 'los_grade' in los_df.columns:
            los_counts = los_df['los_grade'].value_counts().sort_index()
            total = len(los_df)
            for grade in ['A', 'B', 'C', 'D', 'E', 'F']:
                count = los_counts.get(grade, 0)
                pct = (count / total * 100) if total > 0 else 0
                print(f"  LOS {grade}: {count:6d}개 ({pct:5.1f}%)")
            
            print(f"\n  총계: {total}개")
            
            # E, F 등급 상세 정보
            e_count = los_counts.get('E', 0)
            f_count = los_counts.get('F', 0)
            ef_total = e_count + f_count
            print(f"\n  LOS E 이상 (E+F): {ef_total}개")
            print(f"    - LOS E: {e_count}개")
            print(f"    - LOS F: {f_count}개")
        else:
            print("경고: 'los_grade' 컬럼이 없습니다.")
    
    # 요일별, 시간대별 통계
    if len(los_df) > 0:
        print("\n=== 요일별 통계 ===")
        if 'weekday_name' in los_df.columns:
            print(los_df['weekday_name'].value_counts())
        
        print("\n=== 시간대별 통계 ===")
        print(los_df['hour'].value_counts().sort_index())
        
        if 'month' in los_df.columns:
            print("\n=== 월별 통계 ===")
            print(los_df['month'].value_counts().sort_index())
    
    # 지도 생성
    print("\n=== 지도 생성 ===")
    output_html = os.path.join(args.output_dir, f'congestion_map_{args.min_los}.html')
    
    map_obj = create_congestion_map(
        los_df=los_df,
        map_df=map_df,
        output_html=output_html,
        min_los_grade=args.min_los,
        title=f"과밀 구간 분석 (LOS {args.min_los} 이상)"
    )
    
    if map_obj is None:
        print("지도 생성 실패")
        return
    
    # PNG로 저장
    if args.save_png:
        print("\n=== 이미지 저장 ===")
        output_png = os.path.join(args.output_dir, f'congestion_map_{args.min_los}.png')
        save_map_as_image(output_html, output_png)
    
    print("\n=== 완료 ===")


if __name__ == '__main__':
    main()

