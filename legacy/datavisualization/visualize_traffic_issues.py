"""
통계.csv 기반 문제점 지도 시각화

통행량 데이터를 분석하여 문제점을 찾고 지도에 시각화하여 이미지로 저장합니다.
"""

import os
import re
import pandas as pd
import folium
from folium.plugins import HeatMap
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse


def read_csv_safe(path: str, encodings: list = None) -> pd.DataFrame:
    """여러 인코딩을 시도하여 CSV 읽기"""
    if encodings is None:
        encodings = ["cp949", "utf-8-sig", "euc-kr", "utf-8", "latin-1"]
    
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    
    raise ValueError(f"CSV 파일을 읽을 수 없습니다: {path}")


def analyze_traffic_issues(stats_df: pd.DataFrame) -> pd.DataFrame:
    """통행량 데이터를 분석하여 문제점 도출"""
    
    print("\n=== 통행량 데이터 분석 ===")
    
    # 시간대 컬럼 찾기
    hour_cols = {}
    for col in stats_df.columns:
        match = re.match(r'(\d{2})-(\d{2})시', str(col))
        if match:
            hour = int(match.group(1))
            if 0 <= hour < 24:
                hour_cols[hour] = col
    
    print(f"시간대 컬럼 {len(hour_cols)}개 찾음")
    
    # 각 역별, 요일별, 시간대별 통행량 분석
    results = []
    
    for _, row in stats_df.iterrows():
        station = str(row['역명']).strip()
        weekday = str(row['요일']).strip()
        direction = str(row['구분']).strip()
        
        # 각 시간대별 통행량 수집
        hourly_traffic = {}
        for hour, col_name in hour_cols.items():
            try:
                traffic = float(row[col_name]) if pd.notna(row[col_name]) else 0.0
                hourly_traffic[hour] = traffic
            except:
                hourly_traffic[hour] = 0.0
        
        # 통계 계산
        traffic_values = list(hourly_traffic.values())
        max_traffic = max(traffic_values) if traffic_values else 0
        avg_traffic = np.mean(traffic_values) if traffic_values else 0
        peak_hour = max(hourly_traffic, key=hourly_traffic.get) if hourly_traffic else None
        
        # 합계만 사용 (승차+하차 통합)
        if direction == '합계':
            results.append({
                'station': station,
                'weekday': weekday,
                'max_traffic': max_traffic,
                'avg_traffic': avg_traffic,
                'peak_hour': peak_hour,
                'traffic_sum': sum(traffic_values)
            })
    
    analysis_df = pd.DataFrame(results)
    
    # 역별 집계 (요일 평균)
    station_stats = analysis_df.groupby('station').agg({
        'max_traffic': 'max',
        'avg_traffic': 'mean',
        'traffic_sum': 'sum',
        'peak_hour': lambda x: x.mode()[0] if len(x.mode()) > 0 else None
    }).reset_index()
    
    # 문제점 점수 계산 (통행량 기반)
    max_traffic_overall = station_stats['max_traffic'].max()
    station_stats['traffic_score'] = (station_stats['max_traffic'] / max_traffic_overall * 100) if max_traffic_overall > 0 else 0
    
    print(f"\n통행량 통계:")
    print(f"  최대 통행량: {max_traffic_overall:.1f}명/시간")
    print(f"  평균 통행량: {station_stats['avg_traffic'].mean():.1f}명/시간")
    print(f"\n상위 5개 역:")
    top5 = station_stats.nlargest(5, 'max_traffic')
    for _, row in top5.iterrows():
        print(f"  {row['station']}: 최대 {row['max_traffic']:.1f}명/시간, 피크시간 {int(row['peak_hour'])}시")
    
    return station_stats


def create_traffic_map(
    station_stats: pd.DataFrame,
    map_df: pd.DataFrame,
    output_html: str,
    output_png: str = None
) -> folium.Map:
    """통행량 문제점을 지도에 시각화"""
    
    # 역별 통행량 데이터와 지도 데이터 병합
    map_df['station_normalized'] = map_df['역명'].str.replace('역', '', regex=False).str.strip()
    station_stats['station_normalized'] = station_stats['station'].str.replace('역', '', regex=False).str.strip()
    
    # 병합
    merged = map_df.merge(
        station_stats,
        left_on='역명',
        right_on='station',
        how='left'
    )
    
    # 매칭되지 않은 경우 정규화된 이름으로 재시도
    unmatched = merged[merged['station'].isna()]
    if len(unmatched) > 0:
        for idx, row in unmatched.iterrows():
            station_normalized = row['station_normalized']
            match = station_stats[station_stats['station_normalized'] == station_normalized]
            if len(match) > 0:
                matched = match.iloc[0]
                merged.loc[idx, 'max_traffic'] = matched['max_traffic']
                merged.loc[idx, 'avg_traffic'] = matched['avg_traffic']
                merged.loc[idx, 'traffic_score'] = matched['traffic_score']
                merged.loc[idx, 'peak_hour'] = matched['peak_hour']
    
    # 지도 중심 계산
    center_lat = merged['Latitude'].mean()
    center_lon = merged['Longitude'].mean()
    
    # 어두운 테마로 변경 (히트맵이 더 돋보이도록)
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB dark_matter'  # 어두운 배경
    )
    
    # 노선 경로 그리기
    route_coords = merged.sort_values('역구성순서')[['Latitude', 'Longitude']].values.tolist()
    folium.PolyLine(
        locations=route_coords,
        color='#555555',
        weight=4,
        opacity=0.7
    ).add_to(m)
    
    # 통행량 데이터 준비 (히트맵용)
    max_traffic = merged['max_traffic'].max() if merged['max_traffic'].notna().any() else 1
    min_traffic = merged['max_traffic'].min() if merged['max_traffic'].notna().any() else 0
    
    # 히트맵 데이터 생성 (통행량이 높을수록 가중치 증가)
    heat_data = []
    for _, row in merged.iterrows():
        if pd.isna(row['Latitude']) or pd.isna(row['Longitude']) or pd.isna(row['max_traffic']):
            continue
        
        # 통행량을 가중치로 사용 (정규화)
        weight = (row['max_traffic'] - min_traffic) / (max_traffic - min_traffic) if max_traffic > min_traffic else 0.5
        weight = max(0.1, min(1.0, weight))  # 0.1 ~ 1.0 범위로 제한
        
        heat_data.append([row['Latitude'], row['Longitude'], weight])
    
    # 히트맵 레이어 추가 (부드러운 그라데이션 효과)
    HeatMap(
        heat_data,
        min_opacity=0.2,
        max_zoom=18,
        radius=35,  # 부드러운 효과를 위해 반경 크게
        blur=20,    # 블러 효과로 자연스러운 그라데이션
        gradient={
            0.0: 'rgba(0, 255, 255, 0)',      # 저통행량: 투명 청록
            0.2: 'rgba(0, 255, 255, 0.3)',   # 
            0.4: 'rgba(0, 255, 255, 0.5)',   # 중간: 청록
            0.6: 'rgba(255, 255, 0, 0.7)',   # 
            0.8: 'rgba(255, 100, 0, 0.8)',   # 높음: 주황
            1.0: 'rgba(255, 0, 0, 1)'        # 매우높음: 빨강 (밝게)
        },
        overlay=True,
        control=True
    ).add_to(m)
    
    # 각 역에 부드러운 글로우 효과 마커 추가
    for _, row in merged.iterrows():
        if pd.isna(row['Latitude']) or pd.isna(row['Longitude']):
            continue
        
        station_name = row['역명']
        max_traffic_val = row.get('max_traffic', 0) if pd.notna(row.get('max_traffic')) else 0
        avg_traffic_val = row.get('avg_traffic', 0) if pd.notna(row.get('avg_traffic')) else 0
        peak_hour = int(row.get('peak_hour', 0)) if pd.notna(row.get('peak_hour')) else None
        
        # 통행량에 따른 색상 (RGB로 부드러운 그라데이션)
        ratio = (max_traffic_val - min_traffic) / (max_traffic - min_traffic) if max_traffic > min_traffic else 0
        ratio = max(0, min(1, ratio))
        
        # 색상 계산 (청록 → 노랑 → 주황 → 빨강)
        if ratio < 0.33:
            # 청록 계열
            r, g, b = int(0 + ratio * 200), int(255 - ratio * 100), int(255 - ratio * 200)
        elif ratio < 0.66:
            # 노랑 → 주황
            r, g, b = int(200 + (ratio - 0.33) * 100), int(255 - (ratio - 0.33) * 155), 0
        else:
            # 주황 → 빨강
            r, g, b = int(255), int(100 - (ratio - 0.66) * 100), 0
        
        color_hex = f'#{r:02x}{g:02x}{b:02x}'
        color_rgba = f'rgba({r}, {g}, {b}, 0.6)'
        
        # 글로우 효과가 있는 마커 (부드러운 원)
        size = 8 + (ratio * 20)  # 8 ~ 28px
        
        # CSS 글로우 효과가 있는 커스텀 아이콘
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    width: {size}px;
                    height: {size}px;
                    background: radial-gradient(circle, {color_rgba} 0%, {color_hex} 100%);
                    border-radius: 50%;
                    box-shadow: 0 0 {size*2}px {color_hex}, 0 0 {size*3}px {color_hex};
                    border: 2px solid rgba(255, 255, 255, 0.8);
                "></div>
                """,
                icon_size=(int(size*2), int(size*2)),
                icon_anchor=(int(size), int(size)),
                class_name='glow-marker'
            ),
            popup=f"<b>{station_name}</b><br>최대: {max_traffic_val:.1f}명/시간<br>평균: {avg_traffic_val:.1f}명/시간" + (f"<br>피크: {peak_hour}시" if peak_hour else ""),
            tooltip=f"{station_name.replace('역', '')} ({max_traffic_val:.0f}명/시간)"
        ).add_to(m)
        
        # 역명 레이블 (반투명 배경)
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            icon=folium.DivIcon(
                html=f"<div style='font-size:11px; font-weight:bold; color:#fff; text-shadow: 1px 1px 2px rgba(0,0,0,0.8); background:rgba(0,0,0,0.5); padding:2px 6px; border-radius:4px;'>{station_name.replace('역', '')}</div>",
                icon_size=(80, 20),
                icon_anchor=(40, 10),
                class_name='station-label'
            )
        ).add_to(m)
    
    # 범례 추가 (히트맵 스타일)
    legend_html = f"""
    <div style="position: fixed; bottom: 50px; left: 50px; width: 250px;
                background: rgba(0, 0, 0, 0.7); border:2px solid rgba(255,255,255,0.3); z-index:9999;
                font-size:12px; padding: 12px; border-radius: 8px; color: white; backdrop-filter: blur(10px);">
    <h4 style="margin: 0 0 10px 0; color: white;">통행량 밀집도</h4>
    <p style="margin: 2px 0; font-size: 11px; color: #ddd;">히트맵 + 글로우 효과</p>
    <div style="margin-top: 10px;">
        <div style="display:flex; align-items:center; margin:5px 0;">
            <span style="display:inline-block; width:20px; height:20px; background: radial-gradient(circle, rgba(0,255,255,0.6) 0%, #00ffff 100%); border-radius:50%; box-shadow: 0 0 15px #00ffff; margin-right:10px; border: 2px solid rgba(255,255,255,0.8);"></span>
            <span>낮음</span>
        </div>
        <div style="display:flex; align-items:center; margin:5px 0;">
            <span style="display:inline-block; width:20px; height:20px; background: radial-gradient(circle, rgba(255,255,0,0.6) 0%, #ffff00 100%); border-radius:50%; box-shadow: 0 0 15px #ffff00; margin-right:10px; border: 2px solid rgba(255,255,255,0.8);"></span>
            <span>보통</span>
        </div>
        <div style="display:flex; align-items:center; margin:5px 0;">
            <span style="display:inline-block; width:20px; height:20px; background: radial-gradient(circle, rgba(255,150,0,0.6) 0%, #ff9600 100%); border-radius:50%; box-shadow: 0 0 15px #ff9600; margin-right:10px; border: 2px solid rgba(255,255,255,0.8);"></span>
            <span>높음</span>
        </div>
        <div style="display:flex; align-items:center; margin:5px 0;">
            <span style="display:inline-block; width:20px; height:20px; background: radial-gradient(circle, rgba(255,0,0,0.6) 0%, #ff0000 100%); border-radius:50%; box-shadow: 0 0 15px #ff0000; margin-right:10px; border: 2px solid rgba(255,255,255,0.8);"></span>
            <span>매우 높음</span>
        </div>
    </div>
    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.3);">
        <p style="margin: 2px 0; font-size: 10px; color: #ccc;">히트맵은 통행량 밀집도를 나타냅니다</p>
    </div>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # 통계 정보 추가
    top_stations = merged.nlargest(5, 'max_traffic')
    stats_html = f"""
    <div style="position: fixed; top: 10px; right: 10px; width: 280px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:12px; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
    <h4 style="margin: 0 0 10px 0;">통행량 분석 결과</h4>
    <p style="margin: 2px 0; font-size: 11px;"><b>최대 통행량:</b> {max_traffic:.1f}명/시간</p>
    <p style="margin: 2px 0; font-size: 11px;"><b>평균 통행량:</b> {merged['max_traffic'].mean():.1f}명/시간</p>
    <div style="margin-top: 8px; border-top: 1px solid #ddd; padding-top: 8px;">
        <p style="margin: 2px 0; font-size: 11px; font-weight:bold;">상위 5개 역:</p>
    """
    
    for i, (_, row) in enumerate(top_stations.iterrows(), 1):
        peak_hour = int(row.get('peak_hour', 0)) if pd.notna(row.get('peak_hour')) else '-'
        stats_html += f"<p style='margin: 2px 0; font-size: 10px;'>{i}. {row['역명'].replace('역', '')} ({row['max_traffic']:.0f}명, {peak_hour}시)</p>"
    
    stats_html += """
    </div>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(stats_html))
    
    m.save(output_html)
    print(f"지도 저장 완료: {output_html}")
    
    # PNG로 저장
    if output_png:
        save_map_as_image(output_html, output_png)
    
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
            service = Service()
        
        options = Options()
        options.add_argument('--headless=new')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(f'file://{os.path.abspath(html_path)}')
        
        import time
        time.sleep(3)  # 지도 로딩 대기
        
        driver.save_screenshot(output_png)
        driver.quit()
        
        print(f"이미지 저장 완료: {output_png}")
    except Exception as e:
        print(f"이미지 저장 실패 (Selenium 필요): {e}")
        print(f"HTML 파일은 생성되었습니다: {html_path}")


def main():
    parser = argparse.ArgumentParser(description='통행량 문제점 지도 시각화')
    parser.add_argument('--stats', default='data/통계.csv', help='통계 CSV 파일 경로')
    parser.add_argument('--map', default='data/지도.csv', help='지도 CSV 파일 경로')
    parser.add_argument('--output-dir', default='output', help='출력 디렉토리')
    parser.add_argument('--save-png', action='store_true', help='PNG 이미지로도 저장')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== 데이터 로드 ===")
    stats_df = read_csv_safe(args.stats)
    print(f"통계 데이터: {len(stats_df)}개 레코드")
    
    map_df = read_csv_safe(args.map)
    print(f"지도 데이터: {len(map_df)}개 역")
    
    # 통행량 문제점 분석
    station_stats = analyze_traffic_issues(stats_df)
    
    # 지도 생성
    print("\n=== 지도 생성 ===")
    output_html = os.path.join(args.output_dir, 'traffic_issues_map.html')
    output_png = os.path.join(args.output_dir, 'traffic_issues_map.png') if args.save_png else None
    
    create_traffic_map(
        station_stats=station_stats,
        map_df=map_df,
        output_html=output_html,
        output_png=output_png
    )
    
    print("\n=== 완료 ===")


if __name__ == '__main__':
    main()

