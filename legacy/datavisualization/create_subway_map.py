import os
import json
import re
import argparse
from typing import Dict, List, Tuple

import pandas as pd
import folium

def _read_csv_any(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8", "latin-1"]
    last = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last = e
    raise RuntimeError(f"CSV 로드 실패: {path}\n{last}")


def create_subway_map(
    csv_path: str,
    traffic_path: str | None = None,
    output_path: str = 'subway_map.html',
    default_platform_ratio: float = 0.35,
    default_corridor_ratio: float = 0.20,
    default_board_wait_min: float = 5.0,
    default_transit_sec: float = 40.0,
):
    """
    지도.csv 파일을 기반으로 Folium 지도 생성
    - 위경도 정보
    - 역명 표시
    - 역순서 반영
    - 연면적 기반 노드 크기 정규화
    - 날짜/시간별 통행량 시각화
    """
    # CSV 파일 읽기
    df = _read_csv_any(csv_path)
    
    # 통계/통행량 데이터 읽기
    traffic_df = None
    if traffic_path and os.path.exists(traffic_path):
        try:
            traffic_df = _read_csv_any(traffic_path)
            print(f"통계/통행량 데이터 {len(traffic_df)}개 로드 완료")
        except Exception as e:
            print(f"통계/통행량 데이터 로드 실패: {e}")
            traffic_df = None
    
    # 역순서로 정렬
    df = df.sort_values('역구성순서')
    
    # 면적 정보 파악 및 라벨 마커 크기 결정
    total_area_col = None
    for c in df.columns:
        if '연면적' in c:
            total_area_col = c
            break

    platform_area_col = None
    for c in df.columns:
        if '승강장' in c and '면적' in c:
            platform_area_col = c
            break

    corridor_area_col = None
    for c in df.columns:
        if ('통로' in c or '환승' in c) and '면적' in c:
            corridor_area_col = c
            break

    if platform_area_col is None and total_area_col is not None:
        df['platform_area'] = df[total_area_col] * default_platform_ratio
    elif platform_area_col is not None:
        df['platform_area'] = pd.to_numeric(df[platform_area_col], errors='coerce')
    else:
        df['platform_area'] = pd.NA

    if corridor_area_col is None and total_area_col is not None:
        df['corridor_area'] = df[total_area_col] * default_corridor_ratio
    elif corridor_area_col is not None:
        df['corridor_area'] = pd.to_numeric(df[corridor_area_col], errors='coerce')
    else:
        df['corridor_area'] = pd.NA

    base_for_label = None
    if total_area_col is not None:
        base_for_label = pd.to_numeric(df[total_area_col], errors='coerce')
    elif 'platform_area' in df.columns:
        base_for_label = pd.to_numeric(df['platform_area'], errors='coerce')
    else:
        base_for_label = pd.Series([1000.0] * len(df))
    min_area = base_for_label.min()
    max_area = base_for_label.max()
    if pd.isna(min_area) or pd.isna(max_area) or min_area == max_area:
        df['small_marker_size'] = 28
    else:
        df['small_marker_size'] = 20 + ((base_for_label - min_area) / (max_area - min_area) * 15)
    
    # 큰 원형 마커의 기본 크기 (통행량에 따라 변화)
    df['base_traffic_radius'] = 15  # 기본 반경
    
    # 지도 중심 좌표 계산
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    
    # Folium 지도 생성
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron'
    )
    
    # 노선 경로를 위한 좌표 리스트
    route_coords = []
    
    # 역 정보를 JSON으로 저장 (JavaScript에서 사용)
    stations_data = []
    for idx, row in df.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        station_name = row['역명']
        station_order = int(row['역구성순서'])
        total_area = float(row.get(total_area_col, 0)) if total_area_col else 0.0
        platform_area = float(row.get('platform_area', 0)) if pd.notna(row.get('platform_area', pd.NA)) else 0.0
        corridor_area = float(row.get('corridor_area', 0)) if pd.notna(row.get('corridor_area', pd.NA)) else 0.0
        
        route_coords.append([lat, lon])
        
        stations_data.append({
            'name': station_name,
            'order': station_order,
            'lat': lat,
            'lon': lon,
            'small_marker_size': float(row['small_marker_size']),
            'base_radius': float(row['base_traffic_radius']),
            'total_area': float(total_area),
            'platform_area': float(platform_area),
            'corridor_area': float(corridor_area)
        })
    
    # 통행량 데이터를 JSON으로 변환
    traffic_data_json = None
    if traffic_df is not None:
        # 날짜/역명/구분/시간대 컬럼 유추 후 구조화
        date_cols = [c for c in traffic_df.columns if any(k in c for k in ['영업월', '날짜', '일자', 'DATE', 'date'])]
        date_col = date_cols[0] if date_cols else traffic_df.columns[0]
        
        # 역명 컬럼 찾기 (여러 패턴 시도)
        name_cols = [c for c in traffic_df.columns if any(k in c for k in ['역명', '역', 'station', 'name'])]
        if not name_cols and len(traffic_df.columns) > 3:
            # 시간대 컬럼이 아닌 것을 찾기 (3번째나 4번째 컬럼이 역명일 수 있음)
            time_cols = [c for c in traffic_df.columns if any(x in c for x in ['시', 'hour', 'H'])]
            name_cols = [c for c in traffic_df.columns if c not in time_cols and c != date_col and len(traffic_df.columns) > traffic_df.columns.get_loc(c)]
        name_col = name_cols[0] if name_cols else (traffic_df.columns[3] if len(traffic_df.columns) > 3 else traffic_df.columns[1])
        
        dir_cols = [c for c in traffic_df.columns if any(k in c for k in ['승하차', '구분', 'direction', 'type', '구분'])]
        dir_col = dir_cols[0] if dir_cols else (traffic_df.columns[4] if len(traffic_df.columns) > 4 else (traffic_df.columns[2] if len(traffic_df.columns) > 2 else None))

        # 시간대 컬럼 찾기: "00-01시", "03-04시" 형식도 처리
        def find_hourly_columns():
            """시간대 컬럼을 찾아서 {hour: column_name} 딕셔너리 반환"""
            hour_to_col = {}
            
            # 먼저 표준 형식 찾기: "00시 통행량", "01시 통행량" 등
            for h in range(24):
                for pattern in [f"{h:02d}시 통행량", f"{h:02d}시", f"{h}시 통행량", f"H{h:02d}"]:
                    if pattern in traffic_df.columns:
                        hour_to_col[h] = pattern
                        break
            
            # 범위 형식 찾기: "00-01시", "03-04시" 등 (시작 시간을 해당 시간으로 매핑)
            # 표준 형식을 찾지 못했거나 추가로 범위 형식도 찾기
            for col in traffic_df.columns:
                # "00-01시" 형식: 시작 시간(00)을 0시로 매핑
                match = re.match(r'(\d{2})-(\d{2})시', col)
                if match:
                    start_hour = int(match.group(1))
                    if 0 <= start_hour < 24 and start_hour not in hour_to_col:
                        hour_to_col[start_hour] = col
                
                # "23-00시" 같은 경우는 23시로 매핑
                if re.match(r'23-00시', col) and 23 not in hour_to_col:
                    hour_to_col[23] = col
            
            return hour_to_col
        
        hour_to_column = find_hourly_columns()
        
        # 디버깅: 찾은 컬럼 정보 출력
        if hour_to_column:
            print(f"시간대 컬럼 매핑: {len(hour_to_column)}개 시간대 찾음")
            print(f"  예시: {dict(list(hour_to_column.items())[:3])}")
        else:
            print("경고: 시간대 컬럼을 찾을 수 없습니다. 컬럼명 확인 필요.")
            print(f"  사용 가능한 컬럼: {list(traffic_df.columns[:10])}")
        
        print(f"날짜 컬럼: {date_col}, 역명 컬럼: {name_col}, 구분 컬럼: {dir_col}")
        
        traffic_records = []
        for _, row in traffic_df.iterrows():
            date_str = str(row.get(date_col, ''))
            station_name = str(row.get(name_col, ''))
            direction = str(row.get(dir_col, '합계')) if dir_col is not None else '합계'
            hourly_data: Dict[int, int] = {}
            for hour in range(24):
                val = 0
                # 시간대에 해당하는 컬럼 찾기
                if hour in hour_to_column:
                    col_name = hour_to_column[hour]
                    v = row.get(col_name, 0)
                    try:
                        val = int(float(v)) if pd.notna(v) else 0
                    except (ValueError, TypeError):
                        val = 0
                hourly_data[hour] = val
            traffic_records.append({
                'date': date_str,
                'station': station_name,
                'direction': direction,
                'hourly': hourly_data
            })

        traffic_data_json = json.dumps(traffic_records, ensure_ascii=False)
        print(f"통계/통행량 레코드 {len(traffic_records)}개 준비 완료")
        if traffic_records:
            print(f"  예시 레코드: 날짜={traffic_records[0]['date']}, 역명={traffic_records[0]['station']}, 구분={traffic_records[0]['direction']}")
    
    # 각 역에 원형 마커 추가
    for station in stations_data:
        station_name = station['name']
        station_order = station['order']
        lat = station['lat']
        lon = station['lon']
        base_radius = station['base_radius']
        small_marker_size = station['small_marker_size']
        
        # 큰 원형 마커 색상 (진한 회색)
        node_color = '#555555'  # 진한 회색
        
        # 큰 원형 마커 생성 (통행량에 따라 크기 변화)
        circle_marker = folium.CircleMarker(
            location=[lat, lon],
            radius=base_radius,  # 기본 크기, JavaScript에서 통행량에 따라 조정
            popup=f"{station_order}. {station_name}",
            tooltip=f"{station_order}. {station_name}",
            color=node_color,
            fillColor=node_color,
            fillOpacity=0.6,
            weight=2,
            id=f"traffic_marker_{station_name}"
        )
        circle_marker.add_to(m)
        
        # 작은 원형 마커 생성 (연면적 기반 크기, 역 이름 표시)
        # 배경색 통일, 글자 검은색
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    background-color: white;
                    border: 2px solid #333333;
                    border-radius: 50%;
                    width: {small_marker_size}px;
                    height: {small_marker_size}px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    font-size: {int(small_marker_size * 0.35)}px;
                    color: #000000;
                    pointer-events: none;
                    text-align: center;
                    line-height: 1;
                    padding: 2px;
                ">
                    {station_name.replace('역', '')}
                </div>
                """,
                icon_size=(int(small_marker_size), int(small_marker_size)),
                icon_anchor=(int(small_marker_size/2), int(small_marker_size/2)),
                class_name='station-name'
            )
        ).add_to(m)
    
    # 노선 경로 선 그리기
    folium.PolyLine(
        locations=route_coords,
        color='#555555',  # 진한 회색
        weight=4,
        opacity=0.7,
        popup='대전교통공사 1호선'
    ).add_to(m)
    
    # 통행량 시각화를 위한 JavaScript 코드 추가
    stations_json = json.dumps(stations_data, ensure_ascii=False)
    
    # HTML 컨트롤 및 JavaScript 추가
    control_html = f"""
    <div id="traffic-controls" style="position: fixed; top: 10px; right: 10px; 
                width: 280px; background-color: white; border:2px solid grey; 
                z-index:9999; font-size:12px; padding: 12px; border-radius: 5px; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.2); font-family: Arial;">
        <h4 style="margin: 0 0 10px 0; font-size: 14px; font-weight: bold;">혼잡도/통행량 조회</h4>

        <div style="margin-bottom: 8px;">
            <label><strong>평가 모드:</strong></label><br>
            <input type="radio" name="mode" value="waiting" id="mode_waiting" checked>
            <label for="mode_waiting">대기공간(승강장)</label><br>
            <input type="radio" name="mode" value="walking" id="mode_walking">
            <label for="mode_walking">보행공간(통로)</label>
        </div>
        
        <div style="margin-bottom: 10px;">
            <label><strong>구분:</strong></label><br>
            <input type="radio" name="direction" value="승차" id="radio_boarding" checked>
            <label for="radio_boarding">승차 (파란색)</label><br>
            <input type="radio" name="direction" value="하차" id="radio_alighting">
            <label for="radio_alighting">하차 (주황색)</label><br>
            <input type="radio" name="direction" value="합계" id="radio_total">
            <label for="radio_total">합계 (초록색)</label>
        </div>
        
        <div style="margin-bottom: 10px;">
            <label><strong>날짜:</strong></label><br>
            <select id="date-select" style="width: 100%; padding: 4px;">
            </select>
        </div>
        
        <div style="margin-bottom: 10px;">
            <label><strong>시간:</strong> <span id="hour-display">00시</span></label><br>
            <input type="range" id="hour-slider" min="0" max="23" value="0" 
                   style="width: 100%;" step="1">
        </div>

        <div style="margin-bottom: 8px;">
            <label><strong>표시 옵션:</strong></label><br>
            <input type="checkbox" id="only-ef" /> <label for="only-ef">E/F 과밀역만 강조</label>
        </div>

        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd;">
            <div id="traffic-info" style="font-size: 11px; color: #666;">
                날짜와 시간을 선택하세요
            </div>
            <div id="top-stations" style="margin-top:6px; font-size: 11px; color: #333;"></div>
        </div>
    </div>
    """
    
    # JavaScript 코드
    js_code = f"""
    <script>
    // 역 정보 데이터
    const stationsData = {stations_json};
    
    // 통행량 데이터
    const trafficData = {traffic_data_json if traffic_data_json else '[]'};
    
    // 고정 파라미터(조정 UI 제거)
    const WAIT_MIN = {default_board_wait_min};
    const TRANSIT_SEC = {default_transit_sec};
    const PLATFORM_RATIO = {default_platform_ratio};
    const CORRIDOR_RATIO = {default_corridor_ratio};

    // 역별 면적 보정 (연면적만 있을 경우 비율 적용)
    function getPlatformArea(station) {{
        if (station.platform_area && station.platform_area > 0) return station.platform_area;
        return (station.total_area || 0) * PLATFORM_RATIO;
    }}
    function getCorridorArea(station) {{
        if (station.corridor_area && station.corridor_area > 0) return station.corridor_area;
        return (station.total_area || 0) * CORRIDOR_RATIO;
    }}

    // 날짜 목록 추출
    const dates = [...new Set(trafficData.map(d => d.date))].sort();
    
    // 패널 초기화 함수: 날짜 채우기, 이벤트 바인딩, 초기 업데이트
    function initPanel() {{
        const dateSelect = document.getElementById('date-select');
        if (!dateSelect) {{
            setTimeout(initPanel, 100);
            return;
        }}

        // 날짜 옵션 채우기 (중복 방지)
        if (dateSelect.options.length === 0) {{
            dates.forEach(date => {{
                const option = document.createElement('option');
                option.value = date;
                option.textContent = date;
                dateSelect.appendChild(option);
            }});
        }}

        // 이벤트 리스너 바인딩(중복 방지 위해 한 번만)
        if (!dateSelect._bound) {{
            document.querySelectorAll('input[name="direction"]').forEach(radio => {{
                radio.addEventListener('change', updateTraffic);
            }});
            document.querySelectorAll('input[name="mode"]').forEach(radio => {{
                radio.addEventListener('change', updateTraffic);
            }});
            document.getElementById('date-select').addEventListener('change', updateTraffic);
            document.getElementById('hour-slider').addEventListener('input', updateTraffic);
            document.getElementById('only-ef').addEventListener('change', updateTraffic);
            dateSelect._bound = true;
        }}

        // 마커 찾기 이후 갱신 예약
        setTimeout(findCircleMarkers, 500);

        // 초기 업데이트
        if (dates.length > 0) {{
            updateTraffic();
        }}
    }}
    
    // 통행량 최대값 계산 (색상/정규화 참고)
    let maxTraffic = 0;
    trafficData.forEach(record => {{
        Object.values(record.hourly).forEach(val => {{
            if (val > maxTraffic) maxTraffic = val;
        }});
    }});

    // LOS 색상
    const LOS_COLORS = {{ 'A':'#2ECC71','B':'#8BC34A','C':'#FFC107','D':'#FF9800','E':'#FF5722','F':'#8B0000' }};
    function colorByLOS(grade) {{ return LOS_COLORS[grade] || '#CCCCCC'; }}

    // 밀도->LOS (대기공간)
    function losWaiting(areaPerPerson) {{
        if (!isFinite(areaPerPerson) || areaPerPerson <= 0) return 'A';
        if (areaPerPerson >= 1.18) return 'A';
        if (areaPerPerson >= 0.78) return 'B';
        if (areaPerPerson >= 0.54) return 'C';
        if (areaPerPerson >= 0.34) return 'D';
        if (areaPerPerson >= 0.23) return 'E';
        return 'F';
    }}

    // 밀도->LOS (보행공간) - 인/㎡ 기준
    function losWalking(density) {{
        if (!isFinite(density) || density < 0) return 'A';
        if (density <= 0.31) return 'A';
        if (density <= 0.43) return 'B';
        if (density <= 0.71) return 'C';
        if (density <= 1.06) return 'D';
        if (density <= 1.79) return 'E';
        return 'F';
    }}
    
    // 각 역별 일간 최대/최소 통행량 계산
    function getDailyTrafficRange(stationName, date, direction) {{
        let values = [];
        
        if (direction === '합계') {{
            for (let hour = 0; hour < 24; hour++) {{
                const boarding = trafficData.find(r => 
                    r.date === date && r.station === stationName && r.direction === '승차'
                );
                const alighting = trafficData.find(r => 
                    r.date === date && r.station === stationName && r.direction === '하차'
                );
                const boardVal = boarding ? (boarding.hourly[hour] || 0) : 0;
                const alightVal = alighting ? (alighting.hourly[hour] || 0) : 0;
                values.push(boardVal + alightVal);
            }}
        }} else {{
            const records = trafficData.filter(r => 
                r.date === date && r.station === stationName && r.direction === direction
            );
            if (records.length > 0) {{
                const record = records[0];
                for (let hour = 0; hour < 24; hour++) {{
                    values.push(record.hourly[hour] || 0);
                }}
            }}
        }}
        
        if (values.length === 0) return {{ min: 0, max: 1 }};
        return {{
            min: Math.min(...values),
            max: Math.max(...values)
        }};
    }}
    
    // 크기 계산 함수 (통행량에 따라, 일간 최대/최소 기준 정규화)
    function getRadius(value, minValue, maxValue, baseRadius) {{
        if (maxValue === 0 || maxValue === minValue) return baseRadius;
        const ratio = (value - minValue) / (maxValue - minValue);
        // 기본 크기(baseRadius)에서 최대 3배까지 증가
        return baseRadius + (ratio * baseRadius * 2);
    }}
    
    // CircleMarker 레이어 저장 (마커 업데이트용)
    const markerLayers = {{}};
    
    // Folium 지도 객체 찾기
    function getMap() {{
        // Folium이 생성한 지도 변수 찾기
        for (let key in window) {{
            if (window[key] instanceof L.Map) {{
                return window[key];
            }}
        }}
        // 대안: 모든 Leaflet map 객체 찾기
        const mapDiv = document.querySelector('.folium-map');
        if (mapDiv) {{
            const mapId = mapDiv.id;
            return window[mapId];
        }}
        return null;
    }}
    
    // 지도 초기화 후 CircleMarker 찾기
    function findCircleMarkers() {{
        const map = getMap();
        if (!map) {{
            setTimeout(findCircleMarkers, 100);
            return;
        }}
        
        // Leaflet 레이어에서 CircleMarker 찾기
        map.eachLayer(function(layer) {{
            if (layer instanceof L.CircleMarker) {{
                const lat = layer.getLatLng().lat;
                const lon = layer.getLatLng().lng;
                
                // 역 정보와 매칭
                stationsData.forEach(station => {{
                    if (Math.abs(station.lat - lat) < 0.001 && Math.abs(station.lon - lon) < 0.001) {{
                        markerLayers[station.name] = layer;
                    }}
                }});
            }}
        }});
        // 매칭 완료 후 한 번 갱신
        try {{ updateTraffic(); }} catch (e) {{ console.warn(e); }}
    }}
    
    // 역별 통행량/혼잡도 업데이트
    function updateTraffic() {{
        if (trafficData.length === 0) return;

        const map = getMap();
        if (!map) {{
            setTimeout(updateTraffic, 100);
            return;
        }}

        const mode = document.querySelector('input[name="mode"]:checked').value; // waiting|walking
        const direction = document.querySelector('input[name="direction"]:checked').value;
        const date = document.getElementById('date-select').value;
        const hour = parseInt(document.getElementById('hour-slider').value);
        const waitMin = WAIT_MIN;
        const transitSec = TRANSIT_SEC;
        const onlyEF = document.getElementById('only-ef').checked;

        document.getElementById('hour-display').textContent = `${{hour.toString().padStart(2, '0')}}시`;

        let totalTraffic = 0;
        let topList = [];

        // 각 역 업데이트
        stationsData.forEach(station => {{
            const recBoard = trafficData.find(r => r.date === date && r.station === station.name && r.direction === '승차');
            const recAlight = trafficData.find(r => r.date === date && r.station === station.name && r.direction === '하차');
            const boardVal = recBoard ? (recBoard.hourly[hour] || 0) : 0;
            const alightVal = recAlight ? (recAlight.hourly[hour] || 0) : 0;

            let trafficValue = 0;
            if (direction === '합계') trafficValue = boardVal + alightVal;
            else if (direction === '승차') trafficValue = boardVal; else trafficValue = alightVal;
            totalTraffic += trafficValue;

            // LOS 계산
            let density = 0; // 인/㎡
            let areaPerPerson = Infinity; // ㎡/인 (waiting)
            if (mode === 'waiting') {{
                const occ = boardVal * (waitMin / 60.0); // 평균 플랫폼 체류 인원
                const area = getPlatformArea(station);
                if (area > 0) {{
                    density = occ / area;
                    areaPerPerson = area / Math.max(occ, 1e-9);
                }}
            }} else {{
                const flow = boardVal + alightVal; // 통로 통과 인원
                const occ = flow * (transitSec / 3600.0); // 통로 평균 체류 인원
                const area = getCorridorArea(station);
                if (area > 0) density = occ / area;
            }}
            const grade = (mode === 'waiting') ? losWaiting(areaPerPerson) : losWalking(density);
            const color = colorByLOS(grade);

            const dailyRange = getDailyTrafficRange(station.name, date, (direction === '합계' ? '승차' : direction));
            const markerLayer = markerLayers[station.name];
            if (markerLayer) {{
                const radius = getRadius(trafficValue, dailyRange.min, dailyRange.max, station.base_radius);
                // 색상/불투명도 업데이트
                markerLayer.setStyle({{ fillColor: color, color: color, fillOpacity: 0.7 }});
                // 반경은 setRadius 사용
                if (typeof markerLayer.setRadius === 'function') {{
                    markerLayer.setRadius(radius);
                }}
                if (onlyEF && !(grade === 'E' || grade === 'F')) {{
                    markerLayer.setStyle({{ fillOpacity: 0.15, opacity: 0.2 }});
                }}
            }}
            topList.push({{ name: station.name, grade, density, traffic: trafficValue }});
        }});

        // 정보 패널
        document.getElementById('traffic-info').innerHTML = `
            <strong>총 통행량:</strong> ${{totalTraffic.toLocaleString()}}명<br>
            <strong>평균 통행량:</strong> ${{Math.round(totalTraffic / stationsData.length).toLocaleString()}}명
        `;
        topList.sort((a,b)=> (b.grade.localeCompare(a.grade)) || (b.density - a.density));
        const top = topList.slice(0, 5).map((t,i)=> `${{i+1}}. ${{t.name}} <span style=\"color:${{colorByLOS(t.grade)}}\">${{t.grade}}</span>`).join('<br>');
        const target = document.getElementById('top-stations');
        if (target) target.innerHTML = `<strong>TOP 혼잡역:</strong><br>${{top}}`;
    }}
    
    // 페이지 로드 후 패널 초기화
    window.addEventListener('load', function() {{
        initPanel();
    }});
    </script>
    """
    
    # HTML에 컨트롤 및 JavaScript 추가
    m.get_root().html.add_child(folium.Element(control_html))
    m.get_root().html.add_child(folium.Element(js_code))
    
    # 범례 추가
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 220px; height: 170px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 8px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
    <h4 style="margin: 3px 0; font-size: 14px; font-weight: bold;">범례</h4>
    <p style="margin: 2px 0; font-size: 11px;">원 크기: 해당시간 통행량 정규화</p>
    <p style="margin: 2px 0; font-size: 11px;">색상: LOS 등급(A~F)</p>
    <div style="display:flex; gap:6px; align-items:center; margin:4px 0;">
      <span style="display:inline-block; width:14px; height:14px; background:#2ECC71; border:1px solid #999"></span>&nbsp;A
      <span style="display:inline-block; width:14px; height:14px; background:#8BC34A; border:1px solid #999"></span>&nbsp;B
      <span style="display:inline-block; width:14px; height:14px; background:#FFC107; border:1px solid #999"></span>&nbsp;C
      <span style="display:inline-block; width:14px; height:14px; background:#FF9800; border:1px solid #999"></span>&nbsp;D
      <span style="display:inline-block; width:14px; height:14px; background:#FF5722; border:1px solid #999"></span>&nbsp;E
      <span style="display:inline-block; width:14px; height:14px; background:#8B0000; border:1px solid #999"></span>&nbsp;F
    </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # 지도 저장
    m.save(output_path)
    print(f"지도가 '{output_path}'에 저장되었습니다.")
    print(f"총 {len(df)}개의 역이 표시되었습니다.")
    if traffic_df is not None:
        print(f"통행량 데이터를 포함한 인터랙티브 지도가 생성되었습니다.")
    
    return m


# ====== Static export helpers (no JS interaction required) ======

def _parse_traffic_df(traffic_df: pd.DataFrame) -> Tuple[List[Dict], Dict[Tuple[str, str, str], Dict[int, int]]]:
    """Parse stats/traffic dataframe → (records list, index dict).
    index key = (date, station, direction), value = {hour:int -> count:int}
    """
    # 날짜/역명/구분/시간대 컬럼 유추 (create_subway_map과 동일한 로직)
    date_cols = [c for c in traffic_df.columns if any(k in c for k in ['영업월', '날짜', '일자', 'DATE', 'date'])]
    date_col = date_cols[0] if date_cols else traffic_df.columns[0]
    
    # 역명 컬럼 찾기 (여러 패턴 시도)
    name_cols = [c for c in traffic_df.columns if any(k in c for k in ['역명', '역', 'station', 'name'])]
    if not name_cols and len(traffic_df.columns) > 3:
        # 시간대 컬럼이 아닌 것을 찾기 (3번째나 4번째 컬럼이 역명일 수 있음)
        time_cols = [c for c in traffic_df.columns if any(x in c for x in ['시', 'hour', 'H'])]
        name_cols = [c for c in traffic_df.columns if c not in time_cols and c != date_col and len(traffic_df.columns) > traffic_df.columns.get_loc(c)]
    name_col = name_cols[0] if name_cols else (traffic_df.columns[3] if len(traffic_df.columns) > 3 else traffic_df.columns[1])
    
    dir_cols = [c for c in traffic_df.columns if any(k in c for k in ['승하차', '구분', 'direction', 'type', '구분'])]
    dir_col = dir_cols[0] if dir_cols else (traffic_df.columns[4] if len(traffic_df.columns) > 4 else (traffic_df.columns[2] if len(traffic_df.columns) > 2 else None))

    # 시간대 컬럼 찾기: "00-01시", "03-04시" 형식도 처리
    def find_hourly_columns_static():
        """시간대 컬럼을 찾아서 {hour: column_name} 딕셔너리 반환"""
        hour_to_col = {}
        
        # 먼저 표준 형식 찾기: "00시 통행량", "01시 통행량" 등
        for h in range(24):
            for pattern in [f"{h:02d}시 통행량", f"{h:02d}시", f"{h}시 통행량", f"H{h:02d}"]:
                if pattern in traffic_df.columns:
                    hour_to_col[h] = pattern
                    break
        
        # 범위 형식 찾기: "00-01시", "03-04시" 등 (시작 시간을 해당 시간으로 매핑)
        # 표준 형식을 찾지 못했거나 추가로 범위 형식도 찾기
        for col in traffic_df.columns:
            # "00-01시" 형식: 시작 시간(00)을 0시로 매핑
            match = re.match(r'(\d{2})-(\d{2})시', col)
            if match:
                start_hour = int(match.group(1))
                if 0 <= start_hour < 24 and start_hour not in hour_to_col:
                    hour_to_col[start_hour] = col
            
            # "23-00시" 같은 경우는 23시로 매핑
            if re.match(r'23-00시', col) and 23 not in hour_to_col:
                hour_to_col[23] = col
        
        return hour_to_col
    
    hour_to_column = find_hourly_columns_static()

    records: List[Dict] = []
    index: Dict[Tuple[str, str, str], Dict[int, int]] = {}
    for _, row in traffic_df.iterrows():
        date_str = str(row.get(date_col, ''))
        station_name = str(row.get(name_col, ''))
        direction = str(row.get(dir_col, '합계')) if dir_col is not None else '합계'
        hourly_data: Dict[int, int] = {}
        for hour in range(24):
            val = 0
            # 시간대에 해당하는 컬럼 찾기
            if hour in hour_to_column:
                col_name = hour_to_column[hour]
                v = row.get(col_name, 0)
                try:
                    val = int(float(v)) if pd.notna(v) else 0
                except (ValueError, TypeError):
                    val = 0
            hourly_data[hour] = val
        rec = {
            'date': date_str,
            'station': station_name,
            'direction': direction,
            'hourly': hourly_data,
        }
        records.append(rec)
        index[(date_str, station_name, direction)] = hourly_data
    return records, index


def _get_hourly_value(traffic_index: Dict[Tuple[str, str, str], Dict[int, int]], date: str, station: str, direction: str, hour: int) -> int:
    if direction == '합계':
        b = traffic_index.get((date, station, '승차'), {})
        a = traffic_index.get((date, station, '하차'), {})
        return int(b.get(hour, 0)) + int(a.get(hour, 0))
    else:
        return int(traffic_index.get((date, station, direction), {}).get(hour, 0))


def _daily_range(traffic_index: Dict[Tuple[str, str, str], Dict[int, int]], date: str, station: str, direction: str) -> Tuple[int, int]:
    vals: List[int] = []
    for h in range(24):
        vals.append(_get_hourly_value(traffic_index, date, station, direction, h))
    if not vals:
        return (0, 1)
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return (mn, mn + 1)
    return (mn, mx)


def make_static_map_for_hour(
    map_df: pd.DataFrame,
    traffic_index: Dict[Tuple[str, str, str], Dict[int, int]],
    date: str,
    hour: int,
    direction: str = '합계',
    mode: str = 'waiting',  # 'waiting' or 'walking'
    platform_ratio: float = 0.35,
    corridor_ratio: float = 0.20,
    wait_min: float = 5.0,
    transit_sec: float = 40.0,
) -> folium.Map:
    # 좌표 정리
    df = map_df.copy()
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude']).copy()
    center_lat, center_lon = float(df['Latitude'].mean()), float(df['Longitude'].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB positron')

    # 면적 컬럼 준비
    total_area_col = None
    for c in df.columns:
        if '연면적' in c:
            total_area_col = c
            break
    if 'platform_area' not in df.columns:
        if total_area_col is not None:
            df['platform_area'] = pd.to_numeric(df[total_area_col], errors='coerce') * platform_ratio
        else:
            df['platform_area'] = pd.NA
    if 'corridor_area' not in df.columns:
        if total_area_col is not None:
            df['corridor_area'] = pd.to_numeric(df[total_area_col], errors='coerce') * corridor_ratio
        else:
            df['corridor_area'] = pd.NA

    # 경로 폴리라인
    route_coords = df.sort_values('역구성순서')[['Latitude', 'Longitude']].values.tolist()
    if len(route_coords) >= 2:
        folium.PolyLine(locations=route_coords, color='#555555', weight=4, opacity=0.7).add_to(m)

    # LOS 색상
    LOS_COLORS = {'A': '#2ECC71', 'B': '#8BC34A', 'C': '#FFC107', 'D': '#FF9800', 'E': '#FF5722', 'F': '#8B0000'}

    def los_waiting(area_per_person: float) -> str:
        if not pd.notna(area_per_person) or area_per_person <= 0:
            return 'A'
        if area_per_person >= 1.18:
            return 'A'
        if area_per_person >= 0.78:
            return 'B'
        if area_per_person >= 0.54:
            return 'C'
        if area_per_person >= 0.34:
            return 'D'
        if area_per_person >= 0.23:
            return 'E'
        return 'F'

    def los_walking(density: float) -> str:
        if density < 0:
            return 'A'
        if density <= 0.31:
            return 'A'
        if density <= 0.43:
            return 'B'
        if density <= 0.71:
            return 'C'
        if density <= 1.06:
            return 'D'
        if density <= 1.79:
            return 'E'
        return 'F'

    # 역별 처리
    for _, row in df.sort_values('역구성순서').iterrows():
        name = str(row['역명'])
        lat = float(row['Latitude'])
        lon = float(row['Longitude'])
        board_val = _get_hourly_value(traffic_index, date, name, '승차', hour)
        alight_val = _get_hourly_value(traffic_index, date, name, '하차', hour)
        if direction == '합계':
            value = board_val + alight_val
        elif direction == '승차':
            value = board_val
        else:
            value = alight_val

        if mode == 'waiting':
            occ = board_val * (wait_min / 60.0)
            area = row.get('platform_area')
            area = float(area) if pd.notna(area) else None
            area_per_person = (area / max(occ, 1e-9)) if area and area > 0 else float('inf')
            grade = los_waiting(area_per_person)
        else:
            flow = board_val + alight_val
            occ = flow * (transit_sec / 3600.0)
            area = row.get('corridor_area')
            area = float(area) if pd.notna(area) else None
            density = (occ / area) if area and area > 0 else 0.0
            grade = los_walking(density)

        color = LOS_COLORS.get(grade, '#CCCCCC')
        mn, mx = _daily_range(traffic_index, date, name, '승차' if direction == '합계' else direction)
        base = 15.0
        if mx == mn:
            radius = base
        else:
            ratio = (value - mn) / (mx - mn)
            radius = base + ratio * base * 2

        folium.CircleMarker(
            location=[lat, lon],
            radius=float(max(3.0, radius)),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=2,
            popup=f"{int(row['역구성순서'])}. {name} | {date} {hour:02d}시 | {direction} | LOS {grade}",
            tooltip=f"{int(row['역구성순서'])}. {name}",
        ).add_to(m)

    # 상단 표시
    title_html = f"""
    <div style='position: fixed; top: 10px; left: 10px; z-index: 9999; background: rgba(255,255,255,0.9); padding: 6px 10px; border:1px solid #aaa; border-radius: 4px; font-family: Arial; font-size: 14px;'>
      <b>{date} {hour:02d}시</b> · 모드: {mode} · 구분: {direction}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    return m


def export_hourly_frames(
    map_csv: str,
    stats_csv: str,
    out_dir: str,
    date: str | None = None,
    direction: str = '합계',
    mode: str = 'waiting',
    platform_ratio: float = 0.35,
    corridor_ratio: float = 0.20,
    wait_min: float = 5.0,
    transit_sec: float = 40.0,
    to_png: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df_map = _read_csv_any(map_csv)
    df_stats = _read_csv_any(stats_csv)
    _, traffic_index = _parse_traffic_df(df_stats)

    # 날짜 기본값
    if date is None:
        all_dates = sorted({k[0] for k in traffic_index.keys()})
        if not all_dates:
            raise ValueError('통계 데이터에서 날짜를 찾을 수 없습니다.')
        date = all_dates[0]

    # 0~23시 프레임 생성
    for h in range(24):
        fmap = make_static_map_for_hour(
            map_df=df_map,
            traffic_index=traffic_index,
            date=date,
            hour=h,
            direction=direction,
            mode=mode,
            platform_ratio=platform_ratio,
            corridor_ratio=corridor_ratio,
            wait_min=wait_min,
            transit_sec=transit_sec,
        )
        html_path = os.path.join(out_dir, f"{date}_{h:02d}시_{mode}_{direction}.html")
        fmap.save(html_path)

        if to_png:
            try:
                from selenium import webdriver  # type: ignore
                from selenium.webdriver.chrome.options import Options  # type: ignore
                from selenium.webdriver.chrome.service import Service  # type: ignore
                try:
                    from webdriver_manager.chrome import ChromeDriverManager  # type: ignore
                    svc = Service(ChromeDriverManager().install())
                except Exception:
                    svc = Service()  # 시스템 PATH의 chromedriver 사용
                options = Options()
                options.add_argument('--headless=new')
                options.add_argument('--window-size=1400,1000')
                driver = webdriver.Chrome(service=svc, options=options)
                driver.get('file://' + os.path.abspath(html_path))
                png_path = os.path.join(out_dir, f"{date}_{h:02d}시_{mode}_{direction}.png")
                import time
                time.sleep(1.0)
                driver.save_screenshot(png_path)
                driver.quit()
                print(f"saved png: {png_path}")
            except Exception as e:
                print(f"PNG 저장 실패(HTML만 생성): {e}")

if __name__ == "__main__":
    # 스크립트 파일의 디렉토리 기준으로 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser(description='지도.csv + 통계/통행량.csv 기반 과밀도 지도')
    parser.add_argument('--map', dest='map_csv', default=os.path.join(project_root, 'data', '지도.csv'), help='지도.csv 경로')
    parser.add_argument('--stats', dest='stats_csv', default=os.path.join(project_root, 'data', '승하차.csv'), help='통계/통행량 CSV 경로')
    parser.add_argument('--out', dest='output', default=os.path.join(project_root, 'subway_map.html'), help='출력 HTML 경로')
    parser.add_argument('--platform-ratio', type=float, default=0.35, help='총면적 대비 플랫폼 비율(플랫폼면적 없을 때)')
    parser.add_argument('--corridor-ratio', type=float, default=0.20, help='총면적 대비 통로 비율(통로면적 없을 때)')
    parser.add_argument('--wait-min', type=float, default=5.0, help='승차 대기 시간(분) 기본값')
    parser.add_argument('--transit-sec', type=float, default=40.0, help='통로 체류 시간(초) 기본값')
    # Export frames
    parser.add_argument('--export-frames', dest='export_dir', default=None, help='시간대별 HTML(및 PNG) 프레임 출력 디렉토리')
    parser.add_argument('--export-date', dest='export_date', default=None, help='프레임 생성 대상 날짜(미지정 시 첫 날짜)')
    parser.add_argument('--export-direction', dest='export_direction', default='합계', choices=['승차','하차','합계'], help='프레임 생성 대상 구분')
    parser.add_argument('--export-mode', dest='export_mode', default='waiting', choices=['waiting','walking'], help='프레임 생성 모드')
    parser.add_argument('--export-png', dest='export_png', action='store_true', help='PNG 스크린샷까지 저장 (Selenium 필요)')
    args = parser.parse_args()

    # 기본 인터랙티브 지도 생성
    map_obj = create_subway_map(
        csv_path=args.map_csv,
        traffic_path=args.stats_csv,
        output_path=args.output,
        default_platform_ratio=args.platform_ratio,
        default_corridor_ratio=args.corridor_ratio,
        default_board_wait_min=args.wait_min,
        default_transit_sec=args.transit_sec,
    )
    print("\n지도 생성 완료! 브라우저에서 출력 HTML 파일을 열어 확인하세요.")

    # 요청 시 시간대별 프레임도 생성
    if args.export_dir:
        print(f"시간대별 프레임을 생성합니다 → {args.export_dir}")
        export_hourly_frames(
            map_csv=args.map_csv,
            stats_csv=args.stats_csv,
            out_dir=args.export_dir,
            date=args.export_date,
            direction=args.export_direction,
            mode=args.export_mode,
            platform_ratio=args.platform_ratio,
            corridor_ratio=args.corridor_ratio,
            wait_min=args.wait_min,
            transit_sec=args.transit_sec,
            to_png=args.export_png,
        )
