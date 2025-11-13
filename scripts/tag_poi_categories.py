#!/usr/bin/env python3
"""
POI 데이터에 카테고리 태깅을 수행하는 스크립트

주요장소.csv의 '출구별 주요시설명' 컬럼을 분석해서 
각 POI를 적절한 카테고리로 분류하고 태깅된 CSV를 생성합니다.
"""

import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Set
import argparse


def define_category_keywords() -> Dict[str, List[str]]:
    """카테고리별 키워드 정의"""
    return {
        "교육": [
            "초등학교", "중학교", "고등학교", "대학교", "대학원", "유치원", "어린이집",
            "교육원", "학원", "도서관", "교육청", "교육문화원", "교육진흥원", "정보대학"
        ],
        "주거": [
            "아파트", "빌라", "주택", "타운", "마을", "단지", "휴먼시아", 
            "푸르지오", "자이", "힐스테이트", "래미안"
        ],
        "관공서": [
            "청사", "시청", "구청", "동사무소", "행정복지센터", "우체국", "세무서",
            "법원", "검찰청", "경찰청", "경찰서", "파출소", "지구대", "소방서",
            "119안전센터", "보건소", "고용노동청", "병무청", "선거관리위원회",
            "토지주택공사", "환경공단", "보훈청", "통계센터"
        ],
        "상업": [
            "백화점", "마트", "쇼핑몰", "상가", "지하상가", "시장", "트레이더스",
            "이마트", "롯데", "갤러리아", "세이", "뷔페"
        ],
        "의료": [
            "병원", "의원", "클리닉", "보건소", "한방병원", "치과", "내과",
            "여성병원", "검진센터", "헌혈센터"
        ],
        "교통": [
            "터미널", "버스", "지하철", "도시철도", "기지", "주차장", "승강장",
            "교통공사", "BRT", "역", "광장", "통로"
        ],
        "체육문화": [
            "공원", "체육관", "경기장", "운동장", "수영장", "헬스", "골프",
            "박물관", "미술관", "도서관", "문화원", "예술의전당", "갤러리",
            "둔치", "유적지", "생활관", "국민생활관"
        ],
        "종교": [
            "교회", "성당", "절", "사찰", "성전", "침례교", "기독교", "불교", "천주교"
        ],
        "금융": [
            "은행", "신협", "저축은행", "보험", "증권", "카드", "금고"
        ],
        "숙박": [
            "호텔", "모텔", "펜션", "리조트", "게스트하우스"
        ],
        "방향지시": [
            "방면", "네거리", "중앙로", "교류센터"
        ]
    }


def classify_poi(facility_name: str, keywords: Dict[str, List[str]]) -> str:
    """POI 이름을 기반으로 카테고리 분류"""
    facility_name = facility_name.strip()
    
    # 각 카테고리별로 키워드 매칭 점수 계산
    category_scores = {}
    
    for category, keyword_list in keywords.items():
        score = 0
        for keyword in keyword_list:
            if keyword in facility_name:
                # 키워드 길이에 비례한 가중치 (더 구체적인 키워드에 높은 점수)
                score += len(keyword)
        category_scores[category] = score
    
    # 가장 높은 점수의 카테고리 반환
    if max(category_scores.values()) > 0:
        return max(category_scores, key=category_scores.get)
    else:
        return "기타"


def analyze_poi_distribution(df: pd.DataFrame) -> None:
    """POI 카테고리 분포 분석 및 출력"""
    print("\n=== POI 카테고리 분포 ===")
    category_counts = df['category'].value_counts()
    total = len(df)
    
    for category, count in category_counts.items():
        percentage = (count / total) * 100
        print(f"{category:10s}: {count:3d}개 ({percentage:5.1f}%)")
    
    print(f"\n총 POI 수: {total}개")
    
    # 역별 POI 수 통계
    print("\n=== 역별 POI 수 ===")
    station_counts = df['역명'].value_counts()
    print(f"평균 POI/역: {station_counts.mean():.1f}개")
    print(f"최대 POI/역: {station_counts.max()}개 ({station_counts.idxmax()})")
    print(f"최소 POI/역: {station_counts.min()}개 ({station_counts.idxmin()})")


def save_sample_by_category(df: pd.DataFrame, output_dir: Path) -> None:
    """카테고리별 샘플 저장 (검증용)"""
    sample_file = output_dir / "poi_category_samples.txt"
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write("=== POI 카테고리 분류 샘플 ===\n\n")
        
        for category in df['category'].unique():
            f.write(f"[{category}]\n")
            samples = df[df['category'] == category]['출구별 주요시설명'].head(10)
            for sample in samples:
                f.write(f"  - {sample}\n")
            f.write("\n")
    
    print(f"카테고리별 샘플이 저장되었습니다: {sample_file}")


def main():
    parser = argparse.ArgumentParser(description="POI 데이터에 카테고리 태깅")
    parser.add_argument(
        "--input", 
        type=Path, 
        default=Path("data/주요장소.csv"),
        help="입력 POI CSV 파일"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=Path("data/주요장소_tagged.csv"),
        help="출력 태깅된 CSV 파일"
    )
    parser.add_argument(
        "--analysis", 
        action="store_true",
        help="분석 결과 출력 및 샘플 파일 생성"
    )
    
    args = parser.parse_args()
    
    # CSV 읽기 (인코딩 문제 대응)
    try:
        df = pd.read_csv(args.input, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(args.input, encoding='cp949')
    
    print(f"원본 데이터 로드: {len(df)}개 POI")
    
    # 카테고리 키워드 정의
    keywords = define_category_keywords()
    
    # POI 분류 수행
    print("POI 카테고리 분류 중...")
    df['category'] = df['출구별 주요시설명'].apply(
        lambda x: classify_poi(x, keywords)
    )
    
    # 결과 저장
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"태깅된 데이터 저장: {args.output}")
    
    # 분석 수행
    if args.analysis:
        analyze_poi_distribution(df)
        save_sample_by_category(df, args.output.parent)
    
    print("\n태깅 완료!")


if __name__ == "__main__":
    main()
