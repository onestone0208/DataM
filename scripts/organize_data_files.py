#!/usr/bin/env python3
"""
데이터 파일 정리 및 이름 표준화 스크립트
"""

import shutil
from pathlib import Path
import argparse


def organize_data_files(data_dir: Path, backup: bool = True) -> None:
    """데이터 파일들을 정리하고 이름을 표준화"""
    
    print("=== 데이터 파일 정리 시작 ===")
    
    # 백업 디렉토리 생성
    if backup:
        backup_dir = data_dir / "backup"
        backup_dir.mkdir(exist_ok=True)
        print(f"백업 디렉토리 생성: {backup_dir}")
    
    # 1. 중복/임시 파일 정리
    files_to_remove = [
        "temporal_features_fixed.csv",
        "주요장소_tagged.csv"
    ]
    
    print(f"\n=== 불필요한 파일 정리 ===")
    for file_name in files_to_remove:
        file_path = data_dir / file_name
        if file_path.exists():
            if backup:
                shutil.move(str(file_path), str(backup_dir / file_name))
                print(f"백업 후 제거: {file_name}")
            else:
                file_path.unlink()
                print(f"제거: {file_name}")
        else:
            print(f"없음: {file_name}")
    
    # 2. 파일 이름 표준화
    rename_map = {
        "hour_features.csv": "time_features.csv",
        "temporal_features.csv": "date_features.csv"
    }
    
    print(f"\n=== 파일 이름 표준화 ===")
    for old_name, new_name in rename_map.items():
        old_path = data_dir / old_name
        new_path = data_dir / new_name
        
        if old_path.exists():
            if backup and new_path.exists():
                shutil.copy2(str(new_path), str(backup_dir / new_name))
                print(f"기존 파일 백업: {new_name}")
            
            shutil.move(str(old_path), str(new_path))
            print(f"이름 변경: {old_name} → {new_name}")
        else:
            print(f"없음: {old_name}")
    
    # 3. 최종 파일 목록 정리
    print(f"\n=== 최종 데이터 파일 목록 ===")
    
    # 카테고리별 분류
    file_categories = {
        "핵심 데이터": [
            "혼잡도.csv",           # 타깃 데이터
            "승하차.csv",           # 원본 승하차 데이터
            "node_features.csv",    # 역별 특징
            "date_features.csv",    # 날짜별 특징  
            "time_features.csv"     # 시간대별 특징
        ],
        "분할 데이터": [
            "승하차_train.csv",
            "승하차_val.csv", 
            "승하차_test.csv",
            "날짜정보_split.csv"
        ],
        "원본 메타데이터": [
            "지도.csv",             # 역 정보
            "주요장소.csv",         # POI (태깅된 버전)
            "공휴일.csv",           # 공휴일 정보
            "기온.csv",             # 기상 데이터
            "날짜정보.csv"          # 날짜 메타데이터
        ],
        "분석 결과": [
            "los_calculated.csv",   # LOS 계산 결과
            "통계.csv"              # 통계 데이터
        ],
        "참고 자료": [
            "poi_category_samples.txt",
            "node_features_info.json"
        ]
    }
    
    total_files = 0
    for category, files in file_categories.items():
        print(f"\n[{category}]")
        existing_files = []
        for file_name in files:
            file_path = data_dir / file_name
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                existing_files.append(f"{file_name} ({size_mb:.1f}MB)")
                total_files += 1
            else:
                existing_files.append(f"{file_name} (없음)")
        
        for file_info in existing_files:
            print(f"  - {file_info}")
    
    print(f"\n총 {total_files}개 파일 정리 완료")
    
    # 4. README 생성
    create_data_readme(data_dir)


def create_data_readme(data_dir: Path) -> None:
    """데이터 디렉토리 README 생성"""
    
    readme_content = """# 데이터 파일 설명

## 핵심 데이터
- `혼잡도.csv`: 역별 시간대별 혼잡도 타깃 데이터 (128K+ 레코드)
- `승하차.csv`: 원본 승하차 데이터
- `node_features.csv`: 역별 특징 벡터 (22개 역 × 16차원)
- `date_features.csv`: 날짜별 특징 벡터 (243일 × 24차원)
- `time_features.csv`: 시간대별 특징 벡터 (24시간 × 9차원)

## 분할 데이터
- `승하차_train.csv`: 훈련용 승하차 데이터
- `승하차_val.csv`: 검증용 승하차 데이터
- `승하차_test.csv`: 테스트용 승하차 데이터
- `날짜정보_split.csv`: 날짜별 분할 정보

## 원본 메타데이터
- `지도.csv`: 역 기본 정보 (좌표, 연면적 등)
- `주요장소.csv`: POI 데이터 (태깅된 버전, 387개)
- `공휴일.csv`: 공휴일 정보
- `기온.csv`: 기상 데이터 (일별 기온)
- `날짜정보.csv`: 날짜 메타데이터

## 분석 결과
- `los_calculated.csv`: LOS 등급 계산 결과
- `통계.csv`: 통계 분석 데이터

## 참고 자료
- `poi_category_samples.txt`: POI 카테고리 분류 샘플
- `node_features_info.json`: 노드 특징 통계 정보

## 데이터 플로우
```
원본 데이터 → 전처리 → 특징 생성 → 모델 입력
├── 승하차.csv → 혼잡도.csv (타깃)
├── 지도.csv + 주요장소.csv → node_features.csv
├── 공휴일.csv + 기온.csv → date_features.csv
└── 시간대 정보 → time_features.csv
```
"""
    
    readme_path = data_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"README 생성: {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="데이터 파일 정리")
    parser.add_argument(
        "--data-dir", 
        type=Path, 
        default=Path("data"),
        help="데이터 디렉토리 경로"
    )
    parser.add_argument(
        "--no-backup", 
        action="store_true",
        help="백업 없이 바로 삭제"
    )
    
    args = parser.parse_args()
    
    organize_data_files(args.data_dir, backup=not args.no_backup)
    print(f"\n데이터 파일 정리 완료!")


if __name__ == "__main__":
    main()
