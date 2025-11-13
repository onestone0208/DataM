"""
LOS 데이터 기반 밀집도 인사이트 시각화

시간대, 요일, 역별 밀집 패턴을 분석하고 시각화합니다.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional
import argparse

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False


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


def create_time_heatmap(los_df: pd.DataFrame, output_dir: str):
    """시간대별 LOS 등급 히트맵 (요일별)"""
    print("\n=== 시간대별 히트맵 생성 ===")
    
    # LOS 등급을 숫자로 변환
    los_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
    los_df['los_numeric'] = los_df['los_grade'].map(los_map)
    
    # 요일별, 시간대별 평균 LOS 계산
    if 'weekday_name' in los_df.columns:
        heatmap_data = los_df.groupby(['weekday_name', 'hour'])['los_numeric'].mean().unstack(fill_value=0)
        
        # 요일 순서 정렬
        weekday_order = ['월', '화', '수', '목', '금', '토', '일']
        heatmap_data = heatmap_data.reindex([d for d in weekday_order if d in heatmap_data.index])
        
        plt.figure(figsize=(16, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                   cbar_kws={'label': '평균 LOS 등급 (1=A, 6=F)'}, 
                   linewidths=0.5, linecolor='gray')
        plt.title('요일별·시간대별 평균 LOS 등급 히트맵', fontsize=16, fontweight='bold')
        plt.xlabel('시간대', fontsize=12)
        plt.ylabel('요일', fontsize=12)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, '1_시간대별_LOS_히트맵.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"저장: {output_path}")


def create_station_heatmap(los_df: pd.DataFrame, output_dir: str):
    """역별·시간대별 LOS 등급 히트맵"""
    print("\n=== 역별·시간대별 히트맵 생성 ===")
    
    los_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
    los_df['los_numeric'] = los_df['los_grade'].map(los_map)
    
    # 역별, 시간대별 평균 LOS 계산
    heatmap_data = los_df.groupby(['station', 'hour'])['los_numeric'].mean().unstack(fill_value=0)
    
    # 역순서로 정렬 (order 컬럼이 있으면)
    if 'order' in los_df.columns:
        station_order = los_df[['station', 'order']].drop_duplicates().set_index('station')['order'].sort_values()
        heatmap_data = heatmap_data.reindex(station_order.index)
    
    plt.figure(figsize=(20, max(10, len(heatmap_data) * 0.3)))
    sns.heatmap(heatmap_data, annot=False, cmap='RdYlGn_r', 
               cbar_kws={'label': '평균 LOS 등급 (1=A, 6=F)'}, 
               linewidths=0.5, linecolor='gray', xticklabels=True, yticklabels=True)
    plt.title('역별·시간대별 평균 LOS 등급 히트맵', fontsize=16, fontweight='bold')
    plt.xlabel('시간대', fontsize=12)
    plt.ylabel('역명', fontsize=12)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '2_역별_시간대별_LOS_히트맵.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"저장: {output_path}")


def create_los_distribution(los_df: pd.DataFrame, output_dir: str):
    """LOS 등급 분포 시각화"""
    print("\n=== LOS 등급 분포 시각화 ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 전체 LOS 등급 분포 (막대 그래프)
    los_counts = los_df['los_grade'].value_counts().sort_index()
    colors = {'A': '#2ECC71', 'B': '#8BC34A', 'C': '#FFC107', 'D': '#FF9800', 'E': '#FF5722', 'F': '#8B0000'}
    ax1 = axes[0, 0]
    bars = ax1.bar(los_counts.index, los_counts.values, 
                   color=[colors.get(g, '#CCCCCC') for g in los_counts.index])
    ax1.set_title('전체 LOS 등급 분포', fontsize=14, fontweight='bold')
    ax1.set_xlabel('LOS 등급')
    ax1.set_ylabel('개수')
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom')
    
    # 2. 시간대별 LOS 등급 분포
    ax2 = axes[0, 1]
    los_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
    los_df['los_numeric'] = los_df['los_grade'].map(los_map)
    hourly_los = los_df.groupby('hour')['los_numeric'].mean()
    ax2.plot(hourly_los.index, hourly_los.values, marker='o', linewidth=2, markersize=8)
    ax2.set_title('시간대별 평균 LOS 등급 추이', fontsize=14, fontweight='bold')
    ax2.set_xlabel('시간대')
    ax2.set_ylabel('평균 LOS 등급 (1=A, 6=F)')
    ax2.set_ylim([0, 7])
    ax2.grid(alpha=0.3)
    ax2.set_xticks(range(0, 24, 2))
    
    # 3. 요일별 LOS 등급 분포
    if 'weekday_name' in los_df.columns:
        ax3 = axes[1, 0]
        weekday_order = ['월', '화', '수', '목', '금', '토', '일']
        weekday_los = los_df.groupby('weekday_name')['los_numeric'].mean()
        weekday_los = weekday_los.reindex([d for d in weekday_order if d in weekday_los.index])
        bars = ax3.bar(weekday_los.index, weekday_los.values, 
                      color=['#FF9800' if v > 3 else '#8BC34A' for v in weekday_los.values])
        ax3.set_title('요일별 평균 LOS 등급', fontsize=14, fontweight='bold')
        ax3.set_xlabel('요일')
        ax3.set_ylabel('평균 LOS 등급 (1=A, 6=F)')
        ax3.set_ylim([0, 7])
        ax3.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
    
    # 4. 역별 평균 LOS 등급 (상위/하위)
    ax4 = axes[1, 1]
    station_los = los_df.groupby('station')['los_numeric'].mean().sort_values(ascending=False)
    top_n = min(15, len(station_los))
    station_los_top = station_los.head(top_n)
    
    colors_station = ['#8B0000' if v > 4 else '#FF5722' if v > 3 else '#FF9800' if v > 2 else '#8BC34A' 
                     for v in station_los_top.values]
    bars = ax4.barh(range(len(station_los_top)), station_los_top.values, color=colors_station)
    ax4.set_yticks(range(len(station_los_top)))
    ax4.set_yticklabels([s.replace('역', '') for s in station_los_top.index], fontsize=9)
    ax4.set_title(f'역별 평균 LOS 등급 (상위 {top_n}개)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('평균 LOS 등급 (1=A, 6=F)')
    ax4.grid(axis='x', alpha=0.3)
    ax4.set_xlim([0, 7])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '3_LOS_등급_분포_종합.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"저장: {output_path}")


def create_peak_analysis(los_df: pd.DataFrame, output_dir: str):
    """피크 시간대 분석"""
    print("\n=== 피크 시간대 분석 ===")
    
    los_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
    los_df['los_numeric'] = los_df['los_grade'].map(los_map)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 시간대별 통행량과 LOS 등급
    ax1 = axes[0]
    if 'traffic' in los_df.columns:
        hourly_traffic = los_df.groupby('hour')['traffic'].mean()
        hourly_los = los_df.groupby('hour')['los_numeric'].mean()
        
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(hourly_traffic.index, hourly_traffic.values, 
                        'b-', marker='o', linewidth=2, label='평균 통행량')
        ax1.set_xlabel('시간대', fontsize=12)
        ax1.set_ylabel('평균 통행량', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_xticks(range(0, 24, 2))
        ax1.grid(alpha=0.3)
        
        line2 = ax1_twin.plot(hourly_los.index, hourly_los.values, 
                             'r-', marker='s', linewidth=2, label='평균 LOS 등급')
        ax1_twin.set_ylabel('평균 LOS 등급 (1=A, 6=F)', color='r', fontsize=12)
        ax1_twin.tick_params(axis='y', labelcolor='r')
        ax1_twin.set_ylim([0, 7])
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        ax1.set_title('시간대별 통행량 vs LOS 등급', fontsize=14, fontweight='bold')
    
    # 2. 요일별 피크 시간대
    if 'weekday_name' in los_df.columns:
        ax2 = axes[1]
        weekday_order = ['월', '화', '수', '목', '금', '토', '일']
        
        peak_data = []
        for weekday in weekday_order:
            weekday_data = los_df[los_df['weekday_name'] == weekday]
            if len(weekday_data) > 0:
                hourly_los_weekday = weekday_data.groupby('hour')['los_numeric'].mean()
                peak_hour = hourly_los_weekday.idxmax()
                peak_los = hourly_los_weekday.max()
                peak_data.append({'요일': weekday, '피크시간': peak_hour, 'LOS': peak_los})
        
        if peak_data:
            peak_df = pd.DataFrame(peak_data)
            ax2.scatter(peak_df['요일'], peak_df['피크시간'], 
                       s=peak_df['LOS'] * 100, c=peak_df['LOS'], 
                       cmap='RdYlGn_r', alpha=0.7, edgecolors='black', linewidth=2)
            ax2.set_title('요일별 피크 시간대 (크기=LOS 등급)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('요일', fontsize=12)
            ax2.set_ylabel('피크 시간대', fontsize=12)
            ax2.set_ylim([-1, 24])
            ax2.set_yticks(range(0, 24, 2))
            ax2.grid(alpha=0.3)
            plt.colorbar(ax2.collections[0], ax=ax2, label='LOS 등급')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '4_피크_시간대_분석.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"저장: {output_path}")


def create_congestion_patterns(los_df: pd.DataFrame, output_dir: str):
    """과밀 패턴 분석 (LOS D 이상)"""
    print("\n=== 과밀 패턴 분석 ===")
    
    los_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
    los_df['los_numeric'] = los_df['los_grade'].map(los_map)
    
    # LOS D 이상만 필터링
    congestion_df = los_df[los_df['los_numeric'] >= 4].copy()
    
    if len(congestion_df) == 0:
        print("경고: LOS D 이상 데이터가 없습니다.")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # 1. 역별 과밀 발생 빈도
    ax1 = axes[0]
    station_congestion = congestion_df.groupby('station').size().sort_values(ascending=False).head(20)
    bars = ax1.barh(range(len(station_congestion)), station_congestion.values, 
                    color='#FF5722', alpha=0.8)
    ax1.set_yticks(range(len(station_congestion)))
    ax1.set_yticklabels([s.replace('역', '') for s in station_congestion.index], fontsize=10)
    ax1.set_xlabel('과밀 발생 횟수 (LOS D 이상)', fontsize=12)
    ax1.set_title('역별 과밀 발생 빈도 (상위 20개)', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2., 
                f'{int(width)}', ha='left', va='center')
    
    # 2. 시간대별 과밀 발생 분포
    ax2 = axes[1]
    hour_congestion = congestion_df['hour'].value_counts().sort_index()
    bars = ax2.bar(hour_congestion.index, hour_congestion.values, 
                   color='#FF9800', alpha=0.8)
    ax2.set_xlabel('시간대', fontsize=12)
    ax2.set_ylabel('과밀 발생 횟수', fontsize=12)
    ax2.set_title('시간대별 과밀 발생 분포', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '5_과밀_패턴_분석.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"저장: {output_path}")


def create_summary_report(los_df: pd.DataFrame, output_dir: str):
    """요약 리포트 생성 (텍스트)"""
    print("\n=== 요약 리포트 생성 ===")
    
    los_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
    los_df['los_numeric'] = los_df['los_grade'].map(los_map)
    
    report = []
    report.append("=" * 60)
    report.append("LOS 데이터 밀집도 분석 리포트")
    report.append("=" * 60)
    report.append("")
    
    # 기본 통계
    report.append("[기본 통계]")
    report.append(f"  총 레코드 수: {len(los_df):,}개")
    report.append(f"  분석 역 수: {los_df['station'].nunique()}개")
    if 'weekday_name' in los_df.columns:
        report.append(f"  분석 요일 수: {los_df['weekday_name'].nunique()}개")
    report.append("")
    
    # LOS 등급별 분포
    report.append("[LOS 등급별 분포]")
    los_counts = los_df['los_grade'].value_counts().sort_index()
    for grade in ['A', 'B', 'C', 'D', 'E', 'F']:
        count = los_counts.get(grade, 0)
        pct = (count / len(los_df) * 100) if len(los_df) > 0 else 0
        report.append(f"  LOS {grade}: {count:,}개 ({pct:.1f}%)")
    report.append("")
    
    # 평균 LOS 등급
    avg_los = los_df['los_numeric'].mean()
    report.append(f"[평균 LOS 등급]")
    report.append(f"  평균: {avg_los:.2f} (숫자 기준, 1=A ~ 6=F)")
    report.append("")
    
    # 시간대별 분석
    report.append("[시간대별 분석]")
    hourly_los = los_df.groupby('hour')['los_numeric'].mean()
    worst_hour = hourly_los.idxmax()
    best_hour = hourly_los.idxmin()
    report.append(f"  최악 시간대: {worst_hour}시 (평균 LOS: {hourly_los[worst_hour]:.2f})")
    report.append(f"  최선 시간대: {best_hour}시 (평균 LOS: {hourly_los[best_hour]:.2f})")
    report.append("")
    
    # 역별 분석
    report.append("[역별 분석 (상위 10개)]")
    station_los = los_df.groupby('station')['los_numeric'].mean().sort_values(ascending=False).head(10)
    for i, (station, los_val) in enumerate(station_los.items(), 1):
        report.append(f"  {i}. {station}: {los_val:.2f}")
    report.append("")
    
    # 과밀 구간 (LOS D 이상)
    congestion_df = los_df[los_df['los_numeric'] >= 4].copy()
    if len(congestion_df) > 0:
        report.append(f"[과밀 구간 분석 (LOS D 이상)]")
        report.append(f"  총 과밀 발생: {len(congestion_df):,}회")
        report.append(f"  과밀 역 수: {congestion_df['station'].nunique()}개")
        report.append("")
        
        if 'weekday_name' in congestion_df.columns:
            weekday_congestion = congestion_df['weekday_name'].value_counts()
            worst_weekday = weekday_congestion.idxmax()
            report.append(f"  가장 과밀한 요일: {worst_weekday} ({weekday_congestion[worst_weekday]}회)")
            report.append("")
        
        worst_hour_congestion = congestion_df['hour'].value_counts()
        worst_hour = worst_hour_congestion.idxmax()
        report.append(f"  가장 과밀한 시간대: {worst_hour}시 ({worst_hour_congestion[worst_hour]}회)")
        report.append("")
    else:
        report.append("[과밀 구간 분석]")
        report.append("  LOS D 이상 데이터가 없습니다.")
        report.append("")
    
    report.append("=" * 60)
    
    # 리포트 저장
    report_text = "\n".join(report)
    output_path = os.path.join(output_dir, '6_밀집도_분석_리포트.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n리포트 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='LOS 데이터 밀집도 인사이트 시각화')
    parser.add_argument('--los-data', default='data/los_calculated.csv', help='LOS 계산 결과 CSV 파일')
    parser.add_argument('--output-dir', default='output', help='출력 디렉토리')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== 데이터 로드 ===")
    los_df = read_csv_safe(args.los_data)
    print(f"LOS 데이터: {len(los_df)}개 레코드")
    
    print("\n=== LOS 등급별 통계 ===")
    if 'los_grade' in los_df.columns:
        los_counts = los_df['los_grade'].value_counts().sort_index()
        for grade in ['A', 'B', 'C', 'D', 'E', 'F']:
            count = los_counts.get(grade, 0)
            pct = (count / len(los_df) * 100) if len(los_df) > 0 else 0
            print(f"  LOS {grade}: {count:,}개 ({pct:.1f}%)")
    
    # 시각화 생성
    print("\n=== 시각화 생성 시작 ===")
    
    create_time_heatmap(los_df, args.output_dir)
    create_station_heatmap(los_df, args.output_dir)
    create_los_distribution(los_df, args.output_dir)
    create_peak_analysis(los_df, args.output_dir)
    create_congestion_patterns(los_df, args.output_dir)
    create_summary_report(los_df, args.output_dir)
    
    print("\n=== 완료 ===")
    print(f"모든 시각화가 {args.output_dir} 디렉토리에 저장되었습니다.")


if __name__ == '__main__':
    main()

