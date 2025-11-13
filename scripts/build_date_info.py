#!/usr/bin/env python3
"""
Combine holiday and temperature metadata into a date-level lookup table.

The script scans `data/승하차.csv` to pick the set of dates that actually appear
in the ridership dataset, then enriches each date with:
  - 공휴일 여부 및 명칭 (from `data/공휴일.csv`)
  - 일별 평균/최저/최고 기온 (from `data/기온.csv`)

The resulting table is written to `data/날짜정보.csv` and can be joined back to
the ridership data by date as an external feature source.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence


DATA_DIR = Path("data")
RIDERSHIP_FILE = DATA_DIR / "승하차.csv"
HOLIDAY_FILE = DATA_DIR / "공휴일.csv"
TEMP_FILE = DATA_DIR / "기온.csv"
OUTPUT_FILE = DATA_DIR / "날짜정보.csv"
WEEKDAY_NAMES = ["월", "화", "수", "목", "금", "토", "일"]


def load_dates_from_ridership(path: Path) -> List[str]:
    dates: set[str] = set()
    with path.open(encoding="cp949") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row["날짜"].strip()
            if date:
                dates.add(date)
    return sorted(dates)


def load_holidays(path: Path) -> Dict[str, List[str]]:
    holidays: Dict[str, List[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Some rows may include duplicated holidays on the same date.
            raw_date = row.get("Start date") or row.get("\ufeffStart date") or ""
            date = raw_date.strip()
            if not date:
                continue
            subject = (row.get("Subject") or "").strip()
            if subject and subject not in holidays[date]:
                holidays[date].append(subject)
    return holidays


def load_temperatures(path: Path) -> Dict[str, Dict[str, str]]:
    temps: Dict[str, Dict[str, str]] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_date = row.get("날짜") or row.get("\ufeff날짜") or ""
            date = raw_date.strip()
            if not date:
                continue
            temps[date] = {
                "avg_temp_c": row.get("평균기온(℃)", "").strip(),
                "min_temp_c": row.get("최저기온(℃)", "").strip(),
                "max_temp_c": row.get("최고기온(℃)", "").strip(),
            }
    return temps


def write_date_table(
    dates: Sequence[str],
    holidays: Dict[str, List[str]],
    temps: Dict[str, Dict[str, str]],
    output_path: Path,
) -> None:
    fieldnames = [
        "date",
        "weekday",
        "weekday_name",
        "is_holiday",
        "holiday_name",
        "avg_temp_c",
        "min_temp_c",
        "max_temp_c",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for date in dates:
            holiday_list = holidays.get(date, []).copy()
            temp_info = temps.get(date, {})
            current = datetime.strptime(date, "%Y-%m-%d")
            weekday_idx = current.weekday()
            weekday_name = WEEKDAY_NAMES[weekday_idx]

            # Treat weekends as holiday-equivalent signals.
            if weekday_idx >= 5:  # 5=토, 6=일
                weekend_label = f"{weekday_name}요일"
                if weekend_label not in holiday_list:
                    holiday_list.append(weekend_label)

            writer.writerow(
                {
                    "date": date,
                    "weekday": weekday_idx,
                    "weekday_name": weekday_name,
                    "is_holiday": "1" if holiday_list else "0",
                    "holiday_name": "|".join(holiday_list),
                    "avg_temp_c": temp_info.get("avg_temp_c", ""),
                    "min_temp_c": temp_info.get("min_temp_c", ""),
                    "max_temp_c": temp_info.get("max_temp_c", ""),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build date-level metadata table.")
    parser.add_argument(
        "--ridership",
        type=Path,
        default=RIDERSHIP_FILE,
        help="Path to 승하차.csv",
    )
    parser.add_argument(
        "--holidays",
        type=Path,
        default=HOLIDAY_FILE,
        help="Path to 공휴일.csv",
    )
    parser.add_argument(
        "--temperatures",
        type=Path,
        default=TEMP_FILE,
        help="Path to 기온.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="Output CSV path",
    )
    args = parser.parse_args()

    dates = load_dates_from_ridership(args.ridership)
    holidays = load_holidays(args.holidays)
    temps = load_temperatures(args.temperatures)
    write_date_table(dates, holidays, temps, args.output)


if __name__ == "__main__":
    main()
