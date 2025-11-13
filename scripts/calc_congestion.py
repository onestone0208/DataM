#!/usr/bin/env python3
"""
Compute per-station, per-time-slot congestion metrics derived from ridership counts
and station area information. The script focuses on numeric indicators (flow, density,
area per person) to avoid reliance on categorical LOS grades.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TIME_COLUMNS: Tuple[str, ...] = (
    "03-04시",
    "04-05시",
    "05-06시",
    "06-07시",
    "07-08시",
    "08-09시",
    "09-10시",
    "10-11시",
    "11-12시",
    "12-13시",
    "13-14시",
    "14-15시",
    "15-16시",
    "16-17시",
    "17-18시",
    "18-19시",
    "19-20시",
    "20-21시",
    "21-22시",
    "22-23시",
    "23-00시",
    "00-01시",
    "01-02시",
    "02-03시",
)


def parse_float(value: str | None) -> float:
    if value is None:
        return 0.0
    value = value.strip()
    if not value:
        return 0.0
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return 0.0


def parse_floor(value: str | None) -> int:
    if not value:
        return 1
    digits = re.findall(r"\d+", value)
    if not digits:
        return 1
    floor = int(digits[0])
    return max(1, floor)


def load_station_info(path: Path) -> Dict[str, Dict[str, float]]:
    info: Dict[str, Dict[str, float]] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            station = (row.get("역명") or "").strip()
            if not station:
                continue
            gross_area = parse_float(row.get("연면적(제곱미터)"))
            floors = parse_floor(row.get("층수"))
            effective_area = gross_area / floors if floors else gross_area
            info[station] = {
                "gross_area": gross_area,
                "floors": floors,
                "effective_area": effective_area,
            }
    return info


def load_date_info(path: Path | None) -> Dict[str, Dict[str, str]]:
    if path is None or not path.exists():
        return {}
    meta: Dict[str, Dict[str, str]] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = (row.get("date") or "").strip()
            if date:
                meta[date] = row
    return meta


def load_ridership(path: Path) -> List[Dict[str, str]]:
    encodings = ("cp949", "utf-8", "utf-8-sig")
    for enc in encodings:
        try:
            with path.open(encoding=enc) as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("all", b"", 0, 0, "Unable to decode ridership CSV")


def aggregate_ridership(
    rows: Iterable[Dict[str, str]]
) -> Tuple[
    Dict[Tuple[str, str, str], float],
    Dict[Tuple[str, str, str], float],
    Dict[Tuple[str, str, str], float],
    set[Tuple[str, str]],
]:
    totals: Dict[Tuple[str, str, str], float] = defaultdict(float)
    boardings: Dict[Tuple[str, str, str], float] = defaultdict(float)
    alightings: Dict[Tuple[str, str, str], float] = defaultdict(float)
    stations: set[Tuple[str, str]] = set()
    for row in rows:
        category = (row.get("구분") or "").strip()
        date = (row.get("날짜") or "").strip()
        station = (row.get("역명") or "").strip()
        if not date or not station or not category:
            continue
        stations.add((date, station))
        for col in TIME_COLUMNS:
            if col in row:
                value = parse_float(row[col])
                key = (date, station, col)
                if category == "합계":
                    totals[key] += value
                elif category == "승차":
                    boardings[key] += value
                elif category == "하차":
                    alightings[key] += value
    return totals, boardings, alightings, stations


def compute_decay(mean_dwell_minutes: float) -> float:
    mean_dwell_hours = mean_dwell_minutes / 60.0
    if mean_dwell_hours <= 0:
        return 0.0
    return math.exp(-1.0 / mean_dwell_hours)


def compute_occupancy(
    station_dates: Iterable[Tuple[str, str]],
    boardings: Dict[Tuple[str, str, str], float],
    alightings: Dict[Tuple[str, str, str], float],
    decay: float,
) -> Dict[Tuple[str, str, str], float]:
    occupancy: Dict[Tuple[str, str, str], float] = {}
    for date, station in sorted(station_dates):
        prev = 0.0
        for time_slot in TIME_COLUMNS:
            key = (date, station, time_slot)
            stay = prev * decay + boardings.get(key, 0.0) - alightings.get(key, 0.0)
            if stay < 0:
                stay = 0.0
            occupancy[key] = stay
            prev = stay
    return occupancy


def build_rows(
    totals: Dict[Tuple[str, str, str], float],
    station_info: Dict[str, Dict[str, float]],
    date_meta: Dict[str, Dict[str, str]],
    occupancy: Dict[Tuple[str, str, str], float],
) -> Iterable[Dict[str, str]]:
    for (date, station, time_slot), total_flow in sorted(totals.items()):
        area_info = station_info.get(station, {})
        effective_area = area_info.get("effective_area", 0.0)
        stay = occupancy.get((date, station, time_slot), 0.0)

        density = ""
        area_per_person = ""
        if effective_area and effective_area > 0:
            density = f"{total_flow / effective_area:.6f}"
            if total_flow > 0:
                area_per_person = f"{effective_area / total_flow:.6f}"

        row = {
            "date": date,
            "station": station,
            "time_slot": time_slot,
            "total_flow": f"{total_flow:.2f}",
            "occupancy": f"{stay:.2f}",
            "effective_area": f"{effective_area:.2f}" if effective_area else "",
            "density": density,
            "area_per_person": area_per_person,
        }
        if date in date_meta:
            for key, value in date_meta[date].items():
                if key == "date":
                    continue
                row[key] = value
        yield row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate numeric congestion metrics (density, area per person)."
    )
    parser.add_argument(
        "--ridership",
        type=Path,
        default=Path("data/승하차.csv"),
        help="Input 승하차 CSV (wide format, 구분=합계 행 사용)",
    )
    parser.add_argument(
        "--station-info",
        type=Path,
        default=Path("data/지도.csv"),
        help="Station metadata CSV containing 역명, 연면적(제곱미터), 층수",
    )
    parser.add_argument(
        "--date-info",
        type=Path,
        default=Path("data/날짜정보.csv"),
        help="Optional date metadata CSV (weekday/holiday/split etc.)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/혼잡도.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--mean-dwell-minutes",
        type=float,
        default=15.0,
        help="평균 체류 시간(분). 0 이하이면 이전 시점 체류 인원을 유지하지 않음.",
    )
    args = parser.parse_args()

    station_info = load_station_info(args.station_info)
    date_meta = load_date_info(args.date_info)
    ridership_rows = load_ridership(args.ridership)
    totals, boardings, alightings, station_dates = aggregate_ridership(ridership_rows)
    decay = compute_decay(args.mean_dwell_minutes)
    occupancy = compute_occupancy(station_dates, boardings, alightings, decay)
    rows = list(build_rows(totals, station_info, date_meta, occupancy))
    if not rows:
        raise SystemExit("No congestion rows computed. Check input data.")

    fieldnames = list(rows[0].keys())
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
