#!/usr/bin/env python3
"""
Split 승하차 시계열 데이터를 train/val/test로 나누기 위한 도구.

기본(Sequential) 모드는 날짜 경계를 기준으로 연속 구간을 나눠
시간 순서를 지키면서 데이터 누수를 방지한다.
Random 모드는 날짜 블록 단위로 셔플해 비율대로 분배한다.

출력:
  - data/날짜정보_split.csv  (date metadata + split column)
  - data/승하차_train.csv
  - data/승하차_val.csv
  - data/승하차_test.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List


DATA_DIR = Path("data")
RIDERSHIP_FILE = DATA_DIR / "승하차.csv"
DATE_INFO_FILE = DATA_DIR / "날짜정보.csv"
DATE_INFO_SPLIT_FILE = DATA_DIR / "날짜정보_split.csv"


def normalize_date(date_str: str) -> str:
    return datetime.strptime(date_str.strip(), "%Y-%m-%d").date().isoformat()


def build_sequential_split(
    dates: List[str], train_end: str, val_end: str
) -> Dict[str, str]:
    train_end_date = datetime.strptime(train_end, "%Y-%m-%d").date()
    val_end_date = datetime.strptime(val_end, "%Y-%m-%d").date()
    if val_end_date < train_end_date:
        raise ValueError("val_end must be >= train_end")

    split_map: Dict[str, str] = {}
    for d in dates:
        current = datetime.strptime(d, "%Y-%m-%d").date()
        if current <= train_end_date:
            split_map[d] = "train"
        elif current <= val_end_date:
            split_map[d] = "val"
        else:
            split_map[d] = "test"
    return split_map


def build_random_split(
    dates: List[str], train_ratio: float, val_ratio: float, seed: int
) -> Dict[str, str]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be >0 and sum <1.")

    rng = random.Random(seed)
    shuffled = dates[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_cut = int(total * train_ratio)
    val_cut = train_cut + int(total * val_ratio)

    split_map: Dict[str, str] = {}
    for idx, date in enumerate(shuffled):
        if idx < train_cut:
            split_map[date] = "train"
        elif idx < val_cut:
            split_map[date] = "val"
        else:
            split_map[date] = "test"

    # 보정: 비율에 따라 일부 split이 비어 있을 수 있으므로 최소 한 날짜는 보장
    for split in ("train", "val", "test"):
        if split not in split_map.values():
            split_map[shuffled[-1]] = split
    return split_map


def read_dates(date_info_path: Path) -> List[str]:
    dates: List[str] = []
    with date_info_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row.get("date")
            if date:
                dates.append(date.strip())
    if not dates:
        raise ValueError("No dates found in 날짜정보.csv")
    return sorted(set(dates))


def write_date_info_with_split(
    date_info_path: Path, output_path: Path, split_map: Dict[str, str]
) -> None:
    with date_info_path.open(encoding="utf-8") as src, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames + ["split"] if "split" not in reader.fieldnames else reader.fieldnames
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            date = row.get("date")
            if not date:
                continue
            row["split"] = split_map[date.strip()]
            writer.writerow(row)


def split_ridership(
    ridership_path: Path, split_map: Dict[str, str], output_dir: Path
) -> None:
    output_dir.mkdir(exist_ok=True)
    writers: Dict[str, csv.DictWriter] = {}
    files = {}
    try:
        with ridership_path.open(encoding="cp949") as src:
            reader = csv.DictReader(src)
            headers = reader.fieldnames or []

            for split in ("train", "val", "test"):
                file_path = output_dir / f"승하차_{split}.csv"
                f = file_path.open("w", newline="", encoding="utf-8")
                files[split] = f
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writers[split] = writer

            for row in reader:
                date = (row.get("날짜") or "").strip()
                split = split_map.get(date)
                if not split:
                    # skip rows with dates not in split map
                    continue
                writers[split].writerow(row)
    finally:
        for f in files.values():
            f.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Split ridership data into train/val/test.")
    parser.add_argument(
        "--ridership",
        type=Path,
        default=RIDERSHIP_FILE,
        help="Path to 승하차.csv",
    )
    parser.add_argument(
        "--date-info",
        type=Path,
        default=DATE_INFO_FILE,
        help="Path to 날짜정보.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory to write split CSV files",
    )
    parser.add_argument(
        "--mode",
        choices=("sequential", "random"),
        default="sequential",
        help="Split strategy",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2025-06-30",
        help="Inclusive end date for train split (sequential mode)",
    )
    parser.add_argument(
        "--val-end",
        type=str,
        default="2025-07-31",
        help="Inclusive end date for val split (sequential mode)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train ratio (random mode)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Val ratio (random mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (random mode)",
    )
    args = parser.parse_args()

    dates = read_dates(args.date_info)
    if args.mode == "sequential":
        split_map = build_sequential_split(dates, args.train_end, args.val_end)
    else:
        split_map = build_random_split(dates, args.train_ratio, args.val_ratio, args.seed)

    write_date_info_with_split(args.date_info, DATE_INFO_SPLIT_FILE, split_map)
    split_ridership(args.ridership, split_map, args.output_dir)


if __name__ == "__main__":
    main()
