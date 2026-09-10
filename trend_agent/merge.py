from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import load_csv

DEFAULT_TICKER_MAP_PATH = Path(__file__).resolve().parent.parent / "ticker_map.json"

_RENAME_COLUMNS = {
    "color -- streamlit-generated": "series",
    "value -- streamlit-generated": "value",
}


def load_ticker_map(path: str | Path = DEFAULT_TICKER_MAP_PATH) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_csv_folder(
    folder: str | Path,
    ticker_map: dict[str, str],
    ticker_column: str = "종목",
) -> pd.DataFrame:
    """folder 안의 CSV들을 ticker_map(파일명 -> 종목명)에 따라 하나의 long-format DataFrame으로 합친다."""
    folder = Path(folder)
    frames = []
    missing = []

    for filename, ticker in ticker_map.items():
        path = folder / filename
        if not path.exists():
            missing.append(filename)
            continue
        df = load_csv(path)
        df = df.rename(columns=_RENAME_COLUMNS)
        df.insert(0, ticker_column, ticker)
        frames.append(df)

    if missing:
        raise FileNotFoundError(f"ticker_map에 있지만 폴더에서 찾지 못한 파일: {missing}")
    if not frames:
        raise ValueError("병합할 CSV가 없습니다.")

    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    return merged.sort_values([ticker_column, "date"]).reset_index(drop=True)


def merge_and_save(
    folder: str | Path,
    ticker_map_path: str | Path = DEFAULT_TICKER_MAP_PATH,
    output_path: str | Path = "merged.csv",
    ticker_column: str = "종목",
) -> Path:
    ticker_map = load_ticker_map(ticker_map_path)
    merged = merge_csv_folder(folder, ticker_map, ticker_column=ticker_column)
    output_path = Path(output_path)
    merged.to_csv(output_path, index=False)
    return output_path
