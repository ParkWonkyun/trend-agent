from __future__ import annotations

import argparse
from pathlib import Path

from trend_agent.merge import merge_and_save

DEFAULT_RAW_DATA_DIR = Path(__file__).resolve().parent.parent  # .../raw data


def main() -> None:
    parser = argparse.ArgumentParser(description="원본 CSV들을 종목명을 붙여 하나의 CSV로 병합한다.")
    parser.add_argument("--folder", default=str(DEFAULT_RAW_DATA_DIR), help="원본 CSV들이 있는 폴더")
    parser.add_argument("--ticker-map", default="ticker_map.json", help="파일명->종목명 매핑 JSON 경로")
    parser.add_argument("--output", default="merged.csv", help="출력 CSV 경로")
    args = parser.parse_args()

    output_path = merge_and_save(args.folder, args.ticker_map, args.output)
    print(f"병합 완료: {output_path}")


if __name__ == "__main__":
    main()
