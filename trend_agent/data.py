from __future__ import annotations

import pandas as pd


def load_csv(path_or_buffer) -> pd.DataFrame:
    """CSV를 읽어 pandas DataFrame으로 변환한다.

    pandas/streamlit이 내보낸 CSV에 흔히 붙는 인덱스 컬럼(Unnamed: 0)은 제거한다.
    """
    df = pd.read_csv(path_or_buffer)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def schema_summary(df: pd.DataFrame, sample_values: int = 5) -> str:
    """LLM에게 넘길 컬럼 스키마 요약 텍스트를 만든다."""
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = df[col].dropna().unique()[:sample_values].tolist()
        lines.append(f"- {col} ({dtype}), 예시 값: {sample}")
    return "\n".join(lines)
