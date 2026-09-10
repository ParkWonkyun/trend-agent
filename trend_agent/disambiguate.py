from __future__ import annotations

import re

import pandas as pd

from .agent import PlotSpec

_TOKEN_SPLIT_RE = re.compile(r"[\s,/·]+")
_MIN_TOKEN_LEN = 2


def _tokenize(command: str) -> list[str]:
    return [t for t in _TOKEN_SPLIT_RE.split(command) if len(t) >= _MIN_TOKEN_LEN]


def _filter_columns(spec: PlotSpec) -> set[str]:
    columns = {condition["column"] for condition in spec.filters}
    if spec.group_column:
        columns.add(spec.group_column)
    return columns


def find_ambiguous_filters(df: pd.DataFrame, spec: PlotSpec, command: str) -> list[dict]:
    """spec이 사용하는 범주형 컬럼에서, 사용자 명령이 여러 실제 값에 걸쳐 있는 표현을 썼는지 확인한다.

    두 가지 경우를 모두 잡는다.
    1. '삼성' 처럼 여러 값의 일부(접두어 등)만 언급한 경우 -> '삼성전자'/'삼성전자우'
    2. '이동평균' 처럼 여러 값에 공통으로 들어있는 부분 표현만 언급한 경우 -> '5일 이동평균'/'20일 이동평균'

    명령에 값 전체가 정확히(하나만) 등장하면 확실한 것으로 보고 확인을 건너뛴다.
    """
    ambiguous: list[dict] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    lowered_command = command.lower()
    tokens = _tokenize(command)

    for column in _filter_columns(spec):
        if column not in df.columns:
            continue
        uniques = [str(v) for v in df[column].dropna().unique().tolist()]

        full_matches = [v for v in uniques if v.lower() in lowered_command]
        if len(full_matches) >= 1:
            # 값 전체가 그대로 언급됨: 정확히 하나든 여러 개(명시적 다중 선택)든 확실한 것으로 본다.
            continue

        for token in tokens:
            matched_values = sorted({v for v in uniques if token in v or v in token})
            if len(matched_values) < 2:
                continue
            key = (column, tuple(matched_values))
            if key in seen:
                continue
            seen.add(key)
            ambiguous.append({"column": column, "picked": token, "candidates": matched_values})

    return ambiguous


def apply_resolution(spec: PlotSpec, resolutions: dict[tuple[str, str], list[str]]) -> PlotSpec:
    """(column, picked_term) -> 사용자가 확정한 값 목록 매핑을 spec.filters에 반영한다."""
    for condition in spec.filters:
        column = condition["column"]
        current_values = condition["value"] if isinstance(condition["value"], list) else [condition["value"]]
        is_negative = condition["op"] in ("!=", "not_in")

        for (res_column, term), chosen_values in resolutions.items():
            if res_column != column:
                continue
            if any(term in str(v) or str(v) in term for v in current_values):
                condition["value"] = list(chosen_values)
                condition["op"] = "not_in" if is_negative else "in"
    return spec
