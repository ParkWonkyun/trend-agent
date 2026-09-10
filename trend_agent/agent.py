from __future__ import annotations

import json
import operator as _operator
import os
from dataclasses import dataclass, field, fields
from typing import Any

import pandas as pd
import plotly.express as px
from openai import OpenAI

from .data import schema_summary

TOOL_NAME = "build_trend_chart"

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "사용자의 자연어 요청을 해석해 DataFrame으로부터 추세(trend) 차트를 그리기 위한 "
            "데이터 변환 명세를 만든다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date_column": {
                    "type": "string",
                    "description": "x축(시간축)으로 사용할 날짜/시간 컬럼명. 스키마에 실제로 존재해야 함",
                },
                "value_column": {
                    "type": "string",
                    "description": "y축으로 사용할 숫자형 컬럼명. 스키마에 실제로 존재해야 함",
                },
                "group_column": {
                    "type": ["string", "null"],
                    "description": "여러 계열로 나눠서 그릴 범주형 컬럼명. 필요 없으면 null",
                },
                "aggregation": {
                    "type": "string",
                    "enum": ["mean", "sum", "min", "max", "median", "count"],
                    "description": "같은 시점/그룹 내 값들을 집계하는 방식",
                },
                "resample_freq": {
                    "type": "string",
                    "enum": ["D", "W", "ME", "QE", "YE", "none"],
                    "description": (
                        "추세를 보기 위한 리샘플링 주기(D=일, W=주, ME=월, QE=분기, YE=연). "
                        "원본 그대로 쓰려면 none"
                    ),
                },
                "rolling_window": {
                    "type": ["integer", "null"],
                    "description": "이동평균 스무딩 윈도우 크기(리샘플링 후 포인트 개수 기준). 필요 없으면 null",
                },
                "filters": {
                    "type": "array",
                    "description": (
                        "차트를 그리기 전에 적용할 행 필터 조건 목록. "
                        "사용자가 특정 범주형 컬럼 값 여러 개를 동시에 보고 싶어하면(예: '종가와 20일 이동평균을 함께') "
                        "op를 'in'으로 쓰고 value를 문자열 배열로 준다."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "op": {
                                "type": "string",
                                "enum": ["==", "!=", ">", "<", ">=", "<=", "contains", "in", "not_in"],
                            },
                            "value": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "array", "items": {"type": "string"}},
                                ],
                                "description": "op가 'in'/'not_in'이면 문자열 배열, 그 외에는 단일 문자열",
                            },
                        },
                        "required": ["column", "op", "value"],
                    },
                },
                "start_date": {
                    "type": ["string", "null"],
                    "description": "YYYY-MM-DD 형식 시작일. 없으면 null",
                },
                "end_date": {
                    "type": ["string", "null"],
                    "description": "YYYY-MM-DD 형식 종료일. 없으면 null",
                },
                "chart_type": {
                    "type": "string",
                    "enum": ["line", "area"],
                },
                "title": {
                    "type": "string",
                    "description": "차트 제목 (한국어)",
                },
            },
            "required": [
                "date_column",
                "value_column",
                "aggregation",
                "resample_freq",
                "chart_type",
                "title",
            ],
        },
    },
}

_OPS = {
    "==": _operator.eq,
    "!=": _operator.ne,
    ">": _operator.gt,
    "<": _operator.lt,
    ">=": _operator.ge,
    "<=": _operator.le,
}


@dataclass
class PlotSpec:
    date_column: str
    value_column: str
    aggregation: str
    resample_freq: str
    chart_type: str
    title: str
    group_column: str | None = None
    rolling_window: int | None = None
    filters: list[dict[str, Any]] = field(default_factory=list)
    start_date: str | None = None
    end_date: str | None = None

    def __post_init__(self) -> None:
        if self.filters is None:
            self.filters = []

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlotSpec":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class TrendAgentError(RuntimeError):
    pass


def _apply_filter(df: pd.DataFrame, condition: dict[str, Any]) -> pd.DataFrame:
    col, op, value = condition["column"], condition["op"], condition["value"]
    series = df[col]

    if op == "contains":
        return df[series.astype(str).str.contains(str(value), case=False, na=False)]

    if op in ("in", "not_in"):
        values = value if isinstance(value, list) else [value]
        values = [str(v) for v in values]
        mask = series.astype(str).isin(values)
        return df[mask if op == "in" else ~mask]

    try:
        target = float(value)
        compare_series = pd.to_numeric(series, errors="coerce")
    except ValueError:
        target = value
        compare_series = series.astype(str)

    return df[_OPS[op](compare_series, target)]


class TrendAgent:
    """자연어 명령을 해석해 DataFrame의 추세 차트를 그리는 AI 에이전트."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def parse_command(self, df: pd.DataFrame, command: str) -> PlotSpec:
        schema = schema_summary(df)
        system = (
            "너는 pandas DataFrame의 추세(trend)를 시각화하도록 돕는 데이터 분석 에이전트다. "
            "사용자의 자연어 요청과 DataFrame 컬럼 스키마를 보고, "
            f"반드시 `{TOOL_NAME}` 함수를 호출해서 차트 명세를 반환하라. "
            "date_column, value_column, group_column, filters의 column 값은 "
            "반드시 스키마에 실제로 존재하는 컬럼명이어야 한다. "
            "사용자가 기간을 명시하지 않으면 start_date/end_date는 null로 둔다. "
            "하나의 value 컬럼 안에 서로 다른 종류의 지표(예: 종가, 이동평균 등)가 "
            "범주형 컬럼(예: series, color)의 값으로 섞여 있는 경우가 흔하다. "
            "사용자 요청에 등장한 단어가 그런 범주형 컬럼의 실제 값(스키마의 예시 값)과 일치하면, "
            "그 값들을 서로 합산/평균하지 않도록 반드시 filters에 해당 조건을 추가해라. "
            "예를 들어 사용자가 '종가'만 요청했는데 같은 컬럼에 '5일 이동평균', '20일 이동평균' 같은 "
            "다른 지표 값이 섞여 있다면, series == '종가' 필터를 추가해야 한다. "
            "특정 지표를 지정하지 않고 전체를 비교/평균하라고 명시한 경우에만 필터 없이 진행한다."
        )
        user = f"[DataFrame 컬럼 스키마]\n{schema}\n\n[사용자 요청]\n{command}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=[TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
            temperature=0,
        )

        message = response.choices[0].message
        if not message.tool_calls:
            raise TrendAgentError("모델이 차트 명세를 반환하지 않았습니다.")

        args = json.loads(message.tool_calls[0].function.arguments)
        spec = PlotSpec.from_dict(args)
        self._validate(df, spec)
        return spec

    def _validate(self, df: pd.DataFrame, spec: PlotSpec) -> None:
        for col in (spec.date_column, spec.value_column):
            if col not in df.columns:
                raise TrendAgentError(f"모델이 존재하지 않는 컬럼을 지정했습니다: {col}")
        if spec.group_column and spec.group_column not in df.columns:
            raise TrendAgentError(f"모델이 존재하지 않는 그룹 컬럼을 지정했습니다: {spec.group_column}")
        for condition in spec.filters:
            if condition.get("column") not in df.columns:
                raise TrendAgentError(f"모델이 존재하지 않는 필터 컬럼을 지정했습니다: {condition.get('column')}")

    def transform(self, df: pd.DataFrame, spec: PlotSpec) -> pd.DataFrame:
        out = df.copy()
        out[spec.date_column] = pd.to_datetime(out[spec.date_column], errors="coerce")
        out = out.dropna(subset=[spec.date_column])

        for condition in spec.filters:
            out = _apply_filter(out, condition)

        if spec.start_date:
            out = out[out[spec.date_column] >= pd.to_datetime(spec.start_date)]
        if spec.end_date:
            out = out[out[spec.date_column] <= pd.to_datetime(spec.end_date)]

        out[spec.value_column] = pd.to_numeric(out[spec.value_column], errors="coerce")
        out = out.dropna(subset=[spec.value_column])

        if out.empty:
            raise TrendAgentError("필터/기간 조건을 적용한 뒤 남은 데이터가 없습니다.")

        group_cols = [spec.group_column] if spec.group_column else []

        if spec.resample_freq != "none":
            grouper = [pd.Grouper(key=spec.date_column, freq=spec.resample_freq)] + group_cols
            out = out.groupby(grouper, dropna=False)[spec.value_column].agg(spec.aggregation).reset_index()
        elif group_cols:
            out = (
                out.groupby([spec.date_column] + group_cols, dropna=False)[spec.value_column]
                .agg(spec.aggregation)
                .reset_index()
            )

        out = out.sort_values(spec.date_column)

        if spec.rolling_window and spec.rolling_window > 1:
            if group_cols:
                out[spec.value_column] = out.groupby(group_cols)[spec.value_column].transform(
                    lambda s: s.rolling(spec.rolling_window, min_periods=1).mean()
                )
            else:
                out[spec.value_column] = out[spec.value_column].rolling(
                    spec.rolling_window, min_periods=1
                ).mean()

        return out

    def build_chart(self, transformed: pd.DataFrame, spec: PlotSpec):
        plot_fn = px.area if spec.chart_type == "area" else px.line
        fig = plot_fn(
            transformed,
            x=spec.date_column,
            y=spec.value_column,
            color=spec.group_column if spec.group_column else None,
            title=spec.title,
            markers=True,
        )
        fig.update_layout(hovermode="x unified")
        return fig

    def run(self, df: pd.DataFrame, command: str):
        spec = self.parse_command(df, command)
        transformed = self.transform(df, spec)
        fig = self.build_chart(transformed, spec)
        return spec, transformed, fig
