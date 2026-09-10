from __future__ import annotations

import os
import traceback
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from trend_agent.agent import TrendAgent, TrendAgentError
from trend_agent.data import load_csv, schema_summary
from trend_agent.disambiguate import apply_resolution, find_ambiguous_filters
from trend_agent.merge import DEFAULT_TICKER_MAP_PATH, load_ticker_map, merge_csv_folder

load_dotenv()

DEFAULT_RAW_DATA_DIR = Path(__file__).resolve().parent.parent  # .../raw data

st.set_page_config(page_title="CSV Trend Agent", layout="wide")
st.title("CSV Trend Agent")
st.caption("CSV를 업로드하고, 자연어로 원하는 추세(trend) 차트를 요청하세요.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("설정")
    model = st.text_input("OpenAI 모델", value=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    api_key_input = st.text_input(
        "OPENAI_API_KEY (.env에 설정했다면 비워두세요)", type="password", value=""
    )

st.subheader("데이터 불러오기")
source_mode = st.radio(
    "데이터 소스",
    ["여러 CSV 자동 병합 (종목별)", "CSV 파일 직접 업로드"],
    horizontal=True,
)

df = None

if source_mode == "여러 CSV 자동 병합 (종목별)":
    folder = st.text_input("원본 CSV 폴더", value=str(DEFAULT_RAW_DATA_DIR))
    ticker_map_path = st.text_input("파일명→종목명 매핑 JSON", value=str(DEFAULT_TICKER_MAP_PATH))
    if st.button("병합하기"):
        try:
            ticker_map = load_ticker_map(ticker_map_path)
            st.session_state.merged_df = merge_csv_folder(folder, ticker_map)
        except Exception as e:  # noqa: BLE001
            st.error(f"병합에 실패했습니다: {e}")
    df = st.session_state.get("merged_df")
    if df is None:
        st.info("`병합하기` 버튼을 눌러 raw CSV들을 하나로 합쳐주세요.")
        st.stop()
else:
    uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"])
    if uploaded is None:
        st.info("CSV 파일을 업로드하면 시작할 수 있습니다.")
        st.stop()
    df = load_csv(uploaded)

st.subheader("데이터 미리보기")
st.dataframe(df.head(20), use_container_width=True)
with st.expander("컬럼 스키마"):
    st.text(schema_summary(df))

command = st.text_input(
    "어떤 추세를 보고 싶으신가요?",
    placeholder="예: 월별 평균 종가 추이를 선 그래프로 보여줘",
)


def _render_result(spec, transformed, fig) -> None:
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("에이전트가 해석한 차트 명세"):
        st.json(spec.__dict__)
    with st.expander("변환된 데이터"):
        st.dataframe(transformed, use_container_width=True)


def _resolve_api_key() -> str | None:
    return api_key_input or os.environ.get("OPENAI_API_KEY")


if st.button("차트 그리기", type="primary") and command:
    api_key = _resolve_api_key()
    if not api_key:
        st.error("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일에 추가하거나 사이드바에 입력하세요.")
        st.stop()

    try:
        agent = TrendAgent(model=model, api_key=api_key)
        with st.spinner("명령을 해석하는 중..."):
            spec = agent.parse_command(df, command)

        ambiguous = find_ambiguous_filters(df, spec, command)
        if ambiguous:
            st.session_state.pending = {"command": command, "spec": spec, "ambiguous": ambiguous}
        else:
            transformed = agent.transform(df, spec)
            fig = agent.build_chart(transformed, spec)
            st.session_state.history.append(command)
            _render_result(spec, transformed, fig)

    except TrendAgentError as e:
        st.error(str(e))
    except Exception as e:  # noqa: BLE001
        st.error(f"차트를 생성하지 못했습니다: {e}")
        with st.expander("상세 오류"):
            st.code(traceback.format_exc())

def _current_values_for(spec, column: str, term: str) -> list[str]:
    for condition in spec.filters:
        if condition["column"] != column:
            continue
        current_values = condition["value"] if isinstance(condition["value"], list) else [condition["value"]]
        current_values = [str(v) for v in current_values]
        if any(term in v or v in term for v in current_values):
            return current_values
    return []


pending = st.session_state.get("pending")
if pending is not None:
    st.warning(
        f"'{pending['command']}' 요청에 이름이 비슷한 값이 있습니다. "
        "어떤 값을 의미하는지 확인해주세요."
    )

    chosen_by_item: dict[int, list[str]] = {}
    for i, item in enumerate(pending["ambiguous"]):
        default = [c for c in item["candidates"] if c in _current_values_for(pending["spec"], item["column"], item["picked"])]
        chosen_by_item[i] = st.multiselect(
            f"[{item['column']}] '{item['picked']}' 은(는) 다음 중 무엇을 의미하나요? (복수 선택 가능)",
            item["candidates"],
            default=default or item["candidates"][:1],
            key=f"disambig_{i}_{item['column']}_{item['picked']}",
        )

    confirm_col, cancel_col = st.columns(2)
    confirm = confirm_col.button("확인하고 차트 그리기", type="primary")
    cancel = cancel_col.button("취소")

    if cancel:
        del st.session_state["pending"]
        st.rerun()

    if confirm and any(not v for v in chosen_by_item.values()):
        st.error("각 항목에서 최소 하나 이상의 값을 선택해주세요.")
    elif confirm:
        try:
            resolutions = {
                (item["column"], item["picked"]): chosen_by_item[i]
                for i, item in enumerate(pending["ambiguous"])
            }
            spec = apply_resolution(pending["spec"], resolutions)

            api_key = _resolve_api_key()
            agent = TrendAgent(model=model, api_key=api_key)
            transformed = agent.transform(df, spec)
            fig = agent.build_chart(transformed, spec)

            st.session_state.history.append(pending["command"])
            del st.session_state["pending"]
            _render_result(spec, transformed, fig)

        except TrendAgentError as e:
            st.error(str(e))
        except Exception as e:  # noqa: BLE001
            st.error(f"차트를 생성하지 못했습니다: {e}")
            with st.expander("상세 오류"):
                st.code(traceback.format_exc())

if st.session_state.history:
    with st.sidebar:
        st.subheader("이전 요청")
        for h in reversed(st.session_state.history[-10:]):
            st.caption(f"• {h}")
