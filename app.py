import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from generator import (
    MODELS,
    ARTWORKS,
    SELECTION_MODEL,
    load_prompt,
    build_prompt,
    generate_docent,
    generate_v1,
)

from selector import select_knowledge
from structured_generation import VARIANTS, generate_structured, load_structured_prompt

VERSION_LABELS = {
    "V0": "원문 기반 직접 생성",
    "V1": "기존 관계 기반 생성",
    "STRUCTURED_PAIR": "구조화 후보 · 단일 관계",
    "STRUCTURED_MULTI": "구조화 후보 · 복수 관계",
    "STRUCTURED_CU": "구조화 후보 · CU만",
}

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Docent Generation Lab",
    layout="wide",
)

st.title("Docent Generation Lab")

WHY = {
    "DISCOVERY": "발견",
    "RELAXATION": "휴식",
    "REFLECTION": "성찰·영감",
    "SOCIAL": "동반 관람",
}

WHAT = {
    "VISUAL": "시각적 특징",
    "MEANING": "의미와 해석",
    "TECHNIQUE": "재료와 제작 기법",
    "BACKGROUND": "작가 및 제작 배경",
}

HOW = {
    "BASIC": "쉽고 직관적인 설명",
    "ADVANCED": "전문적이고 심층적인 설명",
}

if "results" not in st.session_state:
    st.session_state.results = []


with st.sidebar:

    st.header("Artwork")

    artwork = st.selectbox(
        "작품 선택",
        ARTWORKS,
    )

    st.header("User Profile")

    why = st.radio(
        "WHY | 관람 목적",
        list(WHY),
        format_func=lambda x: WHY[x],
    )

    what = st.radio(
        "WHAT | 정보 선호",
        list(WHAT),
        format_func=lambda x: WHAT[x],
    )

    how = st.radio(
        "HOW | 설명 수준",
        list(HOW),
        format_func=lambda x: HOW[x],
    )

    context = st.text_area(
        "CONTEXT | 개인적 상황",
        placeholder="새로운 시작을 앞두고 있어요.",
    )

    st.header("Generation Settings")

    model = st.selectbox(
        "생성 모델",
        MODELS,
        index=MODELS.index(
            "gemini-3.5-flash-lite"
        ),
    )

    versions = st.multiselect(
        "비교할 생성 방식",
        list(VERSION_LABELS),
        default=["V0", "V1"],
        format_func=VERSION_LABELS.get,
    )

    temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.7,
        0.1,
    )

    target_length = st.number_input(
        "목표 글자 수",
        min_value=200,
        max_value=1500,
        value=500,
        step=100,
    )

    call_count = sum(
        1 if v == "V0" else 2
        for v in versions
    )

    st.caption(
        f"예상 API 호출: {call_count}회"
    )

    if "V1" in versions:
        st.caption(
            f"정보 선정 모델: {SELECTION_MODEL}"
        )


st.subheader("User Profile")

p1, p2, p3, p4 = st.columns(4)

p1.metric("WHY", WHY[why])
p2.metric("WHAT", WHAT[what])
p3.metric("HOW", how)
p4.metric(
    "CONTEXT",
    "있음" if context.strip() else "없음",
)

if context.strip():
    st.write(context)


st.subheader("Relation Candidates")

try:
    candidates = select_knowledge(
        artwork,
        what,
    )

    stats = candidates["statistics"]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Core 후보",
        stats["core_count"],
    )

    c2.metric(
        "Related 후보",
        stats["supporting_count"],
    )

    c3.metric(
        "Relation 후보",
        stats["candidate_relation_count"],
    )

    with st.expander("Core Unit 후보"):

        for unit in candidates["core_units"]:
            st.markdown(
                f"**{unit['id']}**"
            )
            st.write(unit["content"])
            st.caption(
                " / ".join(unit["categories"])
            )

    with st.expander("Related Unit 후보"):

        for unit in candidates["supporting_units"]:
            st.markdown(
                f"**{unit['id']}**"
            )
            st.write(unit["content"])

    with st.expander("Semantic Relation 후보"):

        for relation in candidates["relations"]:
            st.markdown(
                f"**{relation['id']}**"
            )

            st.caption(
                f"{relation['source_unit_id']} → "
                f"{relation['target_unit_id']}"
            )

            st.write(relation["description"])

            st.caption(
                f"{relation['relation_type']} | "
                f"{relation['grounding_type']}"
            )

except Exception as error:
    st.warning(
        f"관계 데이터 로딩 실패: {error}"
    )


st.subheader("Prompt Settings")

tab0, tab1, tab2, tab3 = st.tabs([
    "V0 Prompt",
    "V1 Prompt",
    "V1 Selection Prompt",
    "구조화 후보 프롬프트",
])

with tab0:
    v0_template = st.text_area(
        "V0 프롬프트",
        value=load_prompt("V0"),
        height=300,
    )

with tab1:
    v1_template = st.text_area(
        "V1 프롬프트",
        value=load_prompt("V1"),
        height=300,
    )

with tab2:
    selection_template = st.text_area(
        "정보 선정 프롬프트",
        value=load_prompt("SELECTION"),
        height=300,
    )


with tab3:
    structured_selection = st.text_area(
        "구조화 후보 선택 및 관찰 계획", value=load_structured_prompt("selection"), height=260
    )
    structured_generation = st.text_area(
        "구조화 후보 도슨트 생성", value=load_structured_prompt("generation"), height=260
    )


if st.button(
    "Generate & Compare",
    type="primary",
    disabled=not versions,
):
    results = []

    for version in versions:

        with st.spinner(
            f"{version} 생성 중..."
        ):
            try:
                if version == "V0":

                    prompt = build_prompt(
                        artwork_name=artwork,
                        why=why,
                        what=what,
                        how=how,
                        context=context,
                        target_length=target_length,
                        template=v0_template,
                        version="V0",
                    )

                    result = generate_docent(
                        model=model,
                        prompt=prompt,
                        temperature=temperature,
                        version="V0",
                    )

                    result["api_calls"] = 1

                elif version == "V1":

                    result = generate_v1(
                        model=model,
                        artwork_name=artwork,
                        why=why,
                        what=what,
                        how=how,
                        context=context,
                        target_length=target_length,
                        temperature=temperature,
                        template=v1_template,
                        selection_template=(
                            selection_template
                        ),
                    )

                else:
                    result = generate_structured(
                        version, model, artwork, why, what, how, context, target_length,
                        temperature=temperature,
                        template=structured_generation,
                        selection_template=structured_selection,
                    )

            except Exception as error:

                result = {
                    "model": model,
                    "version": version,
                    "error": f"생성을 완료하지 못했습니다 ({type(error).__name__}).",
                }

            results.append(result)

    experiment = {
        "timestamp": datetime.now().isoformat(),
        "artwork": artwork,
        "model": model,
        "versions": versions,
        "profile": {
            "why": why,
            "what": what,
            "how": how,
            "context": context,
        },
        "settings": {
            "temperature": temperature,
            "target_length": target_length,
            "selection_model": SELECTION_MODEL,
            "selection_temperature": 0.2,
        },
        "prompt_templates": {
            "V0": v0_template,
            "V1": v1_template,
            "SELECTION": selection_template,
            "STRUCTURED_SELECTION": structured_selection,
            "STRUCTURED_GENERATION": structured_generation,
        },
        "results": results,
    }

    st.session_state.results = results
    st.session_state.experiment = experiment

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    filename = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    path = results_dir / f"{filename}.json"

    path.write_text(
        json.dumps(
            experiment,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


st.divider()
st.subheader("Generation Results")


def show_bundle(bundle):

    with st.container(border=True):

        st.markdown(
            f"#### {bundle['bundle_id']} | "
            f"{bundle['topic']}"
        )

        core = bundle["core_unit"]

        st.markdown("**Core Unit**")

        st.info(
            f"{core['id']}\n\n"
            f"{core['content']}"
        )

        st.caption(
            " / ".join(core["categories"])
        )

        st.markdown("**Related Units**")

        if not bundle["related_units"]:
            st.caption("독립 정보")

        for unit in bundle["related_units"]:

            st.markdown(
                f"**{unit['id']}**"
            )

            st.write(unit["content"])

            st.caption(
                " / ".join(unit["categories"])
            )

        st.markdown("**Semantic Relations**")

        for relation in bundle["relations"]:

            st.caption(
                f"{relation['source_unit_id']} → "
                f"{relation['target_unit_id']}"
            )

            st.markdown(
                f"**{relation['relation_type']}**"
            )

            st.write(relation["description"])

            st.caption(
                relation["grounding_type"]
            )

        st.markdown("**선정 이유**")

        st.write(bundle["selection_reason"])
        if bundle.get("observation_focus"):
            st.write(bundle["observation_focus"])
        with st.expander("CU 및 관계 원문 근거"):
            st.json({
                record["id"]: record.get("evidence", record.get("source_refs", []))
                for record in [core, *bundle["related_units"], *bundle["relations"]]
            })


if st.session_state.results:

    experiment = st.session_state.experiment

    st.caption(
        f"작품: {experiment['artwork']} | "
        f"프로필: {experiment['profile']}"
    )

    results = st.session_state.results

    columns = st.columns(len(results))

    for column, result in zip(columns, results):

        with column:

            with st.container(border=True):

                version = result["version"]

                st.markdown(f"### {version}")

                st.caption(VERSION_LABELS[version])

                st.caption(result["model"])

                if "error" in result:
                    st.error(result["error"])
                    continue

                st.write(result["docent"])

                st.divider()

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "응답 시간",
                    f"{result['elapsed_seconds']}초",
                )

                m2.metric(
                    "글자 수",
                    result["character_count"],
                )

                m3.metric(
                    "전체 토큰",
                    result["total_tokens"],
                )

                st.caption(
                    f"API 호출: {result['api_calls']}회"
                )

                with st.expander("전체 API 사용량"):
                    st.json({
                        key: result.get(key)
                        for key in ("input_tokens", "output_tokens", "thinking_tokens", "wait_seconds")
                    })

                if "bundles" in result:

                    st.divider()

                    st.markdown(
                        "#### Selected Knowledge Bundles"
                    )

                    bundles = result["bundles"]

                    unit_count = sum(
                        1 + len(b["related_units"])
                        for b in bundles
                    )

                    relation_count = sum(
                        len(b["relations"])
                        for b in bundles
                    )

                    st.caption(
                        f"Bundle {len(bundles)}개 | "
                        f"Unit {unit_count}개 | "
                        f"Relation {relation_count}개"
                    )

                    for bundle in bundles:
                        show_bundle(bundle)

                    with st.expander(
                        "단계별 API 사용량"
                    ):
                        st.json({
                            "selection": {
                                k: v
                                for k, v in result[
                                    "selection_usage"
                                ].items()
                                if k != "text"
                            },
                            "generation": result[
                                "generation_usage"
                            ],
                        })

                    with st.expander(
                        "실제 Selection Prompt"
                    ):
                        st.code(
                            result["selection_prompt"],
                            language="text",
                        )

                    with st.expander(
                        "LLM 선정 원본 JSON"
                    ):
                        st.json(
                            result["raw_selection"]
                        )

                if version in VARIANTS:
                    with st.expander("모델에 제공한 후보와 근거"):
                        st.json(result["candidates"], expanded=False)

                with st.expander(
                    "실제 Generation Prompt"
                ):
                    st.code(
                        result["full_prompt"],
                        language="text",
                    )

    st.download_button(
        "실험 결과 JSON 다운로드",
        data=json.dumps(
            experiment,
            ensure_ascii=False,
            indent=2,
        ),
        file_name="docent_experiment.json",
        mime="application/json",
    )

else:
    st.info(
        "작품과 프로필을 선택하고 "
        "생성 버튼을 눌러주세요."
    )