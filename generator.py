import os
import json
import time
import re
from pathlib import Path
from contextvars import ContextVar

from dotenv import load_dotenv
from google import genai
from google.genai import types

from selector import (
    select_knowledge,
    validate_bundles,
)

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

ARTWORKS = [
    "The Bedroom",
    "The Starry Night",
]

SELECTION_MODEL = "gemini-3.5-flash-lite"

LAST_CALL = {}
CALL_OBSERVER = ContextVar("docent_call_observer", default=None)


def load_artwork(artwork_name):
    if artwork_name not in ARTWORKS:
        raise ValueError("지원하지 않는 작품입니다.")

    folder = BASE_DIR / "data" / artwork_name

    metadata = json.loads(
        (folder / "artwork.json").read_text(
            encoding="utf-8-sig"
        )
    )

    description = (
        folder / "description.txt"
    ).read_text(encoding="utf-8-sig")

    return metadata, description


def load_prompt(version="V0"):
    filename = {
        "V0": "v0.txt",
        "V1": "v1.txt",
        "SELECTION": "selection.txt",
    }[version.upper()]

    return (
        BASE_DIR / "prompts" / filename
    ).read_text(encoding="utf-8-sig")


def render_template(template, values):

    def replace(match):
        key = match.group(1)

        if key not in values:
            raise ValueError(
                f"Missing prompt variable: {key}"
            )

        value = values[key]

        if isinstance(value, (dict, list)):
            return json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
            )

        return str(value)

    return re.sub(
        r"\{\{(\w+)\}\}",
        replace,
        template,
    )


def build_prompt(
    artwork_name,
    why,
    what,
    how,
    context,
    target_length,
    template=None,
    version="V0",
    bundles=None,
):
    metadata, description = load_artwork(
        artwork_name
    )

    if template is None:
        template = load_prompt(version)

    values = {
        "artwork_metadata": metadata,
        "raw_description": description,
        "knowledge_bundles": bundles or [],
        "why": why,
        "what": what,
        "how": how,
        "context": context.strip() or "없음",
        "target_length": target_length,
    }

    return render_template(
        template,
        values,
    )


def call_model(
    model,
    prompt,
    temperature=0.7,
    json_output=False,
):
    if model not in MODELS:
        raise ValueError("지원하지 않는 모델입니다.")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            ".env에 GEMINI_API_KEY를 설정하세요."
        )

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
    )

    config = {
        "temperature": temperature,
    }

    if json_output:
        config["response_mime_type"] = (
            "application/json"
        )

    # 연속 API 호출 간격 확보
    interval = (
        5 if "lite" in model else 15
    )

    previous = LAST_CALL.get(model)

    wait_start = time.perf_counter()

    if previous is not None:
        remaining = interval - (
            time.monotonic() - previous
        )

        if remaining > 0:
            time.sleep(remaining)

    wait_seconds = (
        time.perf_counter() - wait_start
    )

    observer = CALL_OBSERVER.get()
    finish = None
    start = time.perf_counter()
    try:
        if observer:
            finish = observer({
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "json_output": json_output,
                "wait_seconds": round(wait_seconds, 2),
            })
        start = time.perf_counter()
        LAST_CALL[model] = time.monotonic()
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                **config
            ),
        )

        elapsed = time.perf_counter() - start

        text = (response.text or "").strip()

        if not text:
            raise RuntimeError("생성된 텍스트가 없습니다.")

        usage = response.usage_metadata

        result = {
            "model": model,
            "text": text,
            "elapsed_seconds": round(elapsed, 2),
            "wait_seconds": round(wait_seconds, 2),
            "input_tokens": getattr(
                usage, "prompt_token_count", None
            ),
            "output_tokens": getattr(
                usage, "candidates_token_count", None
            ),
            "thinking_tokens": getattr(
                usage, "thoughts_token_count", None
            ),
            "total_tokens": getattr(
                usage, "total_token_count", None
            ),
        }

    except Exception as error:
        if finish:
            finish({
                "status": "failed",
                "error_type": type(error).__name__,
                "http_status": getattr(error, "code", None),
                "elapsed_seconds": round(time.perf_counter() - start, 2),
            })
        raise
    finally:
        client.close()

    if finish:
        finish({**result, "status": "success"})
    return result


def generate_docent(
    model,
    prompt,
    temperature=0.7,
    version="V0",
):
    response = call_model(
        model=model,
        prompt=prompt,
        temperature=temperature,
    )

    return {
        "model": model,
        "version": version,
        "docent": response["text"],
        "elapsed_seconds": response["elapsed_seconds"],
        "character_count": len(response["text"]),
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
        "thinking_tokens": response["thinking_tokens"],
        "total_tokens": response["total_tokens"],
        "wait_seconds": response["wait_seconds"],
        "full_prompt": prompt,
    }


def select_bundles(
    artwork_name,
    what,
    target_length,
    template=None,
):
    metadata, _ = load_artwork(artwork_name)

    candidates = select_knowledge(
        artwork_name,
        what,
    )

    if template is None:
        template = load_prompt("SELECTION")

    prompt = render_template(
        template,
        {
            "artwork_metadata": metadata,
            "what": what,
            "target_length": target_length,
            "core_units": candidates["core_units"],
            "supporting_units": (
                candidates["supporting_units"]
            ),
            "relations": candidates["relations"],
        },
    )

    response = call_model(
        model=SELECTION_MODEL,
        prompt=prompt,
        temperature=0.2,
        json_output=True,
    )

    try:
        selection = json.loads(
            response["text"]
        )

        bundles = validate_bundles(
            selection,
            candidates,
        )

    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError(
            "Knowledge Bundle 검증 실패: "
            + str(error)
            + "\nRaw response:\n"
            + response["text"]
        ) from error

    return {
        "bundles": bundles,
        "candidate_statistics": (
            candidates["statistics"]
        ),
        "selection_prompt": prompt,
        "raw_selection": selection,
        "selection_usage": response,
    }


def sum_tokens(*values):
    if any(v is None for v in values):
        return None

    return sum(values)


def generate_v1(
    model,
    artwork_name,
    why,
    what,
    how,
    context,
    target_length,
    temperature=0.7,
    template=None,
    selection_template=None,
):
    start = time.perf_counter()

    selected = select_bundles(
        artwork_name=artwork_name,
        what=what,
        target_length=target_length,
        template=selection_template,
    )

    prompt = build_prompt(
        artwork_name=artwork_name,
        why=why,
        what=what,
        how=how,
        context=context,
        target_length=target_length,
        template=template,
        version="V1",
        bundles=selected["bundles"],
    )

    result = generate_docent(
        model=model,
        prompt=prompt,
        temperature=temperature,
        version="V1",
    )

    selection_usage = selected[
        "selection_usage"
    ]

    generation_usage = {
        key: result[key]
        for key in [
            "elapsed_seconds",
            "input_tokens",
            "output_tokens",
            "thinking_tokens",
            "total_tokens",
            "wait_seconds",
        ]
    }

    result["generation_usage"] = generation_usage
    result["selection_usage"] = selection_usage

    result["bundles"] = selected["bundles"]

    result["candidate_statistics"] = (
        selected["candidate_statistics"]
    )

    result["selection_prompt"] = (
        selected["selection_prompt"]
    )

    result["raw_selection"] = (
        selected["raw_selection"]
    )

    for key in [
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ]:
        result[key] = sum_tokens(
            selection_usage.get(key),
            generation_usage.get(key),
        )

    result["thinking_tokens"] = sum_tokens(
        selection_usage.get("thinking_tokens"),
        generation_usage.get("thinking_tokens"),
    )

    result["elapsed_seconds"] = round(
        time.perf_counter() - start,
        2,
    )

    result["api_calls"] = 2

    return result