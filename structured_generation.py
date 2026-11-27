"""Experimental two-stage generation shared by Streamlit and the CLI."""

import json
import time

from generator import BASE_DIR, call_model, generate_docent, load_artwork, render_template, sum_tokens
from selector import compact_relation, compact_unit
from structured_candidates import build_candidates, resolve_selection


VARIANTS = {
    "STRUCTURED_PAIR": {"max_relations": 1},
    "STRUCTURED_MULTI": {"max_relations": 2},
    "STRUCTURED_CU": {"use_relations": False},
}


def load_structured_prompt(stage):
    return (BASE_DIR / "prompts" / f"structured_{stage}.txt").read_text(encoding="utf-8")


def compact_bundle(bundle):
    return {
        "core_unit": compact_unit(bundle["core_unit"]),
        "related_units": [compact_unit(unit) for unit in bundle["related_units"]],
        "relations": [compact_relation(edge) for edge in bundle["relations"]],
    }


def parse_plan(text, candidates):
    plan = json.loads(text)
    choices = plan.get("choices") if isinstance(plan, dict) else None
    if not isinstance(choices, list) or any(
        not isinstance(choice, dict)
        or any(
            not isinstance(choice.get(key), str) or not choice[key].strip()
            for key in ("bundle_id", "reason", "focus")
        )
        for choice in choices
    ):
        raise ValueError("잘못된 후보 선정 형식입니다.")
    resolved = resolve_selection([choice["bundle_id"] for choice in choices], candidates)
    return [
        {
            **resolved[choice["bundle_id"]],
            "bundle_id": choice["bundle_id"],
            "topic": resolved[choice["bundle_id"]]["core_unit"]["content"],
            "selection_reason": choice["reason"],
            "observation_focus": choice["focus"],
        }
        for choice in choices
    ]


def generate_structured(
    version, model, artwork_name, why, what, how, context, target_length,
    temperature=0.7, template=None, selection_template=None,
):
    start = time.perf_counter()
    metadata, _ = load_artwork(artwork_name)
    candidates = build_candidates(artwork_name, what, **VARIANTS[version])
    profile = {"why": why, "what": what, "how": how, "context": context.strip()}
    selection_prompt = render_template(
        selection_template if selection_template is not None else load_structured_prompt("selection"),
        {"selection_input": {
            "artwork": metadata,
            "profile": profile,
            "target_length": target_length,
            "candidates": {key: compact_bundle(bundle) for key, bundle in candidates.items()},
        }},
    )
    selection = call_model(model, selection_prompt, temperature=0.2, json_output=True)
    bundles = parse_plan(selection["text"], candidates)
    prompt = render_template(
        template if template is not None else load_structured_prompt("generation"),
        {
            "artwork_metadata": metadata,
            "profile": profile,
            "target_length": target_length,
            "knowledge_bundles": [
                {
                    **compact_bundle(bundle),
                    "selection_reason": bundle["selection_reason"],
                    "observation_focus": bundle["observation_focus"],
                }
                for bundle in bundles
            ],
        },
    )
    result = generate_docent(model, prompt, temperature=temperature, version=version)
    measurements = (
        "input_tokens", "output_tokens", "thinking_tokens", "total_tokens",
        "elapsed_seconds", "wait_seconds",
    )
    result.update(
        bundles=bundles,
        candidates=candidates,
        selection_prompt=selection_prompt,
        raw_selection=json.loads(selection["text"]),
        selection_usage=selection,
        generation_usage={key: result[key] for key in measurements},
        api_calls=2,
        elapsed_seconds=round(time.perf_counter() - start, 2),
    )
    for key in (*measurements[:4], "wait_seconds"):
        result[key] = sum_tokens(selection[key], result[key])
    return result
