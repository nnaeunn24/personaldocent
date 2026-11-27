"""Run one development/validation batch with a persistent Gemini call budget."""

import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from generator import (
    ARTWORKS, BASE_DIR, CALL_OBSERVER, MODELS, build_prompt, generate_docent, generate_v1,
)
from structured_generation import VARIANTS, generate_structured

VERSIONS = ("V0", "V1", *VARIANTS)
COST = {version: 1 if version == "V0" else 2 for version in VERSIONS}
DEFAULT_MODEL = "gemini-3.5-flash-lite"
TARGET = 500
TEMPERATURE = 0.7
CALL_LIMIT = 30
LENGTH_TOLERANCE = 0.10
WEIGHTS = {
    "Factual Grounding": 20, "Artwork Specificity": 15, "WHAT Alignment": 15,
    "WHY Alignment": 10, "HOW Alignment": 10, "Contextual Personalization": 10,
    "Explanatory Coherence": 10, "Naturalness": 5, "Semantic Diversity": 5,
}
GATES = (
    "material_facts", "attributed_intent_and_psychology", "correct_artwork_and_version",
    "valid_units_and_relations", "profile_respected", "context_not_forced",
)
DEVELOPMENT = {
    "anchor": dict(why="DISCOVERY", what="VISUAL", how="BASIC", context=""),
    "what": {"what": "TECHNIQUE"},
    "why": {"why": "SOCIAL"},
    "how": {"how": "ADVANCED"},
    "context": {"context": "익숙했던 공간을 떠나 새 환경에 적응하고 있어요."},
    "context_second": {"context": "요즘 바빠서 잠시 여유를 가지고 싶어요."},
}
HOLDOUT = (
    dict(why="RELAXATION", what="BACKGROUND", how="BASIC", context=""),
    dict(why="REFLECTION", what="MEANING", how="ADVANCED", context="오래 기억에 남는 풍경을 떠올리고 있어요."),
)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class CallBudget:
    def __init__(self, directory, limit=CALL_LIMIT):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.limit = limit

    @property
    def remaining(self):
        return max(0, self.limit - len(list(self.directory.glob("*.reserved"))))

    def observer(self, case_id):
        def reserve(settings):
            for index in range(1, self.limit + 1):
                slot = self.directory / f"{index:03d}.reserved"
                try:
                    with slot.open("x", encoding="utf-8") as file:
                        json.dump({
                            **settings, "case_id": case_id, "started_at": timestamp(),
                        }, file, ensure_ascii=False)
                except FileExistsError:
                    continue

                def finish(result):
                    write_json(slot.with_suffix(".json"), {
                        **result, "case_id": case_id, "finished_at": timestamp(),
                    })

                return finish
            raise RuntimeError("API 호출 예산을 모두 사용했습니다.")

        return reserve


def make_cases(phase, versions, model, axes=None, artwork_filter=None):
    if phase == "baseline":
        profiles = [("anchor", DEVELOPMENT["anchor"])] * len(ARTWORKS)
    elif phase == "holdout":
        profiles = [(f"holdout_{i}", profile) for i, profile in enumerate(HOLDOUT, 1)]
    else:
        selected_axes = axes or ("what", "why", "how", "context")
        profiles = [
            [(axis, {**DEVELOPMENT["anchor"], **DEVELOPMENT[axis]}) for axis in selected_axes]
        ] * len(ARTWORKS)
    for artwork, entry in zip(ARTWORKS, profiles):
        if artwork_filter and artwork != artwork_filter:
            continue
        artwork_profiles = entry if phase == "axes" else [entry]
        for label, profile in artwork_profiles:
            for version in versions:
                yield {
                    "phase": phase, "profile_id": label, "artwork": artwork,
                    "version": version, "model": model, "profile": profile,
                    "target_length": TARGET, "temperature": TEMPERATURE,
                }


def generate_case(case):
    version = case["version"]
    model = case["model"]
    temperature = case["temperature"]
    kwargs = dict(
        artwork_name=case["artwork"], target_length=case["target_length"], **case["profile"]
    )
    if version == "V0":
        result = generate_docent(model, build_prompt(**kwargs), temperature=temperature)
        result["api_calls"] = 1
        return result
    if version == "V1":
        return generate_v1(model=model, temperature=temperature, **kwargs)
    return generate_structured(version, model=model, temperature=temperature, **kwargs)


def implementation_id():
    digest = hashlib.sha256()
    paths = [
        BASE_DIR / name for name in (
            "generator.py", "selector.py", "structured_candidates.py", "structured_generation.py", "experiment.py",
        )
    ]
    paths += sorted((BASE_DIR / "prompts").glob("*.txt"))
    paths += sorted(path for path in (BASE_DIR / "data").rglob("*") if path.is_file())
    for path in paths:
        digest.update(str(path.relative_to(BASE_DIR)).encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def case_key(case, revision, round_name):
    payload = json.dumps([case, revision, round_name], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def execute_cases(cases, directory, round_name, *, retry_failed=False):
    budget = CallBudget(directory / "calls")
    revision = implementation_id()
    (directory / "cases").mkdir(parents=True, exist_ok=True)
    for case in cases:
        key = case_key(case, revision, round_name)
        path = directory / "cases" / f"{key}.json"
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous["status"] == "success":
                print(key, "reuse")
                continue
            if not retry_failed:
                raise RuntimeError(f"{key} failed earlier. Investigate before requesting a retry.")
        if budget.remaining < COST[case["version"]]:
            raise RuntimeError(f"{key} requires {COST[case['version']]} calls, {budget.remaining} remain.")
        token = CALL_OBSERVER.set(budget.observer(key))
        record = {"case_id": key, "case": case, "revision": revision, "round": round_name, "at": timestamp()}
        try:
            result = generate_case(case)
        except Exception as error:
            failure = {
                **record, "status": "failed", "error_type": type(error).__name__,
                "http_status": getattr(error, "code", None),
            }
            write_json(directory / "cases" / f"{key}.failure-{time_suffix()}.json", failure)
            write_json(path, failure)
            raise RuntimeError(f"{key} failed ({type(error).__name__}). Inspect the call log.") from None
        finally:
            CALL_OBSERVER.reset(token)
        write_json(path, {**record, "status": "success", "result": result})
        print(key, "success", f"({budget.remaining} calls remain)")


def time_suffix():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")


def build_review(directory):
    rows = []
    for path in sorted((directory / "cases").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["status"] != "success":
            continue
        case, result = record["case"], record["result"]
        rows.append((record, {
            "artwork": case["artwork"], "profile": case["profile"], "docent": result["docent"],
            "target_length": case["target_length"],
        }))
    random.Random(317).shuffle(rows)
    review, key, worksheet = {}, {}, {}
    for index, (record, sample) in enumerate(rows, 1):
        label = f"S{index:03}"
        case, result = record["case"], record["result"]
        length = result["character_count"]
        review[label] = sample
        worksheet[label] = {
            "ratings": {
                criterion: {"score": None, "quote": "", "rationale": ""}
                for criterion in WEIGHTS
                if criterion != "Contextual Personalization" or case["profile"]["context"]
            },
            "gates": {gate: None for gate in GATES},
            "gate_notes": "",
        }
        key[label] = {
            "case_id": record["case_id"], "case": case, "round": record["round"],
            "revision": record["revision"],
            "length_compliant": abs(length - case["target_length"]) <= case["target_length"] * LENGTH_TOLERANCE,
            **{field: result.get(field) for field in (
                "character_count", "input_tokens", "output_tokens", "thinking_tokens",
                "total_tokens", "elapsed_seconds", "wait_seconds", "api_calls",
            )},
        }
    review_id = time_suffix()
    for name, data in (("blind-review", review), ("review-key", key), ("ratings", worksheet)):
        write_json(directory / f"{name}-{review_id}.json", data)
    print(f"Prepared {len(review)} successful samples. See cases/ and calls/ for failures and per-call traces.")


def score_review(ratings, review_key):
    if ratings.keys() != review_key.keys():
        raise ValueError("Ratings and review key must cover the same samples")
    scored = {}
    for label, entry in ratings.items():
        context = review_key[label]["case"]["profile"]["context"]
        weights = {
            criterion: weight for criterion, weight in WEIGHTS.items()
            if criterion != "Contextual Personalization" or context
        }
        if entry["ratings"].keys() != weights.keys() or entry["gates"].keys() != set(GATES):
            raise ValueError(f"{label}: incomplete criteria or gates")
        total = 0
        for criterion, weight in weights.items():
            rating = entry["ratings"][criterion]
            score = rating["score"]
            if type(score) is not int or not 1 <= score <= 5 or not rating["quote"].strip() or not rating["rationale"].strip():
                raise ValueError(f"{label}: provide a 1–5 score, quote and rationale for {criterion}")
            total += weight * score
        gates = entry["gates"]
        if any(type(value) is not bool for value in gates.values()):
            raise ValueError(f"{label}: give each gate a true or false result")
        passed = all(gates.values())
        if not passed and not entry["gate_notes"].strip():
            raise ValueError(f"{label}: document the failed gates")
        scored[label] = {
            **review_key[label], **entry,
            "weighted_score_out_of_5": round(total / sum(weights.values()), 3),
            "passed_gates": passed,
        }
    return scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("baseline", "axes", "holdout", "review", "score"))
    parser.add_argument("--versions", nargs="+", choices=VERSIONS)
    parser.add_argument("--axes", nargs="+", choices=tuple(DEVELOPMENT)[1:])
    parser.add_argument("--artwork", choices=ARTWORKS)
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--review-key", type=Path)
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
    parser.add_argument("--round", default="initial")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "results" / "docent-quality")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Make live calls using a verified free-tier key")
    args = parser.parse_args()
    if args.phase == "score":
        if not args.ratings or not args.review_key:
            parser.error("Score requires --ratings and --review-key.")
        try:
            scored = score_review(
                json.loads(args.ratings.read_text(encoding="utf-8")),
                json.loads(args.review_key.read_text(encoding="utf-8")),
            )
        except (ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
        destination = args.ratings.parent / f"scored-{time_suffix()}.json"
        write_json(destination, scored)
        print(destination)
        return
    if args.phase == "review":
        if not (args.output / "cases").is_dir():
            parser.error("No experiment cases found.")
        build_review(args.output)
        return
    versions = args.versions or (("V0", "V1", "STRUCTURED_PAIR", "STRUCTURED_MULTI") if args.phase == "baseline" else None)
    if not versions:
        parser.error("Choose --versions after development review.")
    cases = list(make_cases(args.phase, dict.fromkeys(versions), args.model, args.axes, args.artwork))
    if not args.execute:
        print(json.dumps(cases, ensure_ascii=False, indent=2))
        print(f"Full plan: at most {sum(COST[case['version']] for case in cases)} calls before reuse.")
        return
    if not os.getenv("GEMINI_API_KEY"):
        parser.error("Set GEMINI_API_KEY in the local .env first.")
    try:
        execute_cases(cases, args.output, args.round, retry_failed=args.retry_failed)
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
