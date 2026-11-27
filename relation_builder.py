
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

MODEL = "gemini-3.5-flash-lite"

ARTWORKS = [
    "The Bedroom",
    "The Starry Night",
]

RELATION_TYPES = {
    "EXEMPLIFIES",
    "EXPLAINS",
    "EXPRESSES",
    "CONTEXTUALIZES",
    "CONTRASTS",
    "ELABORATES",
}

GROUNDING_TYPES = {
    "EXPLICIT",
    "INTERPRETIVE",
}


# ==========================================
# 1. 작품 데이터 불러오기
# ==========================================

def load_artwork_data(artwork):

    folder = BASE_DIR / "data" / artwork

    metadata = json.loads(
        (folder / "artwork.json").read_text(
            encoding="utf-8-sig"
        )
    )

    description = (
        folder / "description.txt"
    ).read_text(encoding="utf-8-sig")

    knowledge = json.loads(
        (folder / "knowledge.json").read_text(
            encoding="utf-8-sig"
        )
    )

    return metadata, description, knowledge


# ==========================================
# 2. 관계 추출 프롬프트 구성
# ==========================================

def build_prompt(metadata, description, knowledge):

    template = (
        BASE_DIR
        / "prompts"
        / "relation_extraction.txt"
    ).read_text(encoding="utf-8-sig")

    prompt = template.replace(
        "{{artwork_metadata}}",
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
    )

    prompt = prompt.replace(
        "{{raw_description}}",
        description,
    )

    prompt = prompt.replace(
        "{{knowledge_data}}",
        json.dumps(
            knowledge,
            ensure_ascii=False,
            indent=2,
        ),
    )

    return prompt


# ==========================================
# 3. 관계 데이터 기본 검증
# ==========================================

def normalize_text(text):
    return " ".join(str(text).split()).casefold()


def validate_relations(
    relations,
    knowledge,
    description,
):

    unit_ids = {
        unit["id"]
        for unit in knowledge["content_units"]
    }

    valid = []
    rejected = []
    seen = set()

    normalized_description = normalize_text(
        description
    )

    for relation in relations:

        if not isinstance(relation, dict):
            rejected.append({
                "relation": relation,
                "errors": ["Invalid relation format"],
            })
            continue

        errors = []

        source = relation.get("source_unit_id")
        target = relation.get("target_unit_id")
        relation_type = relation.get("relation_type")

        if source not in unit_ids:
            errors.append("Invalid source Unit ID")

        if target not in unit_ids:
            errors.append("Invalid target Unit ID")

        if source == target:
            errors.append("Self relation")

        if relation_type not in RELATION_TYPES:
            errors.append("Invalid relation type")

        if relation.get("grounding_type") not in GROUNDING_TYPES:
            errors.append("Invalid grounding type")

        if not relation.get("description"):
            errors.append("Missing description")

        if not relation.get("explanation_value"):
            errors.append("Missing explanation value")

        evidence = relation.get("evidence", [])

        if not isinstance(evidence, list) or not evidence:
            errors.append("Missing evidence")

        else:
            for item in evidence:

                if not isinstance(item, dict):
                    errors.append("Invalid evidence format")
                    continue

                quote = normalize_text(
                    item.get("quote", "")
                )

                if not item.get("source_id"):
                    errors.append("Missing source ID")

                if not quote:
                    errors.append("Empty evidence quote")

                elif quote not in normalized_description:
                    errors.append(
                        "Quote not found in raw description"
                    )

        # CONTRASTS는 방향에 관계없이 중복 검사
        if (
            relation_type == "CONTRASTS"
            and isinstance(source, str)
            and isinstance(target, str)
        ):
            key = (
                tuple(sorted([source, target])),
                relation_type,
            )
        else:
            key = (
                str(source),
                str(target),
                str(relation_type),
            )

        if key in seen:
            errors.append("Duplicate relation")

        if errors:
            rejected.append({
                "relation": relation,
                "errors": errors,
            })
            continue

        seen.add(key)

        relation["review_status"] = "pending"
        relation["validation_status"] = "basic_pass"

        valid.append(relation)

    return valid, rejected


# ==========================================
# 4. Gemini 관계 추출
# ==========================================

def extract_relations(artwork):

    metadata, description, knowledge = (
        load_artwork_data(artwork)
    )

    prompt = build_prompt(
        metadata,
        description,
        knowledge,
    )

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")

    client = genai.Client(api_key=api_key)

    print("\n" + "=" * 60)
    print("Artwork:", artwork)
    print("Model:", MODEL)
    print(
        "Content Units:",
        len(knowledge["content_units"])
    )
    print("Extracting semantic relations...")

    start = time.perf_counter()

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            max_output_tokens=16384,
        ),
    )

    elapsed = time.perf_counter() - start

    raw_text = response.text

    if not raw_text:
        raise ValueError("Empty Gemini response")

    # 원본 응답 먼저 보존
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    raw_path = (
        results_dir
        / f'{artwork.replace(" ", "_")}_relations_raw.txt'
    )

    raw_path.write_text(
        raw_text,
        encoding="utf-8",
    )

    parsed = json.loads(raw_text)

    relations = parsed.get("relations", [])

    if not isinstance(relations, list):
        raise ValueError("Invalid relations JSON")

    valid, rejected = validate_relations(
        relations,
        knowledge,
        description,
    )

    artwork_id = knowledge["artwork_id"]

    for index, relation in enumerate(
        valid,
        start=1,
    ):
        relation["id"] = (
            f"{artwork_id}_REL_{index:03d}"
        )

    usage = response.usage_metadata

    result = {
        "artwork": artwork,
        "artwork_id": artwork_id,
        "model": MODEL,
        "review_status": "draft_pending_human_review",
        "elapsed_seconds": round(elapsed, 2),
        "statistics": {
            "generated_count": len(relations),
            "valid_count": len(valid),
            "rejected_count": len(rejected),
        },
        "usage": {
            "input_tokens": getattr(
                usage,
                "prompt_token_count",
                None,
            ),
            "output_tokens": getattr(
                usage,
                "candidates_token_count",
                None,
            ),
            "total_tokens": getattr(
                usage,
                "total_token_count",
                None,
            ),
        },
        "relations": valid,
        "rejected_relations": rejected,
    }

    # 검수용 별도 파일 저장
    output_path = (
        BASE_DIR
        / "data"
        / artwork
        / "relations_draft.json"
    )

    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Elapsed:", round(elapsed, 2), "seconds")
    print("Generated:", len(relations))
    print("Valid:", len(valid))
    print("Rejected:", len(rejected))
    print("Saved:", output_path)

    return result


# ==========================================
# 5. 사람이 읽기 쉬운 검수 보고서 생성
# ==========================================

def create_review_report(artwork, result):

    _, _, knowledge = load_artwork_data(artwork)

    unit_map = {
        unit["id"]: unit
        for unit in knowledge["content_units"]
    }

    lines = [
        "=" * 70,
        f"Artwork: {artwork}",
        f'Model: {result["model"]}',
        f'Statistics: {result["statistics"]}',
        "=" * 70,
    ]

    for relation in result["relations"]:

        source = unit_map[
            relation["source_unit_id"]
        ]

        target = unit_map[
            relation["target_unit_id"]
        ]

        lines.extend([
            "",
            f'{relation["id"]}',
            f'Type: {relation["relation_type"]}',
            f'Grounding: {relation["grounding_type"]}',
            "",
            f'Source: {source["id"]}',
            source["content"],
            "",
            f'Target: {target["id"]}',
            target["content"],
            "",
            "Relation:",
            relation["description"],
            "",
            "Explanation Value:",
            relation["explanation_value"],
            "",
            "Evidence:",
        ])

        for evidence in relation["evidence"]:
            lines.append(
                f'[{evidence["source_id"]}] '
                f'{evidence["quote"]}'
            )

        lines.append("-" * 70)

    lines.append("\n=== REJECTED RELATIONS ===")

    for rejected in result["rejected_relations"]:
        lines.append(
            json.dumps(
                rejected,
                ensure_ascii=False,
                indent=2,
            )
        )

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    path = (
        results_dir
        / f'{artwork.replace(" ", "_")}_relations_review.txt'
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("Review report:", path)


# ==========================================
# 6. 실행
# ==========================================

if __name__ == "__main__":

    # 첫 테스트는 The Bedroom만 실행
    artwork = "The Bedroom"

    result = extract_relations(artwork)

    create_review_report(
        artwork,
        result,
    )
