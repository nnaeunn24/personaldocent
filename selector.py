import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ARTWORKS = [
    "The Bedroom",
    "The Starry Night",
]

CATEGORIES = [
    "VISUAL",
    "MEANING",
    "TECHNIQUE",
    "BACKGROUND",
]


def load_json(artwork, filename):
    if artwork not in ARTWORKS:
        raise ValueError(f"Invalid artwork: {artwork}")

    path = BASE_DIR / "data" / artwork / filename

    if not path.exists():
        raise FileNotFoundError(str(path))

    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def load_knowledge(artwork):
    return load_json(artwork, "knowledge.json")


def load_relations(artwork):
    return load_json(artwork, "relations.json")


def compact_unit(unit):
    return {
        "id": unit["id"],
        "content": unit["content"],
        "categories": unit["categories"],
        "claim_type": unit.get("claim_type"),
        "scope": unit.get("scope"),
        "source_refs": [
            {
                "source_id": e["source_id"],
                "line": e.get("line"),
            }
            for e in unit.get("evidence", [])
        ],
    }


def compact_relation(relation):
    return {
        "id": relation["id"],
        "source_unit_id": relation["source_unit_id"],
        "target_unit_id": relation["target_unit_id"],
        "relation_type": relation["relation_type"],
        "description": relation["description"],
        "explanation_value": relation["explanation_value"],
        "grounding_type": relation["grounding_type"],
    }


def select_knowledge(artwork, what, max_hops=2):
    what = what.upper()

    if what not in CATEGORIES:
        raise ValueError(f"Invalid WHAT: {what}")

    knowledge = load_knowledge(artwork)
    relation_data = load_relations(artwork)

    if knowledge["artwork_id"] != relation_data["artwork_id"]:
        raise ValueError("Artwork ID mismatch")

    units = knowledge["content_units"]
    relations = relation_data["relations"]

    unit_map = {
        u["id"]: u
        for u in units
    }

    relation_map = {
        r["id"]: r
        for r in relations
    }

    if len(unit_map) != len(units):
        raise ValueError("Duplicate unit IDs")

    if len(relation_map) != len(relations):
        raise ValueError("Duplicate relation IDs")

    adjacency = {
        uid: []
        for uid in unit_map
    }

    for relation in relations:
        source = relation["source_unit_id"]
        target = relation["target_unit_id"]

        if source not in unit_map or target not in unit_map:
            raise ValueError(
                f"Invalid relation: {relation['id']}"
            )

        if source == target:
            raise ValueError(
                f"Self relation: {relation['id']}"
            )

        adjacency[source].append((target, relation))
        adjacency[target].append((source, relation))

    core_ids = {
        u["id"]
        for u in units
        if what in u["categories"]
    }

    if not core_ids:
        raise ValueError("No matching Core Units")

    visited = set(core_ids)
    frontier = set(core_ids)
    selected_relation_ids = set()

    for _ in range(max_hops):
        next_frontier = set()

        for uid in frontier:
            for neighbor, relation in adjacency[uid]:

                selected_relation_ids.add(
                    relation["id"]
                )

                if neighbor not in visited:
                    next_frontier.add(neighbor)

        visited.update(next_frontier)
        frontier = next_frontier

        if not frontier:
            break

    supporting_ids = visited - core_ids

    core_units = [
        compact_unit(u)
        for u in units
        if u["id"] in core_ids
    ]

    supporting_units = [
        compact_unit(u)
        for u in units
        if u["id"] in supporting_ids
    ]

    candidate_relations = [
        compact_relation(r)
        for r in relations
        if r["id"] in selected_relation_ids
    ]

    return {
        "artwork": artwork,
        "artwork_id": knowledge["artwork_id"],
        "what": what,
        "core_units": core_units,
        "supporting_units": supporting_units,
        "relations": candidate_relations,
        "statistics": {
            "total_units": len(units),
            "core_count": len(core_units),
            "supporting_count": len(supporting_units),
            "candidate_relation_count": len(
                candidate_relations
            ),
            "candidate_unit_count": len(visited),
        },
    }

def validate_bundles(selection, candidates):

    unit_map = {
        u["id"]: u
        for u in (
            candidates["core_units"]
            + candidates["supporting_units"]
        )
    }

    core_ids = {
        u["id"]
        for u in candidates["core_units"]
    }

    relations = candidates["relations"]

    bundles = selection.get("bundles", [])

    if not isinstance(bundles, list):
        raise ValueError("Invalid bundles format")

    if not 1 <= len(bundles) <= 3:
        raise ValueError("Select 1 to 3 bundles")

    validated = []
    all_unit_ids = set()

    for index, bundle in enumerate(bundles, start=1):

        core_id = bundle["core_unit_id"]

        if core_id not in core_ids:
            raise ValueError(
                f"Invalid Core Unit: {core_id}"
            )

        if core_id in all_unit_ids:
            raise ValueError(
                f"Duplicate Core Unit: {core_id}"
            )

        related_ids = bundle.get(
            "related_unit_ids", []
        )

        if not isinstance(related_ids, list):
            raise ValueError(
                "Invalid related_unit_ids"
            )

        # 잘못된 ID 및 중복 제거
        cleaned_ids = []

        for uid in related_ids:

            if uid not in unit_map:
                continue

            if uid == core_id:
                continue

            if uid in all_unit_ids:
                continue

            if uid in cleaned_ids:
                continue

            cleaned_ids.append(uid)

        # 이후 Bundle의 Core Unit을 위한 공간 확보
        remaining_bundles = len(bundles) - index

        remaining_budget = (
            8
            - len(all_unit_ids)
            - 1
            - remaining_bundles
        )

        max_related = max(
            0,
            min(3, remaining_budget)
        )

        cleaned_ids = cleaned_ids[:max_related]

        unit_ids = [core_id] + cleaned_ids

        # 선정된 Unit 사이에 실제 존재하는 관계 탐색
        matched_relations = []

        graph = {
            uid: set()
            for uid in unit_ids
        }

        for relation in relations:

            source = relation["source_unit_id"]
            target = relation["target_unit_id"]

            if source in graph and target in graph:

                matched_relations.append(relation)

                graph[source].add(target)
                graph[target].add(source)

        # Core와 실제로 연결된 Unit 탐색
        visited = {core_id}
        frontier = [core_id]

        while frontier:

            current = frontier.pop()

            for neighbor in graph[current]:

                if neighbor not in visited:

                    visited.add(neighbor)
                    frontier.append(neighbor)

        # 연결되지 않은 Related Unit 제외
        valid_related_ids = [
            uid for uid in cleaned_ids
            if uid in visited
        ]

        removed_ids = [
            uid for uid in related_ids
            if uid not in valid_related_ids
        ]

        valid_ids = {
            core_id,
            *valid_related_ids,
        }

        # 최종 Unit 사이의 관계만 유지
        bundle_relations = [
            relation
            for relation in matched_relations
            if (
                relation["source_unit_id"] in valid_ids
                and
                relation["target_unit_id"] in valid_ids
            )
        ]

        all_unit_ids.update(valid_ids)

        validated.append({
            "bundle_id": f"BUNDLE_{index:02d}",

            "topic": bundle["topic"],

            "core_unit": unit_map[core_id],

            "related_units": [
                unit_map[uid]
                for uid in valid_related_ids
            ],

            "relations": bundle_relations,

            "selection_reason": (
                bundle["selection_reason"]
                if not removed_ids
                else "Core Unit과 실제 의미적 관계로 연결된 정보만 유지함."
            ),

            "validation": {
                "original_related_unit_ids": related_ids,
                "removed_unit_ids": removed_ids,
                "retained_unit_ids": valid_related_ids,
                "auto_mapped_relation_ids": [
                    r["id"]
                    for r in bundle_relations
                ],
                "adjusted": bool(removed_ids),
            },
        })

    if len(all_unit_ids) > 8:
        raise ValueError("Too many selected units")

    return validated

def run_all_tests():
    results = []

    for artwork in ARTWORKS:
        for what in CATEGORIES:

            result = select_knowledge(
                artwork,
                what,
            )

            results.append(result)

            print("=" * 60)
            print(artwork, "|", what)
            print(result["statistics"])

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    path = results_dir / "relation_candidates.json"

    path.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", path)


if __name__ == "__main__":
    run_all_tests()