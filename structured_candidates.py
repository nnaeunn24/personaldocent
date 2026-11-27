"""Bounded, source-backed bundle candidates for an offline experiment."""

from selector import load_json, load_knowledge, load_relations, select_knowledge


# Use only the stored edge. Pairing by category alone cannot establish a relation.
MAX_CANDIDATES = 14
RELATION_PRIORITY = {
    "EXPLAINS": 0,
    "EXPRESSES": 0,
    "CONTRASTS_WITH": 0,
    "DISTINGUISHES_FROM": 0,
    "SUPPORTS_INTERPRETATION": 0,
    "EXEMPLIFIES": 1,
    "ELABORATES": 1,
    "PROVIDES_CONTEXT": 1,
    "COMPLEMENTS": 2,
}


def build_candidates(artwork, what, *, use_relations=True):
    """Return distinct, grounded options with at most one stored edge each."""
    graph = select_knowledge(artwork, what, max_hops=1)
    knowledge = load_knowledge(artwork)
    relations = load_relations(artwork)
    metadata = load_json(artwork, "artwork.json")
    if metadata["artwork_id"] != knowledge["artwork_id"]:
        raise ValueError("Knowledge does not match artwork metadata")

    units = {unit["id"]: unit for unit in knowledge["content_units"] if unit.get("evidence")}
    edges = {edge["id"]: edge for edge in relations["relations"] if edge.get("evidence")}
    core_ids = [unit["id"] for unit in graph["core_units"] if unit["id"] in units]
    eligible_cores = set(core_ids)
    paired = {core: [] for core in core_ids}
    if use_relations:
        for edge_ref in graph["relations"]:
            edge = edges.get(edge_ref["id"])
            if edge is None:
                continue
            source, target = edge["source_unit_id"], edge["target_unit_id"]
            if source not in units or target not in units:
                continue
            core, other = (source, target) if source in eligible_cores else (target, source)
            paired[core].append((other, edge))

    # Reserve independent options and distribute pair slots across different cores.
    ranked_cores = sorted(
        core_ids, key=lambda core: (units[core].get("scope") in {"general", "later_context"}, core)
    )
    choices = [(core, None, None) for core in ranked_cores]
    if use_relations:
        for core in ranked_cores:
            paired[core].sort(
                key=lambda pair: (
                    RELATION_PRIORITY.get(pair[1]["relation_type"], 3),
                    pair[1]["grounding_type"] != "EXPLICIT",
                    pair[1]["id"],
                )
            )
        independent = choices[:4]
        covered = {core for core, _, _ in independent}
        pairs = []
        for index in range(2):
            for core in sorted(ranked_cores, key=lambda uid: (uid in covered, uid)):
                if index < len(paired[core]):
                    other, edge = paired[core][index]
                    pairs.append((core, other, edge))
        chosen_pairs = pairs[: MAX_CANDIDATES - len(independent)]
        choices = chosen_pairs + independent
        if len(choices) < MAX_CANDIDATES:
            choices += [(core, None, None) for core in ranked_cores[4:]]
    return {
        f"B{index:02d}": {
            "core_unit": units[core],
            "related_units": [units[other]] if other else [],
            "relations": [edge] if edge else [],
        }
        for index, (core, other, edge) in enumerate(choices[:MAX_CANDIDATES], 1)
    }


def resolve_selection(candidate_ids, candidates, *, max_bundles=2):
    """Resolve whole candidates atomically and reject shared CU IDs."""
    if (
        not isinstance(candidate_ids, list)
        or not 1 <= len(candidate_ids) <= max_bundles
        or any(not isinstance(item, str) or item not in candidates for item in candidate_ids)
    ):
        raise ValueError("Invalid bundle selection")
    selected = {}
    used_units = set()
    for candidate_id in candidate_ids:
        bundle = candidates[candidate_id]
        unit_ids = {bundle["core_unit"]["id"], *(unit["id"] for unit in bundle["related_units"])}
        if unit_ids & used_units:
            raise ValueError("Selected bundles share content units")
        used_units.update(unit_ids)
        selected[candidate_id] = bundle
    return selected
