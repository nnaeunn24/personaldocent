import unittest

from selector import ARTWORKS, BASE_DIR, CATEGORIES, load_knowledge, load_relations
from structured_candidates import MAX_CANDIDATES, build_candidates, resolve_selection


class StructuredCandidatesTest(unittest.TestCase):
    def test_shortlist_keeps_eligible_cores_original_edges_and_evidence(self):
        for artwork in ARTWORKS:
            lines = (BASE_DIR / "data" / artwork / "description.txt").read_text(
                encoding="utf-8-sig"
            ).splitlines()
            units = load_knowledge(artwork)["content_units"]
            edges = load_relations(artwork)["relations"]
            for record in units + edges:
                for evidence in record["evidence"]:
                    self.assertIn(evidence["quote"], lines[evidence["line"] - 1], record["id"])
            for what in CATEGORIES:
                with self.subTest(artwork=artwork, what=what):
                    candidates = build_candidates(artwork, what)
                    self.assertTrue(candidates)
                    self.assertLessEqual(len(candidates), MAX_CANDIDATES)
                    self.assertEqual(candidates, build_candidates(artwork, what))
                    for bundle in candidates.values():
                        core = bundle["core_unit"]
                        self.assertIn(what, core["categories"])
                        self.assertIn(core, units)
                        for edge in bundle["relations"]:
                            self.assertIn(edge, edges)
                            self.assertSetEqual(
                                {core["id"], bundle["related_units"][0]["id"]},
                                {edge["source_unit_id"], edge["target_unit_id"]},
                            )
                    control = build_candidates(artwork, what, use_relations=False)
                    self.assertTrue(control)
                    self.assertTrue(all(not b["relations"] and not b["related_units"] for b in control.values()))

    def test_selection_rejects_invalid_overlapping_and_oversized_choices(self):
        candidates = build_candidates(ARTWORKS[0], CATEGORIES[0])
        first, second, third = list(candidates)[:3]
        for invalid in ([], [first, first], [first, second, third], ["invented"], [{}]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    resolve_selection(invalid, candidates)
        disjoint = [key for key, bundle in candidates.items() if not bundle["relations"]][:2]
        self.assertEqual(list(resolve_selection(disjoint, candidates)), disjoint)
