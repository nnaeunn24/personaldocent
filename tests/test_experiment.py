import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

from experiment import (
    ARTWORKS, COST, DEFAULT_MODEL, GATES, WEIGHTS,
    CallBudget, execute_cases, make_cases, score_review,
)
from generator import CALL_OBSERVER, call_model
from selector import CATEGORIES
from structured_candidates import MAX_CANDIDATES, build_candidates
from structured_generation import generate_structured, parse_plan


def usage(text):
    return dict(
        text=text, model=DEFAULT_MODEL, input_tokens=3, output_tokens=2,
        thinking_tokens=1, total_tokens=6, elapsed_seconds=0.01, wait_seconds=0,
    )


def plan(bundle_id):
    return json.dumps({"choices": [{"bundle_id": bundle_id, "reason": "선정 근거", "focus": "관찰 순서"}]})


class StructuredPipelineTest(unittest.TestCase):
    def test_multi_shortlists_have_connected_two_edge_options(self):
        for artwork in ARTWORKS:
            for what in CATEGORIES:
                with self.subTest(artwork=artwork, what=what):
                    pair = build_candidates(artwork, what)
                    multi = build_candidates(artwork, what, max_relations=2)
                    self.assertLessEqual(len(multi), MAX_CANDIDATES)
                    self.assertTrue(any(len(bundle["relations"]) == 2 for bundle in multi.values()))
                    self.assertTrue(all(len(bundle["relations"]) <= 1 for bundle in pair.values()))
                    for bundle in multi.values():
                        core = bundle["core_unit"]
                        self.assertIn(what, core["categories"])
                        reached = {core["id"]}
                        for edge in bundle["relations"]:
                            endpoints = {edge["source_unit_id"], edge["target_unit_id"]}
                            self.assertTrue(reached & endpoints)
                            reached.update(endpoints)
                        self.assertEqual(reached, {core["id"], *(u["id"] for u in bundle["related_units"])})

    def test_plan_and_generation_retain_only_verified_bundles(self):
        candidates = build_candidates(ARTWORKS[0], "TECHNIQUE", max_relations=2)
        selected = next(key for key, bundle in candidates.items() if len(bundle["relations"]) == 2)
        with patch("structured_generation.call_model", return_value=usage(plan(selected))) as select:
            with patch("generator.call_model", return_value=usage("모의 도슨트")) as generate:
                result = generate_structured(
                    "STRUCTURED_MULTI", DEFAULT_MODEL, ARTWORKS[0], "DISCOVERY",
                    "TECHNIQUE", "BASIC", "산책하고 싶어요.", 500,
                )
        self.assertTrue(select.call_args.kwargs["json_output"])
        self.assertEqual(generate.call_args.kwargs["temperature"], 0.7)
        self.assertEqual(result["docent"], "모의 도슨트")
        self.assertEqual(result["bundles"][0]["relations"], candidates[selected]["relations"])
        self.assertIn("산책하고 싶어요.", result["selection_prompt"])
        self.assertIn("관찰 순서", result["full_prompt"])
        self.assertEqual(result["total_tokens"], 12)
        for invalid in ("null", "[]", plan("invalid"), '{"choices":[null]}'):
            with self.assertRaises(ValueError):
                parse_plan(invalid, candidates)


class RunnerTest(unittest.TestCase):
    def test_initial_schedule_and_controls(self):
        baseline = list(make_cases("baseline", ["V0", "V1", "STRUCTURED_PAIR", "STRUCTURED_MULTI"], DEFAULT_MODEL))
        axes = list(make_cases("axes", ["STRUCTURED_MULTI"], DEFAULT_MODEL, artwork_filter=ARTWORKS[0]))
        holdout = list(make_cases("holdout", ["STRUCTURED_MULTI"], DEFAULT_MODEL))
        self.assertEqual(sum(COST[c["version"]] for c in baseline + axes + holdout), 26)
        anchor = baseline[0]["profile"]
        for case in axes:
            self.assertEqual(sum(case["profile"][key] != value for key, value in anchor.items()), 1)
        self.assertEqual({case["artwork"] for case in holdout}, set(ARTWORKS))

    def test_budget_reserves_failures_and_blocks_excess_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            budget = CallBudget(path, limit=1)
            budget.observer("case")({"prompt": "실험"})({"status": "failed"})
            self.assertEqual(CallBudget(path, limit=1).remaining, 0)
            self.assertEqual(json.loads((path / "001.json").read_text())["status"], "failed")
            with self.assertRaises(RuntimeError):
                budget.observer("next")({})

    def test_model_observer_records_failed_attempt_and_disables_retries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = CallBudget(root, limit=1)
            fake_client = MagicMock()
            fake_client.models.generate_content.side_effect = RuntimeError("fake error")
            token = CALL_OBSERVER.set(budget.observer("case"))
            try:
                with patch.dict(os.environ, {"GEMINI_API_KEY": "mock"}), patch(
                    "generator.genai.Client", return_value=fake_client
                ) as construct, patch.dict("generator.LAST_CALL", clear=True):
                    with self.assertRaises(RuntimeError):
                        call_model(DEFAULT_MODEL, "prompt")
                    fake_client.models.generate_content.assert_called_once()
                    self.assertEqual(construct.call_args.kwargs["http_options"].retry_options.attempts, 1)
            finally:
                CALL_OBSERVER.reset(token)
            self.assertEqual(json.loads((root / "001.json").read_text())["status"], "failed")

    def test_identical_success_is_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            case = next(make_cases("baseline", ["V0"], DEFAULT_MODEL))
            with patch("experiment.generate_case", return_value={"docent": "모의 도슨트"}) as generate:
                execute_cases([case], Path(temp), "initial")
                execute_cases([case], Path(temp), "initial")
            generate.assert_called_once()

    def test_context_absence_is_normalized_and_gate_stays_independent(self):
        present = {
            name: {"score": 5, "quote": "인용", "rationale": "이유"}
            for name in WEIGHTS if name != "Contextual Personalization"
        }
        gates = dict.fromkeys(GATES, True)
        gates["material_facts"] = False
        scored = score_review(
            {"S001": {"ratings": present, "gates": gates, "gate_notes": "사실 오류"}},
            {"S001": {"case": {"profile": {"context": ""}}}},
        )["S001"]
        self.assertEqual(scored["weighted_score_out_of_5"], 5)
        self.assertFalse(scored["passed_gates"])


class StreamlitTest(unittest.TestCase):
    def test_streamlit_renders_new_variant_and_download_data(self):
        candidate = next(iter(build_candidates(ARTWORKS[0], "VISUAL", max_relations=2)))
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        metadata = SimpleNamespace(
            prompt_token_count=3, candidates_token_count=2,
            thoughts_token_count=1, total_token_count=6,
        )
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "mock"}),
            patch("generator.genai.Client") as client,
            patch("generator.LAST_CALL", new=MagicMock(**{"get.return_value": None})),
            patch("pathlib.Path.write_text", autospec=True) as save,
        ):
            client.return_value.models.generate_content.side_effect = [
                SimpleNamespace(text=plan(candidate), usage_metadata=metadata),
                SimpleNamespace(text="모의 도슨트", usage_metadata=metadata),
            ]
            app.run(timeout=15)
            app.multiselect[0].set_value(["STRUCTURED_MULTI"])
            app.button[0].click().run(timeout=15)
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state.results[0]["docent"], "모의 도슨트")
            self.assertEqual(app.session_state.results[0]["api_calls"], 2)
            self.assertTrue(any("선정 원본" in item.label for item in app.expander))
            save.assert_called()
