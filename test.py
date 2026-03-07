"""
test.py — Unit tests for all three pipeline versions.

All tests use a MockModel that requires no GPU or model download.

Run:
    python -m pytest test.py -v
    python -m pytest test.py -v -k "TestV1"   # single version
"""

from __future__ import annotations

import re
import sys
import os
import math
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from shared.schemas import (
    AnchorResult,
    LocalizationResult,
    LocalizationTrial,
    AnchorValidationResult,
    AnchorValidationTrial,
    ConfidenceBreakdown,
    TokenAttribution,
    ExtractionResult,
)
from shared.value_generator import ValueGenerator, detect_value_type


# ─────────────────────────────────────────────── Mock model ──

class MockModel:
    """
    Deterministic mock extraction model.  Extracts the value that appears
    after 'Total Due:' in the text, or falls back to a fixed token.
    Compatible with HFExtractionModel's interface.
    """

    def __init__(self):
        self.device = "cpu"
        self.is_seq2seq = False
        self._use_chat_template = False
        self._quantized = False

    def extract(self, text: str, field: str) -> str:
        m = re.search(r"Total Due:\s*([\$\d,\.]+)", text)
        if m:
            return m.group(1).strip()
        words = text.split()
        return words[min(5, len(words) - 1)] if words else "unknown"

    def build_prompt(self, text: str, field: str) -> str:
        return f"Extract {field} from:\n{text}"

    @property
    def vocab_words(self) -> list[str]:
        return ["apple", "bridge", "castle", "dragon", "engine"]

    def forward_with_grad(self, text: str, field: str, target_value: str):
        """Return synthetic gradient attribution scores."""
        prompt = self.build_prompt(text, field)
        n_tokens = max(1, len(prompt) // 4)
        scores = torch.zeros(n_tokens)
        mid = n_tokens // 2
        scores[mid] = 1.0
        if mid + 1 < n_tokens:
            scores[mid + 1] = 0.8
        offsets = [(i * 4, (i + 1) * 4) for i in range(n_tokens)]
        return scores, offsets


# ─────────────────────────────────────────────── Shared: ValueGenerator ──

class TestValueGenerator(unittest.TestCase):

    def setUp(self):
        self.gen = ValueGenerator()

    def test_detect_currency(self):
        self.assertEqual(detect_value_type("$22,242.50"), "currency")
        self.assertEqual(detect_value_type("$100.00"), "currency")

    def test_detect_number(self):
        self.assertEqual(detect_value_type("42"), "number")
        self.assertEqual(detect_value_type("3.14"), "number")

    def test_detect_date_iso(self):
        self.assertEqual(detect_value_type("2024-03-15"), "date")

    def test_detect_date_text(self):
        self.assertEqual(detect_value_type("March 15, 2024"), "date")

    def test_detect_string(self):
        self.assertEqual(detect_value_type("Acme Corp"), "string")

    def test_generate_currency_differs(self):
        original = "$901.48"
        alternatives = self.gen.generate(original, n=5, exclude=[original])
        self.assertEqual(len(alternatives), 5)
        for alt in alternatives:
            self.assertNotEqual(alt, original)

    def test_generate_date_differs(self):
        original = "March 15, 2024"
        alternatives = self.gen.generate(original, n=5, exclude=[original])
        self.assertGreater(len(alternatives), 0)
        for alt in alternatives:
            self.assertNotEqual(alt.lower(), original.lower())

    def test_generate_no_duplicates(self):
        alternatives = self.gen.generate("100", n=10)
        self.assertEqual(len(alternatives), len(set(alternatives)))


# ─────────────────────────────────────────────── Shared: Localizer ──

class TestLocalizer(unittest.TestCase):

    def setUp(self):
        from shared.localizer import Localizer
        self.model = MockModel()
        self.localizer = Localizer(self.model, n_trials=3)

    def test_runs_correct_trial_count(self):
        text = "Invoice Total Due: $22,242.50\nDate: 2024-03-15"
        result = self.localizer.test(
            full_text=text, field="total amount due",
            original_value="$22,242.50",
            anchor_text=text, anchor_start=0, anchor_end=len(text),
        )
        self.assertEqual(result.n_trials, 3)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)

    def test_result_fields(self):
        text = "Total Due: $100.00"
        result = self.localizer.test(
            full_text=text, field="total",
            original_value="$100.00",
            anchor_text=text, anchor_start=0, anchor_end=len(text),
        )
        self.assertEqual(len(result.trials), 3)
        for trial in result.trials:
            self.assertIsInstance(trial.injected_value, str)
            self.assertIsInstance(trial.success, bool)


# ─────────────────────────────────────────────── Shared: AnchorValidator ──

class TestAnchorValidator(unittest.TestCase):

    def setUp(self):
        from shared.anchor_validator import AnchorValidator
        self.model = MockModel()
        self.validator = AnchorValidator(self.model, n_trials=3)

    def test_runs_correct_trial_count(self):
        text = "Invoice Total Due: $22,242.50\nDate: 2024-03-15"
        result = self.validator.validate(
            full_text=text, field="total",
            original_value="$22,242.50",
            anchor_text=text, anchor_start=0, anchor_end=len(text),
        )
        self.assertEqual(result.n_trials, 3)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)


# ─────────────────────────────────────────────── V1: HeuristicAnchorFinder ──

class TestV1HeuristicAnchorFinder(unittest.TestCase):

    def setUp(self):
        from v1_basic.anchor_finder import HeuristicAnchorFinder
        self.model = MockModel()
        self.finder = HeuristicAnchorFinder(self.model, context_chars=20)

    def test_value_match_finds_sentence(self):
        text = "This is a preamble. Total Due: $19,530.00 as of today. Thank you."
        result = self.finder.find(text, "total", "$19,530.00")
        self.assertIn("$19,530.00", result.text)
        self.assertEqual(result.method, "value_match")

    def test_anchor_offsets_valid(self):
        text = "Vendor: Acme Corp\nTotal Due: $500.00"
        result = self.finder.find(text, "total", "$500.00")
        self.assertGreater(result.end_char, result.start_char)
        self.assertEqual(result.text, text[result.start_char:result.end_char])

    def test_fallback_for_unliteral_value(self):
        text = "The amount owed is five hundred dollars. Payment is overdue."
        result = self.finder.find(text, "amount", "$500")
        self.assertIn(result.method, ("sentence_mask", "whole_text"))
        self.assertIsNotNone(result.text)

    def test_whole_text_fallback(self):
        from v1_basic.anchor_finder import HeuristicAnchorFinder
        finder = HeuristicAnchorFinder(self.model)
        result = finder._whole_text_anchor("hello world")
        self.assertEqual(result.start_char, 0)
        self.assertEqual(result.end_char, 11)


# ─────────────────────────────────────────────── V1: ConfidenceScorer ──

class TestV1ConfidenceScorer(unittest.TestCase):

    def _loc(self, score: float, n: int = 5) -> LocalizationResult:
        return LocalizationResult(score=score, n_trials=n, n_success=int(score * n), trials=[])

    def _av(self, score: float, n: int = 5) -> AnchorValidationResult:
        return AnchorValidationResult(score=score, n_trials=n, n_failures=int(score * n), trials=[])

    def setUp(self):
        from v1_basic.confidence import ConfidenceScorer
        self.scorer = ConfidenceScorer()

    def test_perfect_score(self):
        bd = self.scorer.compute(self._loc(1.0, 20), self._av(1.0, 20))
        self.assertAlmostEqual(bd.combined_confidence, 1.0, places=2)

    def test_one_zero_gives_low_score(self):
        bd = self.scorer.compute(self._loc(0.0), self._av(1.0))
        self.assertLess(bd.combined_confidence, 0.3)

    def test_score_in_range(self):
        bd = self.scorer.compute(self._loc(0.6), self._av(0.8))
        self.assertGreaterEqual(bd.combined_confidence, 0.0)
        self.assertLessEqual(bd.combined_confidence, 1.0)

    def test_interpretation_high(self):
        bd = self.scorer.compute(self._loc(1.0, 20), self._av(1.0, 20))
        self.assertIn("HIGH", bd.interpretation)

    def test_no_attention_score(self):
        bd = self.scorer.compute(self._loc(0.8), self._av(0.8))
        self.assertIsNone(bd.attention_score)


# ─────────────────────────────────────────────── V2: GradientAnchorFinder ──

class TestV2GradientAnchorFinder(unittest.TestCase):

    def setUp(self):
        from v2_intermediate.anchor_finder import GradientAnchorFinder, _attention_concentration
        self.model = MockModel()
        self.finder = GradientAnchorFinder(self.model, top_k_fraction=0.30, context_chars=10)
        self._concentration_fn = _attention_concentration

    def test_concentration_uniform_is_low(self):
        scores = np.ones(50)
        c = self._concentration_fn(scores)
        self.assertLess(c, 0.1)

    def test_concentration_spike_is_high(self):
        scores = np.zeros(50)
        scores[25] = 1.0
        c = self._concentration_fn(scores)
        self.assertGreater(c, 0.9)

    def test_concentration_in_range(self):
        for _ in range(10):
            scores = np.random.rand(30)
            c = self._concentration_fn(scores)
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)

    def test_find_returns_anchor_and_concentration(self):
        text = "Total Due: $500.00 as noted."
        anchor, concentration = self.finder.find(text, "total", "$500.00")
        self.assertIsInstance(anchor, AnchorResult)
        self.assertGreaterEqual(concentration, 0.0)
        self.assertLessEqual(concentration, 1.0)


# ─────────────────────────────────────────────── V2: WeightedConfidenceScorer ──

class TestV2WeightedConfidenceScorer(unittest.TestCase):

    def _loc(self, score: float, n: int = 5) -> LocalizationResult:
        return LocalizationResult(score=score, n_trials=n, n_success=int(score * n), trials=[])

    def _av(self, score: float, n: int = 5) -> AnchorValidationResult:
        return AnchorValidationResult(score=score, n_trials=n, n_failures=int(score * n), trials=[])

    def setUp(self):
        from v2_intermediate.confidence import WeightedConfidenceScorer
        self.scorer = WeightedConfidenceScorer(0.20, 0.40, 0.40)

    def test_invalid_weights_raise(self):
        from v2_intermediate.confidence import WeightedConfidenceScorer
        with self.assertRaises(ValueError):
            WeightedConfidenceScorer(0.3, 0.3, 0.3)  # sums to 0.9

    def test_perfect_score(self):
        bd = self.scorer.compute(1.0, self._loc(1.0), self._av(1.0))
        self.assertAlmostEqual(bd.combined_confidence, 1.0, places=5)

    def test_zero_attention_uses_perturbation_scores(self):
        bd = self.scorer.compute(0.0, self._loc(1.0), self._av(1.0))
        # 0*0.20 + 1*0.40 + 1*0.40 = 0.80
        self.assertAlmostEqual(bd.combined_confidence, 0.80, places=5)

    def test_attention_score_stored(self):
        bd = self.scorer.compute(0.75, self._loc(0.8), self._av(0.6))
        self.assertAlmostEqual(bd.attention_score, 0.75, places=5)


# ─────────────────────────────────────────────── V3: EntropyScorer ──

class TestV3EntropyScorer(unittest.TestCase):

    def setUp(self):
        from v3_advanced.entropy_scorer import compute_generation_entropy
        self.fn = compute_generation_entropy

    def test_empty_returns_neutral(self):
        self.assertAlmostEqual(self.fn([]), 0.5)

    def test_uniform_is_uncertain(self):
        logits = [torch.ones(1000) for _ in range(5)]
        self.assertLess(self.fn(logits), 0.1)

    def test_peaked_is_confident(self):
        logits_list = []
        for _ in range(5):
            l = torch.full((1000,), -10.0)
            l[42] = 100.0
            logits_list.append(l)
        self.assertGreater(self.fn(logits_list), 0.9)

    def test_score_in_range(self):
        logits = [torch.randn(500) for _ in range(10)]
        score = self.fn(logits)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


# ─────────────────────────────────────────────── V3: AdvancedConfidenceScorer ──

class TestV3AdvancedConfidenceScorer(unittest.TestCase):

    def _loc(self, score: float, n: int = 10) -> LocalizationResult:
        trials = [LocalizationTrial(f"v{i}", "x", i < int(score * n)) for i in range(n)]
        return LocalizationResult(score=score, n_trials=n, n_success=int(score * n), trials=trials)

    def _av(self, score: float, n: int = 10) -> AnchorValidationResult:
        trials = [AnchorValidationTrial("ctx", "x", i < int(score * n)) for i in range(n)]
        return AnchorValidationResult(score=score, n_trials=n, n_failures=int(score * n), trials=trials)

    def setUp(self):
        from v3_advanced.confidence import AdvancedConfidenceScorer
        self.scorer = AdvancedConfidenceScorer(
            attention_weight=0.15, localization_weight=0.35,
            anchor_weight=0.35, entropy_weight=0.15, n_bootstrap=100,
        )

    def test_valid_weights_pass(self):
        from v3_advanced.confidence import AdvancedConfidenceScorer
        AdvancedConfidenceScorer(0.25, 0.25, 0.25, 0.25)  # sums to 1.0 — fine

    def test_invalid_weights_raise(self):
        from v3_advanced.confidence import AdvancedConfidenceScorer
        with self.assertRaises(ValueError):
            AdvancedConfidenceScorer(0.20, 0.30, 0.30, 0.15)  # sums to 0.95

    def test_perfect_score(self):
        bd = self.scorer.compute(1.0, self._loc(1.0), self._av(1.0), entropy_score=1.0)
        self.assertAlmostEqual(bd.combined_confidence, 1.0, places=5)

    def test_entropy_score_stored(self):
        bd = self.scorer.compute(0.7, self._loc(0.8), self._av(0.6), entropy_score=0.9)
        self.assertAlmostEqual(bd.entropy_score, 0.9, places=5)

    def test_confidence_interval_present(self):
        bd = self.scorer.compute(0.7, self._loc(0.8), self._av(0.6), entropy_score=0.8)
        self.assertIsNotNone(bd.confidence_interval)
        lo, hi = bd.confidence_interval
        self.assertLessEqual(lo, hi)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_mlm_modulation_improves_score(self):
        bd_no = self.scorer.compute(0.5, self._loc(0.5), self._av(0.5), entropy_score=0.5, mlm_score=None)
        bd_hi = self.scorer.compute(0.5, self._loc(0.5), self._av(0.5), entropy_score=0.5, mlm_score=1.0)
        self.assertGreater(bd_hi.combined_confidence, bd_no.combined_confidence)


# ─────────────────────────────────────────────── ExtractionResult.__str__ ──

class TestExtractionResultStr(unittest.TestCase):

    def _make(self, v3: bool = False) -> ExtractionResult:
        anchor = AnchorResult(
            text="Total Due: $500.00", start_char=0, end_char=18,
            token_attributions=[], method="value_match",
        )
        loc = LocalizationResult(score=0.8, n_trials=5, n_success=4, trials=[])
        av = AnchorValidationResult(score=0.6, n_trials=5, n_failures=3, trials=[])
        bd = ConfidenceBreakdown(
            localization_score=0.8, anchor_validation_score=0.6,
            combined_confidence=0.693,
            interpretation="MEDIUM    — extraction is plausible; consider manual review",
            entropy_score=0.85 if v3 else None,
            mlm_perplexity=42.3 if v3 else None,
            confidence_interval=(0.55, 0.82) if v3 else None,
        )
        return ExtractionResult(
            field="total", raw_text="Total Due: $500.00",
            value="$500.00", anchor=anchor,
            localization=loc, anchor_validation=av,
            confidence=0.693, confidence_breakdown=bd,
        )

    def test_basic_str_contains_value(self):
        s = str(self._make(v3=False))
        self.assertIn("$500.00", s)
        self.assertNotIn("Entropy", s)

    def test_v3_str_contains_extra_fields(self):
        s = str(self._make(v3=True))
        self.assertIn("Entropy", s)
        self.assertIn("MLM", s)
        self.assertIn("95%", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
