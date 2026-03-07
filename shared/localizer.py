"""
shared/localizer.py — Stage 2: Value Localization Testing (all versions).

Protocol
--------
For each trial:
  1. Generate a random value of the same type as the original.
  2. Replace the original value with the random value inside the anchor.
  3. Re-run extraction on the modified full text.
  4. Check whether the extracted result matches the injected random value.

Score = fraction of successful trials.

A high score (≥ 0.7) means the model reliably follows the anchor location.
A low score suggests the model is hallucinating or using the wrong location.
"""

from __future__ import annotations

import re
from typing import Optional
from tqdm import tqdm

from shared.schemas import LocalizationResult, LocalizationTrial
from shared.models import HFExtractionModel
from shared.value_generator import ValueGenerator


# ──────────────────────────────────────────────────────────────────────────────
# Matching helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalise(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.lower()).strip()


def _values_match(extracted: str, expected: str, threshold: float = 0.8) -> bool:
    """
    Return True if *extracted* closely matches *expected*.

    Uses exact match after normalisation, substring match, and character-overlap
    ratio for near-misses (numbers, dates).
    """
    e = _normalise(extracted)
    x = _normalise(expected)
    if not e or not x:
        return False
    if e == x:
        return True
    if x in e or e in x:
        return True
    set_e = set(e.split())
    set_x = set(x.split())
    if set_e and set_x:
        overlap = len(set_e & set_x) / max(len(set_e), len(set_x))
        return overlap >= threshold
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Value substitution in text
# ──────────────────────────────────────────────────────────────────────────────

def _substitute_value(
    full_text: str,
    anchor_text: str,
    original_value: str,
    new_value: str,
    anchor_start: int,
    anchor_end: int,
) -> str:
    """Replace *original_value* with *new_value* inside the anchor region."""
    anchor_region = full_text[anchor_start:anchor_end]

    pattern = re.escape(original_value)
    new_anchor, count = re.subn(pattern, new_value, anchor_region, count=1)
    if count == 0:
        new_anchor, count = re.subn(pattern, new_value, anchor_region, count=1, flags=re.IGNORECASE)
    if count == 0:
        num_pattern = r"\b\d[\d,.']*\b"
        new_anchor, count = re.subn(num_pattern, new_value, anchor_region, count=1)
    if count == 0:
        new_anchor = anchor_region + f" [{new_value}]"

    return full_text[:anchor_start] + new_anchor + full_text[anchor_end:]


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class Localizer:
    """
    Stage-2 localisation tester (shared by all versions).

    Parameters
    ----------
    model : HFExtractionModel
    value_generator : ValueGenerator
    n_trials : int
        Number of random-value injection trials (10–20 recommended).
    match_threshold : float
        Jaccard threshold for counting a trial as successful.
    """

    def __init__(
        self,
        model: HFExtractionModel,
        value_generator: Optional[ValueGenerator] = None,
        n_trials: int = 10,
        match_threshold: float = 0.8,
    ) -> None:
        self.model = model
        self.value_gen = value_generator or ValueGenerator()
        self.n_trials = n_trials
        self.match_threshold = match_threshold

    def test(
        self,
        full_text: str,
        field: str,
        original_value: str,
        anchor_text: str,
        anchor_start: int,
        anchor_end: int,
    ) -> LocalizationResult:
        """Run localisation trials and return a LocalizationResult."""
        random_values = self.value_gen.generate(
            original_value, n=self.n_trials, exclude=[original_value]
        )
        while len(random_values) < self.n_trials:
            extras = self.value_gen.generate(original_value, n=5)
            random_values.extend(extras)
        random_values = random_values[:self.n_trials]

        trials: list[LocalizationTrial] = []
        n_success = 0

        for injected in tqdm(random_values, desc="  Localisation trials", leave=False):
            modified_text = _substitute_value(
                full_text, anchor_text, original_value, injected,
                anchor_start, anchor_end
            )
            try:
                extracted = self.model.extract(modified_text, field)
            except Exception:
                extracted = ""

            success = _values_match(extracted, injected, self.match_threshold)
            if success:
                n_success += 1

            trials.append(LocalizationTrial(
                injected_value=injected,
                extracted_value=extracted,
                success=success,
            ))

        score = n_success / len(trials) if trials else 0.0
        return LocalizationResult(
            score=score,
            n_trials=len(trials),
            n_success=n_success,
            trials=trials,
        )
