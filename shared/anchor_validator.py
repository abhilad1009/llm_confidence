"""
shared/anchor_validator.py — Stage 3: Anchor Context Validation (all versions).

Protocol
--------
For each trial:
  1. Identify all words in the anchor that are NOT part of the value.
  2. Replace ≥ 60 % of those context words with random vocabulary words,
     preserving stopwords so grammar stays plausible.
  3. Keep the original value token(s) exactly in place.
  4. Re-run extraction on the modified full text.
  5. Check whether the extracted result DIFFERS from the original (i.e. failed).

Score = fraction of trials in which extraction failed.

High score (≥ 0.6) → the anchor context is causally necessary.
Low score          → the model ignores context (position bias / hallucination).
"""

from __future__ import annotations

import re
import random
from typing import Optional
from tqdm import tqdm

from shared.schemas import AnchorValidationResult, AnchorValidationTrial
from shared.models import HFExtractionModel
from shared.localizer import _values_match


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "about",
    "and", "but", "or", "nor", "so", "yet", "both", "either", "not",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "we", "i", "me", "him", "her", "us", "them", "my", "your", "our",
    "their", "his",
}

_CORRUPTION_VOCAB = [
    "blorp", "zanthor", "flimsy", "grodent", "quilnar", "vexful", "snorble",
    "tramwick", "plebish", "quandor", "slomber", "freznick", "twiddle",
    "glopper", "snivvle", "crambit", "flobber", "glinker", "plonkard",
    "dribwick", "skumble", "troffin", "glazmur", "blonket", "sprixle",
    "crumbage", "flonger", "slibber", "graznik", "plumdor", "snarkle",
]


def _is_stopword(word: str) -> bool:
    return word.lower().strip(".,!?;:\"'()[]{}") in _STOPWORDS


def _tokenise_words(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _corrupt_context(
    anchor_text: str,
    value: str,
    corruption_rate: float,
    vocab: list[str],
) -> str:
    """Replace *corruption_rate* fraction of non-value, non-stopword words."""
    words = _tokenise_words(anchor_text)
    if not words:
        return anchor_text

    value_lower = value.lower()

    def is_value_token(word: str, start: int, end: int) -> bool:
        window = anchor_text[max(0, start - 5): end + 5].lower()
        return value_lower in window and word.lower() in value_lower

    corruptible = [
        i for i, (w, s, e) in enumerate(words)
        if not _is_stopword(w) and not is_value_token(w, s, e)
    ]

    n_corrupt = max(1, int(len(corruptible) * corruption_rate))
    to_corrupt = set(random.sample(corruptible, min(n_corrupt, len(corruptible))))

    result = []
    prev_end = 0
    for i, (word, start, end) in enumerate(words):
        result.append(anchor_text[prev_end:start])
        if i in to_corrupt:
            replacement = random.choice(vocab)
            if word and word[0].isupper():
                replacement = replacement.capitalize()
            result.append(replacement)
        else:
            result.append(word)
        prev_end = end
    result.append(anchor_text[prev_end:])
    return "".join(result)


class AnchorValidator:
    """
    Stage-3 anchor validation tester (shared by all versions).

    Parameters
    ----------
    model : HFExtractionModel
    n_trials : int
    corruption_rate : float
        Fraction of corruptible context words to replace (default 0.65).
    match_threshold : float
    """

    def __init__(
        self,
        model: HFExtractionModel,
        n_trials: int = 10,
        corruption_rate: float = 0.65,
        match_threshold: float = 0.8,
    ) -> None:
        self.model = model
        self.n_trials = n_trials
        self.corruption_rate = corruption_rate
        self.match_threshold = match_threshold

        try:
            extra = model.vocab_words[:200]
            self._vocab = _CORRUPTION_VOCAB + extra
        except Exception:
            self._vocab = _CORRUPTION_VOCAB

    def validate(
        self,
        full_text: str,
        field: str,
        original_value: str,
        anchor_text: str,
        anchor_start: int,
        anchor_end: int,
    ) -> AnchorValidationResult:
        """Run anchor validation trials and return AnchorValidationResult."""
        trials: list[AnchorValidationTrial] = []
        n_failures = 0

        for _ in tqdm(range(self.n_trials), desc="  Anchor validation trials", leave=False):
            corrupted_anchor = _corrupt_context(
                anchor_text, original_value, self.corruption_rate, self._vocab
            )
            modified_text = (
                full_text[:anchor_start]
                + corrupted_anchor
                + full_text[anchor_end:]
            )

            try:
                extracted = self.model.extract(modified_text, field)
            except Exception:
                extracted = ""

            extraction_failed = not _values_match(
                extracted, original_value, self.match_threshold
            )
            if extraction_failed:
                n_failures += 1

            trials.append(AnchorValidationTrial(
                corrupted_anchor=corrupted_anchor,
                extracted_value=extracted,
                extraction_failed=extraction_failed,
            ))

        score = n_failures / len(trials) if trials else 0.0
        return AnchorValidationResult(
            score=score,
            n_trials=len(trials),
            n_failures=n_failures,
            trials=trials,
        )
