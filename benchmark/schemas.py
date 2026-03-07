"""
benchmark/schemas.py — Data structures for the benchmarking harness.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkExample:
    """A single evaluation example, normalised across datasets."""
    id: str
    source_text: str
    query: str              # Field name (FUNSD/CORD) or question (SQuAD)
    gold_answer: str        # Normalised gold string; empty string if unanswerable
    is_answerable: bool     # False for SQuAD unanswerable questions
    dataset: str            # 'squad', 'funsd', 'cord'
    split: str = "test"     # 'dev' or 'test'
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodResult:
    """Result of running a single confidence method on a single example."""
    example_id: str
    method: str             # e.g. 'v1', 'v2', 'v3', 'logprob_mean', 'self_consistency'
    confidence: float       # [0, 1]
    extracted_value: str
    is_correct: bool        # True if extracted_value matches gold_answer
    # Per-token attributions from AnchorResult (v2/v3 only)
    token_attributions: list[dict] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
