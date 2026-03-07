"""
benchmark/data/squad.py — SQuAD 2.0 dataset loader.

Maps (context, question, answer, is_answerable) to BenchmarkExample.
Stratifies: 50% answerable, 50% unanswerable from the validation split.
"""

from __future__ import annotations

import random
from typing import Sequence

from benchmark.schemas import BenchmarkExample


def load_squad(
    n: int = 500,
    *,
    split: str = "test",
    seed: int = 42,
) -> list[BenchmarkExample]:
    """
    Load SQuAD 2.0 examples.

    Args:
        n: Total number of examples to return.
        split: 'test' (uses validation set) or 'dev' (uses subset for tuning).
        seed: Random seed for sampling.

    Returns:
        List of BenchmarkExample, stratified by answerability.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("Install the 'datasets' package: uv add datasets")

    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    rng = random.Random(seed)

    answerable = [ex for ex in ds if len(ex["answers"]["text"]) > 0]
    unanswerable = [ex for ex in ds if len(ex["answers"]["text"]) == 0]

    n_each = n // 2
    answerable = rng.sample(answerable, min(n_each, len(answerable)))
    unanswerable = rng.sample(unanswerable, min(n_each, len(unanswerable)))

    all_examples = answerable + unanswerable
    rng.shuffle(all_examples)

    results: list[BenchmarkExample] = []
    for i, ex in enumerate(all_examples):
        gold = ex["answers"]["text"][0] if ex["answers"]["text"] else ""
        results.append(BenchmarkExample(
            id=f"squad_{ex['id']}",
            source_text=ex["context"],
            query=ex["question"],
            gold_answer=_normalize(gold),
            is_answerable=len(ex["answers"]["text"]) > 0,
            dataset="squad",
            split=split,
            metadata={
                "title": ex.get("title", ""),
                "original_id": ex["id"],
            },
        ))

    return results


def _normalize(text: str) -> str:
    """Normalize answer text for comparison."""
    return text.strip().lower()


def squad_is_correct(predicted: str, gold: str, *, threshold: float = 0.5) -> bool:
    """
    Check if predicted matches gold using exact match and token F1.

    Returns True if exact match OR token F1 >= threshold.
    """
    pred = _normalize(predicted)
    gold_n = _normalize(gold)
    if not gold_n:
        # Unanswerable: correct if model returns empty or abstains
        return pred in ("", "unanswerable", "n/a", "none", "no answer")
    if pred == gold_n:
        return True
    return _token_f1(pred, gold_n) >= threshold


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(gold_tokens) if gold_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
