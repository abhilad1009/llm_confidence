"""
benchmark/data/cord.py — CORD dataset loader.

CORD (Consolidated Receipt Dataset) contains receipt images with OCR text
and structured JSON annotations for fields like menu items, prices, totals.

We reconstruct the receipt text from OCR tokens and map each ground-truth
field to BenchmarkExample(source_text=receipt_text, query=field_name, gold=value).
"""

from __future__ import annotations

import json
import random
from benchmark.schemas import BenchmarkExample


# Fields we extract from CORD's ground_truth.gt_parse
_TARGET_FIELDS = [
    "total.total_price",
    "total.subtotal_price",
    "total.tax_price",
    "total.discount_price",
]


def load_cord(
    n: int = 200,
    *,
    split: str = "test",
    seed: int = 42,
) -> list[BenchmarkExample]:
    """
    Load CORD examples.

    Args:
        n: Maximum number of examples to return.
        split: 'test' (uses test split) or 'dev' (uses validation split).
        seed: Random seed for sampling.

    Returns:
        List of BenchmarkExample where each is a (receipt_text, field_name, value) triple.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("Install the 'datasets' package: uv add datasets")

    hf_split = "test" if split == "test" else "validation"
    ds = load_dataset("naver-clova-ix/cord-v2", split=hf_split)

    rng = random.Random(seed)
    examples: list[BenchmarkExample] = []

    for doc_idx, doc in enumerate(ds):
        # Reconstruct receipt text from OCR annotations
        receipt_text = _extract_receipt_text(doc)
        if not receipt_text.strip():
            continue

        # Parse ground truth JSON
        gt_raw = doc.get("ground_truth", "{}")
        try:
            gt = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
        except (json.JSONDecodeError, TypeError):
            continue

        gt_parse = gt.get("gt_parse", {})
        if not gt_parse:
            continue

        # Extract target fields
        for field_path in _TARGET_FIELDS:
            value = _get_nested(gt_parse, field_path)
            if value is None:
                continue
            value_str = str(value).strip()
            if not value_str:
                continue

            # Human-readable field name for the query
            field_label = _field_label(field_path)

            examples.append(BenchmarkExample(
                id=f"cord_doc{doc_idx}_{field_path.replace('.', '_')}",
                source_text=receipt_text,
                query=f"What is the {field_label}?",
                gold_answer=_normalize(value_str),
                is_answerable=True,
                dataset="cord",
                split=split,
                metadata={"doc_idx": doc_idx, "field_path": field_path},
            ))

    rng.shuffle(examples)
    return examples[:n]


def _extract_receipt_text(doc: dict) -> str:
    """Reconstruct receipt text from OCR word annotations."""
    # CORD stores words in 'ground_truth' or directly as text
    # Try the simple text field first
    if "text" in doc and doc["text"]:
        return doc["text"]

    # Reconstruct from ground_truth OCR
    gt_raw = doc.get("ground_truth", "{}")
    try:
        gt = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
    except (json.JSONDecodeError, TypeError):
        return ""

    words = []
    # gt_parse contains structured fields; extract all visible text
    gt_parse = gt.get("gt_parse", {})
    _collect_text_values(gt_parse, words)
    return " ".join(words)


def _collect_text_values(obj: object, out: list[str]) -> None:
    """Recursively collect all string values from a nested dict/list."""
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_text_values(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_text_values(item, out)


def _get_nested(d: dict, path: str) -> object | None:
    """Get a value at a dot-separated path, e.g. 'total.total_price'."""
    parts = path.split(".")
    current = d
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def _field_label(field_path: str) -> str:
    labels = {
        "total.total_price": "total price",
        "total.subtotal_price": "subtotal price",
        "total.tax_price": "tax amount",
        "total.discount_price": "discount amount",
    }
    return labels.get(field_path, field_path.replace(".", " ").replace("_", " "))


def _normalize(text: str) -> str:
    return text.strip().lower()


def cord_is_correct(predicted: str, gold: str, *, threshold: float = 0.7) -> bool:
    """Check correctness via exact match or character F1 >= threshold."""
    pred = _normalize(predicted)
    gold_n = _normalize(gold)
    if pred == gold_n:
        return True
    # Strip currency symbols for comparison
    pred_clean = pred.replace("$", "").replace("€", "").replace("£", "").strip()
    gold_clean = gold_n.replace("$", "").replace("€", "").replace("£", "").strip()
    if pred_clean == gold_clean:
        return True
    return _char_f1(pred_clean, gold_clean) >= threshold


def _char_f1(pred: str, gold: str) -> float:
    pred_chars = set(pred)
    gold_chars = set(gold)
    common = pred_chars & gold_chars
    if not common:
        return 0.0
    precision = len(common) / len(pred_chars) if pred_chars else 0.0
    recall = len(common) / len(gold_chars) if gold_chars else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
