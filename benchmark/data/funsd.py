"""
benchmark/data/funsd.py — FUNSD dataset loader.

FUNSD (Form Understanding in Noisy Scanned Documents) contains form images
with OCR text and NER annotations for question, answer, and header entities.
We reconstruct the document text from word tokens and map linked question-answer
pairs to BenchmarkExample(source_text=doc_text, query=key_text, gold=value_text).
"""

from __future__ import annotations

import random
from benchmark.schemas import BenchmarkExample


def load_funsd(
    n: int = 200,
    *,
    split: str = "test",
    seed: int = 42,
) -> list[BenchmarkExample]:
    """
    Load FUNSD examples.

    Args:
        n: Maximum number of examples to return.
        split: 'test' (uses test split) or 'dev' (uses training split subset).
        seed: Random seed for sampling.

    Returns:
        List of BenchmarkExample where each is a (form_text, question_field, answer_value) triple.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("Install the 'datasets' package: uv add datasets")

    hf_split = "test" if split == "test" else "train"
    ds = load_dataset("nielsr/funsd", split=hf_split, trust_remote_code=True)

    rng = random.Random(seed)
    examples: list[BenchmarkExample] = []

    for doc_idx, doc in enumerate(ds):
        words = doc["words"]
        bboxes = doc["bboxes"]
        ner_tags = doc["ner_tags"]

        # Reconstruct full document text (space-joined words)
        doc_text = " ".join(words)

        # Group entities by tag:
        # FUNSD NER tags: 0=other, 1=question, 2=answer, 3=header
        # We look for question-answer pairs linked by adjacency in the annotation
        questions: list[dict] = []
        answers: list[dict] = []
        current_q: list[str] = []
        current_a: list[str] = []
        current_tag = None

        for word, tag in zip(words, ner_tags):
            if tag == 1:  # question
                if current_tag == 2 and current_a:
                    answers.append({"text": " ".join(current_a)})
                    current_a = []
                if current_tag != 1:
                    if current_q:
                        questions.append({"text": " ".join(current_q)})
                    current_q = []
                current_q.append(word)
                current_tag = 1
            elif tag == 2:  # answer
                if current_tag != 2:
                    if current_a:
                        answers.append({"text": " ".join(current_a)})
                    current_a = []
                current_a.append(word)
                current_tag = 2
            else:
                if current_tag == 1 and current_q:
                    questions.append({"text": " ".join(current_q)})
                    current_q = []
                if current_tag == 2 and current_a:
                    answers.append({"text": " ".join(current_a)})
                    current_a = []
                current_tag = tag

        # Flush
        if current_q:
            questions.append({"text": " ".join(current_q)})
        if current_a:
            answers.append({"text": " ".join(current_a)})

        # Pair questions with answers (positional: each Q pairs with next A)
        for q_idx, (q, a) in enumerate(zip(questions, answers)):
            q_text = q["text"].strip()
            a_text = a["text"].strip()
            if not q_text or not a_text:
                continue
            examples.append(BenchmarkExample(
                id=f"funsd_doc{doc_idx}_q{q_idx}",
                source_text=doc_text,
                query=q_text,
                gold_answer=_normalize(a_text),
                is_answerable=True,
                dataset="funsd",
                split=split,
                metadata={"doc_idx": doc_idx},
            ))

    rng.shuffle(examples)
    return examples[:n]


def _normalize(text: str) -> str:
    return text.strip().lower()


def funsd_is_correct(predicted: str, gold: str, *, threshold: float = 0.5) -> bool:
    """Check correctness via exact match or character F1 >= threshold."""
    pred = _normalize(predicted)
    gold_n = _normalize(gold)
    if pred == gold_n:
        return True
    return _char_f1(pred, gold_n) >= threshold


def _char_f1(pred: str, gold: str) -> float:
    """Character-level F1 — useful for noisy OCR text."""
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
