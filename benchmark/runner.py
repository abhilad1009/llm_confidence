"""
benchmark/runner.py — Run all confidence methods across datasets.

Usage:
  python benchmark/runner.py --dataset squad --method v2,logprob_mean --n-examples 50
  python benchmark/runner.py --dataset all --method all --n-examples 500
  python benchmark/runner.py --dataset squad --method v2 --ablation perturbation_only

The results are saved to confidence_results_benchmark.json for reporting.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.data.cord import cord_is_correct, load_cord
from benchmark.data.funsd import funsd_is_correct, load_funsd
from benchmark.data.squad import load_squad, squad_is_correct
from benchmark.schemas import BenchmarkExample, MethodResult

_CORRECTNESS_FN = {
    "squad": squad_is_correct,
    "funsd": funsd_is_correct,
    "cord": cord_is_correct,
}

_ABLATION_WEIGHTS = {
    # (attention_weight, localization_weight, anchor_weight)
    "perturbation_only": (0.0, 0.5, 0.5),
    "no_localization": (0.25, 0.0, 0.75),
    "no_anchor_validation": (0.25, 0.75, 0.0),
    "full_v2": (0.20, 0.40, 0.40),
}


def load_dataset(name: str, n: int, split: str = "test") -> list[BenchmarkExample]:
    loaders = {
        "squad": lambda: load_squad(n, split=split),
        "funsd": lambda: load_funsd(n, split=split),
        "cord": lambda: load_cord(n, split=split),
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders)}")
    print(f"[runner] Loading {name} ({n} examples) …")
    examples = loaders[name]()
    print(f"[runner] Loaded {len(examples)} examples from {name}.")
    return examples


def build_perturbation_method(
    method_name: str,
    model_name: str,
    n_trials: int,
    ablation: str | None,
    dataset_name: str,
):
    """Build one of v1/v2/v3 extractors wrapped as a BaseConfidenceMethod."""
    from benchmark.baselines.base import BaseConfidenceMethod

    correct_fn = _CORRECTNESS_FN[dataset_name]

    class PerturbationWrapper(BaseConfidenceMethod):
        def __init__(self, extractor, name, correct_fn):
            self._extractor = extractor
            self.name = name
            self._correct_fn = correct_fn

        def predict(self, example: BenchmarkExample) -> MethodResult:
            try:
                result = self._extractor.extract(example.source_text, example.query)
                token_attrs = None
                if result.anchor.token_attributions:
                    token_attrs = [
                        {
                            "token": ta.token,
                            "score": ta.score,
                            "char_start": ta.char_start,
                            "char_end": ta.char_end,
                        }
                        for ta in result.anchor.token_attributions
                    ]
                is_correct = self._correct_fn(str(result.value), example.gold_answer)
                return MethodResult(
                    example_id=example.id,
                    method=self.name,
                    confidence=result.confidence,
                    extracted_value=str(result.value),
                    is_correct=is_correct,
                    token_attributions=token_attrs,
                )
            except Exception as e:
                print(f"[runner] ERROR on {example.id}: {e}")
                return MethodResult(
                    example_id=example.id,
                    method=self.name,
                    confidence=0.0,
                    extracted_value="",
                    is_correct=False,
                )

    if method_name == "v1" or (method_name == "v2" and ablation == "v1_baseline"):
        from v1_basic.extractor import BasicConfidenceExtractor

        extractor = BasicConfidenceExtractor(model_name=model_name, n_trials=n_trials)
        return PerturbationWrapper(
            extractor, "v1_baseline" if ablation else "v1", correct_fn
        )

    if method_name == "v2":
        from v2_intermediate.extractor import IntermediateConfidenceExtractor

        weights = _ABLATION_WEIGHTS.get(ablation or "full_v2", (0.20, 0.40, 0.40))
        extractor = IntermediateConfidenceExtractor(
            model_name=model_name,
            n_trials=n_trials,
            attention_weight=weights[0],
            localization_weight=weights[1],
            anchor_weight=weights[2],
        )
        name = ablation or "v2"
        return PerturbationWrapper(extractor, name, correct_fn)

    if method_name == "v3":
        from v3_advanced.extractor import AdvancedConfidenceExtractor

        extractor = AdvancedConfidenceExtractor(
            model_name=model_name, n_trials=n_trials
        )
        return PerturbationWrapper(extractor, "v3", correct_fn)

    raise ValueError(f"Unknown perturbation method: {method_name}")


def build_baseline_method(method_name: str, model, dataset_name: str):
    """Build a logprob/self-consistency/attention baseline."""
    correct_fn = _CORRECTNESS_FN[dataset_name]

    if method_name.startswith("logprob"):
        from benchmark.baselines.logprob import LogprobConfidence

        variant = method_name.split("_", 1)[1] if "_" in method_name else "mean"
        return LogprobConfidence(model, variant=variant, correctness_fn=correct_fn)

    if method_name == "self_consistency":
        from benchmark.baselines.self_consistency import SelfConsistencyConfidence

        return SelfConsistencyConfidence(model, correctness_fn=correct_fn)

    if method_name == "attention_entropy":
        from benchmark.baselines.attention_entropy import AttentionEntropyConfidence

        return AttentionEntropyConfidence(model, correctness_fn=correct_fn)

    raise ValueError(f"Unknown baseline method: {method_name}")


def run_method(
    method,
    examples: list[BenchmarkExample],
    *,
    verbose: bool = False,
) -> list[MethodResult]:
    """Run a method on all examples, with progress logging."""
    results: list[MethodResult] = []
    n = len(examples)
    for i, ex in enumerate(examples, 1):
        t0 = time.time()
        result = method.predict(ex)
        elapsed = time.time() - t0
        results.append(result)
        if verbose or i % 10 == 0:
            status = "CORRECT" if result.is_correct else "WRONG"
            print(
                f"  [{i}/{n}] {ex.id[:40]:<40} "
                f"conf={result.confidence:.3f} {status} ({elapsed:.1f}s)"
            )
    return results


def save_results(results: list[MethodResult], path: str) -> None:
    existing: list[dict] = []
    p = Path(path)
    if p.exists():
        with open(p) as f:
            existing = json.load(f)
    # Convert to dicts
    new_records = [dataclasses.asdict(r) for r in results]
    combined = existing + new_records
    with open(p, "w") as f:
        json.dump(combined, f, indent=2)
    print(
        f"[runner] Saved {len(new_records)} results to {path} ({len(combined)} total)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run confidence benchmark")
    parser.add_argument(
        "--dataset",
        default="squad",
        help="Dataset(s): squad, funsd, cord, or all (comma-separated)",
    )
    parser.add_argument(
        "--method",
        default="all",
        help="Method(s): v1, v2, v3, logprob_mean, logprob_min, logprob_norm, "
        "self_consistency, attention_entropy, or all (comma-separated)",
    )
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument(
        "--ablation",
        default=None,
        choices=list(_ABLATION_WEIGHTS) + ["v1_baseline", None],
        help="Ablation variant (v2 only)",
    )
    parser.add_argument("--output", default="confidence_results_benchmark.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Expand 'all'
    all_datasets = ["squad", "funsd", "cord"]
    all_methods = [
        "v1",
        "v2",
        "v3",
        "logprob_mean",
        "logprob_min",
        "logprob_norm",
        "self_consistency",
        "attention_entropy",
    ]
    datasets = (
        all_datasets
        if args.dataset == "all"
        else [d.strip() for d in args.dataset.split(",")]
    )
    methods = (
        all_methods
        if args.method == "all"
        else [m.strip() for m in args.method.split(",")]
    )

    _perturbation = {"v1", "v2", "v3"}
    _baseline = set(all_methods) - _perturbation

    # Load a shared HF model for baselines (if any baseline methods requested)
    shared_model = None
    if any(m in _baseline for m in methods):
        from shared.models import HFExtractionModel

        shared_model = HFExtractionModel(model_name=args.model)

    all_results: list[MethodResult] = []

    for dataset_name in datasets:
        examples = load_dataset(dataset_name, args.n_examples)

        for method_name in methods:
            print(f"\n[runner] Running {method_name!r} on {dataset_name!r} …")

            if method_name in _perturbation:
                method = build_perturbation_method(
                    method_name,
                    args.model,
                    args.n_trials,
                    args.ablation,
                    dataset_name,
                )
            else:
                if shared_model is None:
                    from shared.models import HFExtractionModel

                    shared_model = HFExtractionModel(model_name=args.model)
                method = build_baseline_method(method_name, shared_model, dataset_name)

            results = run_method(method, examples, verbose=args.verbose)
            all_results.extend(results)
            correct = sum(r.is_correct for r in results)
            mean_conf = (
                sum(r.confidence for r in results) / len(results) if results else 0
            )
            print(
                f"[runner] {method_name}/{dataset_name}: "
                f"accuracy={correct}/{len(results)} "
                f"mean_conf={mean_conf:.3f}"
            )

    save_results(all_results, args.output)


if __name__ == "__main__":
    main()
