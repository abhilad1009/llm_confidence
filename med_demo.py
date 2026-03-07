"""
med_demo.py — Extract structured fields from a clinical note.

Usage:
    python med_demo.py                   # runs all 3 versions
    python med_demo.py --version v1      # basic (heuristic)
    python med_demo.py --version v2      # intermediate (gradient)
    python med_demo.py --version v3      # advanced (entropy + MLM)
    python med_demo.py --trials 5        # fewer trials (faster)

Medical notes are a good stress test because:
  - patient_name    : unambiguous, appears once
  - primary_diagnosis : requires clinical reasoning from context
  - furosemide_dosage : multiple medications — tests specificity
  - visit_date      : date extraction
"""

from __future__ import annotations
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

CLINICAL_NOTE = """
PATIENT VISIT SUMMARY
═══════════════════════════════════════════════════════════

Patient Name   : Robert Alan Prescott
Date of Birth  : 1968-07-22
Patient ID     : 847291-MED
Visit Date     : 2024-02-08
Attending      : Dr. Sarah L. Kim, MD — Internal Medicine

─────────────────────────────────────────────────────────
CHIEF COMPLAINT
─────────────────────────────────────────────────────────
Patient presents with persistent shortness of breath,
fatigue on exertion, and bilateral ankle oedema over the
past three weeks. Reports two-pillow orthopnea.

─────────────────────────────────────────────────────────
ASSESSMENT & DIAGNOSIS
─────────────────────────────────────────────────────────
Primary Diagnosis: Congestive Heart Failure (CHF), NYHA Class II

The clinical presentation, BNP elevation (487 pg/mL), and
echocardiographic findings (EF 38 %) are consistent with
newly diagnosed systolic heart failure.

Secondary: Hypertension (pre-existing, controlled)
           Type 2 Diabetes Mellitus (stable)

─────────────────────────────────────────────────────────
MEDICATIONS PRESCRIBED
─────────────────────────────────────────────────────────
1. Furosemide 40 mg — once daily (loop diuretic)
2. Lisinopril 10 mg — once daily (ACE inhibitor)
3. Carvedilol 6.25 mg — twice daily (beta-blocker)
4. Spironolactone 25 mg — once daily (aldosterone antagonist)

─────────────────────────────────────────────────────────
FOLLOW-UP
─────────────────────────────────────────────────────────
Return in 2 weeks for repeat BMP and weight check.
Restrict sodium to < 2 g/day.  Daily weight monitoring.

Electronically signed: Dr. Sarah L. Kim, MD  |  2024-02-08 14:32
"""

FIELDS = ["patient_name", "primary_diagnosis", "visit_date", "furosemide_dosage"]


def _bar(score: float, width: int = 18) -> str:
    filled = int(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {score:.0%}"


def _print_attribution_heatmap(results: dict) -> None:
    """Print token attribution heatmaps for v2/v3 results (gradient-based anchor)."""
    from benchmark.visualize import print_attribution_heatmap
    for field, r in results.items():
        if r.anchor.token_attributions:
            print_attribution_heatmap(
                r.anchor.token_attributions,
                title=f"Token Attribution — {field} (anchor method: {r.anchor.method})",
            )


def print_report(version_label: str, results: dict, show_heatmap: bool = False) -> None:
    print(f"\n{'═' * 64}")
    print(f"  {version_label} — MEDICAL RECORD CONFIDENCE REPORT")
    print(f"{'═' * 64}")
    for field, r in results.items():
        b = r.confidence_breakdown
        print(f"\n  Field     : {field}")
        print(f"  Value     : {r.value}")
        print(f"  Confidence: {r.confidence:.1%}  {_bar(r.confidence)}")
        print(f"  Verdict   : {b.interpretation}")
        if b.attention_score is not None:
            print(f"  Attention : {b.attention_score:.1%}")
        print(f"  Loc score : {b.localization_score:.1%} | "
              f"Anchor score: {b.anchor_validation_score:.1%}")
        if b.entropy_score is not None:
            print(f"  Entropy   : {b.entropy_score:.1%}")
        if b.confidence_interval is not None:
            lo, hi = b.confidence_interval
            print(f"  95% CI    : [{lo:.1%}, {hi:.1%}]")
    print(f"{'═' * 64}")

    if show_heatmap:
        _print_attribution_heatmap(results)


def print_comparison(all_results: dict[str, dict]) -> None:
    print(f"\n{'═' * 64}")
    print("  VERSION COMPARISON")
    print(f"{'═' * 64}")
    print(f"  {'FIELD':<25} {'V1 BASIC':>10} {'V2 INTERM':>10} {'V3 ADV':>10}")
    print(f"  {'-'*57}")
    for field in FIELDS:
        row = f"  {field:<25}"
        for label, results in all_results.items():
            if field in results:
                row += f" {results[field].confidence:>10.1%}"
            else:
                row += f" {'N/A':>10}"
        print(row)
    print(f"{'═' * 64}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical record confidence extraction demo")
    parser.add_argument("--version", choices=["v1", "v2", "v3", "all"], default="all")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--no-mlm", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--show-heatmap", action="store_true",
                        help="Show token attribution heatmap for v2/v3 (gradient anchor)")
    args = parser.parse_args()

    load_4bit = None if not args.no_4bit else False
    all_results: dict[str, dict] = {}
    versions_to_run = ["v1", "v2", "v3"] if args.version == "all" else [args.version]

    if "v1" in versions_to_run:
        from v1_basic.extractor import BasicConfidenceExtractor
        e = BasicConfidenceExtractor(
            model_name=args.model, load_in_4bit=load_4bit,
            n_trials=args.trials, corruption_rate=0.60
        )
        results = e.extract_fields(CLINICAL_NOTE, FIELDS)
        print_report("v1_basic (heuristic)", results)
        all_results["v1"] = results

    if "v2" in versions_to_run:
        from v2_intermediate.extractor import IntermediateConfidenceExtractor
        e = IntermediateConfidenceExtractor(
            model_name=args.model, load_in_4bit=load_4bit,
            n_trials=args.trials, corruption_rate=0.60
        )
        results = e.extract_fields(CLINICAL_NOTE, FIELDS)
        print_report("v2_intermediate (gradient)", results, show_heatmap=args.show_heatmap)
        all_results["v2"] = results

    if "v3" in versions_to_run:
        from v3_advanced.extractor import AdvancedConfidenceExtractor
        mlm_name = None if args.no_mlm else "distilbert-base-uncased"
        e = AdvancedConfidenceExtractor(
            model_name=args.model, mlm_model_name=mlm_name,
            load_in_4bit=load_4bit, n_trials=args.trials, corruption_rate=0.60
        )
        results = e.extract_fields(CLINICAL_NOTE, FIELDS)
        print_report("v3_advanced (entropy + MLM)", results, show_heatmap=args.show_heatmap)
        all_results["v3"] = results

    if len(all_results) > 1:
        print_comparison(all_results)


if __name__ == "__main__":
    main()
