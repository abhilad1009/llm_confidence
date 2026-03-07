"""
invoice_demo.py — Extract structured fields from an invoice.

Usage:
    python invoice_demo.py                   # runs all 3 versions
    python invoice_demo.py --version v1      # basic (heuristic)
    python invoice_demo.py --version v2      # intermediate (gradient)
    python invoice_demo.py --version v3      # advanced (entropy + MLM)
    python invoice_demo.py --trials 5        # fewer trials (faster)
    python invoice_demo.py --no-mlm          # skip MLM scorer in v3

Outputs a side-by-side confidence comparison when running all 3 versions.
"""

from __future__ import annotations
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# ──────────────────────────────────────────────────────────────────────────────
# Sample invoice
# ──────────────────────────────────────────────────────────────────────────────

INVOICE_TEXT = """
INVOICE

Vendor:   Apex Office Supplies Ltd.
          123 Commerce Street
          New York, NY 10001
          Tel: (212) 555-0187

Bill To:  Momentum Analytics Inc.
          47 Park Avenue, Suite 800
          New York, NY 10016

Invoice Number : INV-2024-00783
Invoice Date   : March 15, 2024
Due Date       : April 14, 2024
Payment Terms  : Net 30

─────────────────────────────────────────────────────
DESCRIPTION                      QTY    UNIT    TOTAL
─────────────────────────────────────────────────────
Premium A4 Paper (500 sheets)     20   $12.50   $250.00
Whiteboard Markers (Set of 8)     15    $8.99   $134.85
Ergonomic Mouse Pad               10   $19.99   $199.90
USB-C Charging Hub                 5   $49.99   $249.95
─────────────────────────────────────────────────────
                                        Subtotal  $834.70
                                       Tax (8%)   $66.78
                                          TOTAL  $901.48
─────────────────────────────────────────────────────

Payment Method: Wire Transfer
Bank: First National Bank
Account: 4521-XXXX-XXXX-7823
Routing: 021000089

Please include the invoice number in your payment reference.
Late payments incur a 1.5% monthly fee.

Authorised by: Jennifer Walsh, Accounts Manager
"""

FIELDS = ["invoice_number", "invoice_total", "invoice_date", "vendor_name"]


# ──────────────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bar(score: float, width: int = 18) -> str:
    filled = int(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {score:.0%}"


def _tier(score: float) -> str:
    if score >= 0.80: return "HIGH   "
    if score >= 0.55: return "MEDIUM "
    if score >= 0.30: return "LOW    "
    return "VERY_LOW"


def print_report(version_label: str, results: dict) -> None:
    print(f"\n{'═' * 72}")
    print(f"  {version_label} — EXTRACTION CONFIDENCE REPORT")
    print(f"{'═' * 72}")
    print(f"  {'FIELD':<22} {'VALUE':<22} {'CONF':>6}  BAR")
    print(f"  {'-'*70}")
    for field, r in results.items():
        print(
            f"  {field:<22} {str(r.value):<22} "
            f"{r.confidence:>5.1%}  {_bar(r.confidence)}  [{_tier(r.confidence)}]"
        )
    print(f"{'═' * 72}")

    print(f"\n  DETAILED BREAKDOWN")
    print(f"  {'-'*70}")
    for field, r in results.items():
        b = r.confidence_breakdown
        print(f"\n  {field}")
        print(f"    Value           : {r.value}")
        preview = r.anchor.text[:60].replace("\n", " ")
        print(f"    Anchor [{r.anchor.method}]: «{preview}{'…' if len(r.anchor.text) > 60 else ''}»")
        if b.attention_score is not None:
            print(f"    Attention conc. : {b.attention_score:.2%}")
        print(f"    Localisation    : {b.localization_score:.2%}  "
              f"({r.localization.n_success}/{r.localization.n_trials} ok)")
        print(f"    Anchor valid.   : {b.anchor_validation_score:.2%}  "
              f"({r.anchor_validation.n_failures}/{r.anchor_validation.n_trials} broke)")
        if b.entropy_score is not None:
            print(f"    Entropy score   : {b.entropy_score:.2%}")
        if b.mlm_perplexity is not None:
            print(f"    MLM perplexity  : {b.mlm_perplexity:.2f}")
        if b.confidence_interval is not None:
            lo, hi = b.confidence_interval
            print(f"    95% CI          : [{lo:.2%}, {hi:.2%}]")
        print(f"    CONFIDENCE      : {r.confidence:.2%}  →  {b.interpretation}")
    print(f"  {'-'*70}")


def print_comparison(all_results: dict[str, dict]) -> None:
    print(f"\n{'═' * 72}")
    print("  VERSION COMPARISON")
    print(f"{'═' * 72}")
    print(f"  {'FIELD':<22} {'V1 BASIC':>10} {'V2 INTERM':>10} {'V3 ADV':>10}")
    print(f"  {'-'*55}")
    for field in FIELDS:
        row = f"  {field:<22}"
        for results in all_results.values():
            if field in results:
                row += f" {results[field].confidence:>10.1%}"
            else:
                row += f" {'N/A':>10}"
        print(row)
    print(f"{'═' * 72}")


def save_json(results: dict, path: str = "confidence_results.json") -> None:
    out = {}
    for field, r in results.items():
        b = r.confidence_breakdown
        entry = {
            "value": r.value,
            "confidence": round(r.confidence, 4),
            "localization_score": round(b.localization_score, 4),
            "anchor_validation_score": round(b.anchor_validation_score, 4),
            "anchor_text": r.anchor.text,
            "anchor_method": r.anchor.method,
            "interpretation": b.interpretation,
        }
        if b.attention_score is not None:
            entry["attention_score"] = round(b.attention_score, 4)
        if b.entropy_score is not None:
            entry["entropy_score"] = round(b.entropy_score, 4)
        if b.mlm_perplexity is not None:
            entry["mlm_perplexity"] = round(b.mlm_perplexity, 2)
        if b.confidence_interval is not None:
            entry["confidence_interval"] = [round(b.confidence_interval[0], 4),
                                             round(b.confidence_interval[1], 4)]
        out[field] = entry
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Invoice confidence extraction demo")
    parser.add_argument("--version", choices=["v1", "v2", "v3", "all"], default="all")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--no-mlm", action="store_true", help="Disable MLM scorer in v3")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()

    load_4bit = None if not args.no_4bit else False
    all_results: dict[str, dict] = {}
    versions_to_run = ["v1", "v2", "v3"] if args.version == "all" else [args.version]

    if "v1" in versions_to_run:
        from v1_basic.extractor import BasicConfidenceExtractor
        e = BasicConfidenceExtractor(
            model_name=args.model, load_in_4bit=load_4bit, n_trials=args.trials
        )
        results = e.extract_fields(INVOICE_TEXT, FIELDS)
        print_report("v1_basic (heuristic)", results)
        all_results["v1"] = results

    if "v2" in versions_to_run:
        from v2_intermediate.extractor import IntermediateConfidenceExtractor
        e = IntermediateConfidenceExtractor(
            model_name=args.model, load_in_4bit=load_4bit, n_trials=args.trials
        )
        results = e.extract_fields(INVOICE_TEXT, FIELDS)
        print_report("v2_intermediate (gradient)", results)
        all_results["v2"] = results

    if "v3" in versions_to_run:
        from v3_advanced.extractor import AdvancedConfidenceExtractor
        mlm_name = None if args.no_mlm else "distilbert-base-uncased"
        e = AdvancedConfidenceExtractor(
            model_name=args.model, mlm_model_name=mlm_name,
            load_in_4bit=load_4bit, n_trials=args.trials,
        )
        results = e.extract_fields(INVOICE_TEXT, FIELDS)
        print_report("v3_advanced (entropy + MLM)", results)
        all_results["v3"] = results

    if len(all_results) > 1:
        print_comparison(all_results)

    # Save last-run version's results
    last_results = next(reversed(all_results.values()))
    save_json(last_results)


if __name__ == "__main__":
    main()
