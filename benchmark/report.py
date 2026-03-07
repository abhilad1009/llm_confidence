"""
benchmark/report.py — Load benchmark results and generate tables + figures.

Usage:
  python benchmark/report.py                               # uses default output file
  python benchmark/report.py --input results.json
  python benchmark/report.py --plot                        # also generate figures
  python benchmark/report.py --show-attributions           # print attribution heatmaps
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.metrics import compute_all_metrics, MetricSummary
from benchmark.schemas import MethodResult


def load_results(path: str) -> list[MethodResult]:
    with open(path) as f:
        records = json.load(f)
    results = []
    for r in records:
        # token_attributions may be None or list of dicts
        results.append(MethodResult(**r))
    return results


def group_results(
    results: list[MethodResult],
) -> dict[tuple[str, str], list[MethodResult]]:
    """Group by (method, dataset)."""
    groups: dict[tuple[str, str], list[MethodResult]] = defaultdict(list)
    for r in results:
        # dataset is stored in example_id prefix (e.g. 'squad_xxxx')
        dataset = r.example_id.split("_")[0]
        groups[(r.method, dataset)].append(r)
    return groups


def print_comparison_table(summaries: list[MetricSummary]) -> None:
    """Print a formatted comparison table: methods × datasets, AUROC + ECE."""
    datasets = sorted(set(s.dataset for s in summaries))
    methods = sorted(set(s.method for s in summaries))

    # Index by (method, dataset)
    idx: dict[tuple[str, str], MetricSummary] = {
        (s.method, s.dataset): s for s in summaries
    }

    # Header
    col_w = 12
    header = f"{'Method':<22}"
    for ds in datasets:
        header += f"  {ds.upper()[:col_w]:^{col_w*2+3}}"
    print("\n" + "=" * (22 + len(datasets) * (col_w * 2 + 5)))
    print("MAIN RESULTS — AUROC / ECE")
    print("=" * (22 + len(datasets) * (col_w * 2 + 5)))
    sub_header = f"{'':22}"
    for _ in datasets:
        sub_header += f"  {'AUROC':>{col_w}} {'ECE':>{col_w}}  "
    print(header)
    print(sub_header)
    print("-" * (22 + len(datasets) * (col_w * 2 + 5)))

    for method in methods:
        row = f"{method:<22}"
        for ds in datasets:
            key = (method, ds)
            if key in idx:
                s = idx[key]
                row += f"  {s.auroc:>{col_w}.3f} {s.ece:>{col_w}.3f}  "
            else:
                row += f"  {'—':>{col_w}} {'—':>{col_w}}  "
        print(row)

    print()

    # Brier score table
    print("BRIER SCORE (lower is better)")
    print("-" * (22 + len(datasets) * (col_w + 3)))
    brier_header = f"{'Method':<22}"
    for ds in datasets:
        brier_header += f"  {ds.upper()[:col_w]:>{col_w}}"
    print(brier_header)
    for method in methods:
        row = f"{method:<22}"
        for ds in datasets:
            key = (method, ds)
            if key in idx:
                row += f"  {idx[key].brier:>{col_w}.3f}"
            else:
                row += f"  {'—':>{col_w}}"
        print(row)

    print()

    # Accuracy@90% table
    print("ACCURACY @ 90% COVERAGE")
    print("-" * (22 + len(datasets) * (col_w + 3)))
    for method in methods:
        row = f"{method:<22}"
        for ds in datasets:
            key = (method, ds)
            if key in idx:
                row += f"  {idx[key].accuracy_at_90:>{col_w}.3f}"
            else:
                row += f"  {'—':>{col_w}}"
        print(row)
    print()


def plot_reliability_diagrams(
    summaries: list[MetricSummary],
    groups: dict[tuple[str, str], list[MethodResult]],
    save_dir: str = ".",
) -> None:
    """Plot reliability diagrams (confidence bins vs actual accuracy)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[report] matplotlib not installed — skipping plots.")
        return

    datasets = sorted(set(s.dataset for s in summaries))

    for ds in datasets:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")

        for method in sorted(set(s.method for s in summaries)):
            key = (method, ds)
            if key not in groups:
                continue
            results = groups[key]
            confidences = [r.confidence for r in results]
            is_correct = [r.is_correct for r in results]

            n_bins = 10
            bin_size = 1.0 / n_bins
            bin_confs = []
            bin_accs = []
            for b in range(n_bins):
                lo, hi = b * bin_size, (b + 1) * bin_size
                indices = [i for i, c in enumerate(confidences) if lo <= c < hi]
                if not indices:
                    continue
                bin_confs.append(sum(confidences[i] for i in indices) / len(indices))
                bin_accs.append(sum(is_correct[i] for i in indices) / len(indices))

            ax.plot(bin_confs, bin_accs, marker="o", label=method, lw=1.5)

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Reliability Diagram — {ds.upper()}")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        out_path = f"{save_dir}/reliability_{ds}.png"
        fig.savefig(out_path, dpi=150)
        print(f"[report] Saved reliability diagram: {out_path}")
        plt.close(fig)


def plot_risk_coverage(
    groups: dict[tuple[str, str], list[MethodResult]],
    save_dir: str = ".",
) -> None:
    """Plot risk-coverage curves per dataset."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    from benchmark.metrics import compute_risk_coverage

    datasets = sorted(set(ds for (_, ds) in groups))

    for ds in datasets:
        fig, ax = plt.subplots(figsize=(7, 5))
        for (method, d), results in sorted(groups.items()):
            if d != ds:
                continue
            confidences = [r.confidence for r in results]
            is_correct = [r.is_correct for r in results]
            rc = compute_risk_coverage(confidences, is_correct)
            ax.plot(rc["coverage"], rc["risk"], marker=".", label=f"{method} (AUC={rc['auc']:.3f})", lw=1.5)

        ax.set_xlabel("Coverage (fraction of examples kept)")
        ax.set_ylabel("Risk (error rate)")
        ax.set_title(f"Risk-Coverage Curve — {ds.upper()}")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        out_path = f"{save_dir}/risk_coverage_{ds}.png"
        fig.savefig(out_path, dpi=150)
        print(f"[report] Saved risk-coverage plot: {out_path}")
        plt.close(fig)


def show_attribution_examples(
    groups: dict[tuple[str, str], list[MethodResult]],
    n_examples: int = 5,
) -> None:
    """Print attribution heatmaps for v2/v3 examples."""
    from benchmark.visualize import render_attribution_heatmap
    from shared.schemas import TokenAttribution

    for (method, ds), results in sorted(groups.items()):
        if method not in ("v2", "v3", "attention_entropy"):
            continue
        shown = 0
        for r in results:
            if not r.token_attributions or shown >= n_examples:
                continue
            tas = [
                TokenAttribution(
                    token=ta["token"],
                    score=ta["score"],
                    char_start=ta["char_start"],
                    char_end=ta["char_end"],
                )
                for ta in r.token_attributions
            ]
            status = "CORRECT" if r.is_correct else "WRONG"
            title = (
                f"{method}/{ds} — {r.example_id[:40]} "
                f"conf={r.confidence:.3f} [{status}]"
            )
            print(render_attribution_heatmap(tas, title=title))
            shown += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--input", default="confidence_results_benchmark.json")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib figures")
    parser.add_argument("--save-dir", default=".", help="Directory to save figures")
    parser.add_argument(
        "--show-attributions", action="store_true",
        help="Print token attribution heatmaps for v2/v3 examples"
    )
    parser.add_argument("--n-attribution-examples", type=int, default=3)
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"[report] Results file not found: {args.input}")
        print("[report] Run benchmark/runner.py first.")
        sys.exit(1)

    print(f"[report] Loading results from {args.input} …")
    results = load_results(args.input)
    print(f"[report] {len(results)} results loaded.")

    groups = group_results(results)

    # Compute metrics per (method, dataset)
    summaries: list[MetricSummary] = []
    for (method, ds), group_results_list in groups.items():
        confidences = [r.confidence for r in group_results_list]
        is_correct = [r.is_correct for r in group_results_list]
        summary = compute_all_metrics(method, ds, confidences, is_correct)
        summaries.append(summary)

    print_comparison_table(summaries)

    if args.plot:
        plot_reliability_diagrams(summaries, groups, save_dir=args.save_dir)
        plot_risk_coverage(groups, save_dir=args.save_dir)

    if args.show_attributions:
        show_attribution_examples(groups, n_examples=args.n_attribution_examples)


if __name__ == "__main__":
    main()
