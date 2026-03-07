"""
benchmark/visualize.py — Token attribution heatmap rendering.

Given a list of TokenAttribution objects from AnchorResult.token_attributions,
renders two views:
  1. ANSI text heatmap: inline colored tokens in the source text
  2. Matplotlib bar chart: score vs token index, annotated with token text
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.schemas import TokenAttribution


# ──────────────────────────────────────────────
# ANSI colour helpers
# ──────────────────────────────────────────────

_ANSI_RESET = "\033[0m"
_SUPPORTS_ANSI = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _ansi_bg(r: int, g: int, b: int, text: str) -> str:
    return f"\033[48;2;{r};{g};{b}m{text}{_ANSI_RESET}"


def _score_to_rgb(score: float) -> tuple[int, int, int]:
    """Map score [0, 1] to a colour: low=dark blue, high=bright red."""
    score = max(0.0, min(1.0, score))
    if score < 0.5:
        # blue → green
        t = score * 2
        return (0, int(180 * t), int(180 * (1 - t)))
    else:
        # green → red
        t = (score - 0.5) * 2
        return (int(220 * t), int(180 * (1 - t)), 0)


def render_attribution_heatmap(
    token_attributions: list[TokenAttribution],
    *,
    use_ansi: bool | None = None,
    title: str = "Token Attribution Heatmap",
) -> str:
    """
    Return a string displaying token attributions as a coloured heatmap.

    If ANSI is available (or forced via `use_ansi=True`), tokens are
    printed with background colours proportional to their score.
    Otherwise a plain ASCII table is returned.

    Args:
        token_attributions: From AnchorResult.token_attributions.
        use_ansi: Force ANSI on/off. Auto-detected from terminal if None.
        title: Header line above the heatmap.

    Returns:
        Rendered string (print it directly).
    """
    if not token_attributions:
        return f"[{title}] No token attributions available."

    ansi = _SUPPORTS_ANSI if use_ansi is None else use_ansi

    lines = [f"\n{'─' * 60}", f"  {title}", f"{'─' * 60}"]

    if ansi:
        # Coloured inline display — wrap at ~80 chars
        row = ""
        row_len = 0
        for ta in token_attributions:
            r, g, b = _score_to_rgb(ta.score)
            coloured = _ansi_bg(r, g, b, ta.token)
            # Each token is rendered as-is; spaces between tokens
            display = ta.token
            if row_len + len(display) + 1 > 80:
                lines.append(row)
                row = ""
                row_len = 0
            if row:
                row += " "
                row_len += 1
            row += coloured
            row_len += len(display)
        if row:
            lines.append(row)

        # Legend
        lines.append("")
        lines.append(
            "  Score:  "
            + _ansi_bg(0, 0, 180, "  low  ")
            + " → "
            + _ansi_bg(0, 180, 0, "  mid  ")
            + " → "
            + _ansi_bg(220, 0, 0, "  high  ")
        )
    else:
        # Plain ASCII table
        lines.append(f"  {'Token':<20} {'Score':>6}  {'Bar'}")
        lines.append(f"  {'-'*20} {'-'*6}  {'-'*20}")
        for ta in token_attributions:
            bar = "█" * int(ta.score * 20)
            tok = ta.token[:20]
            lines.append(f"  {tok:<20} {ta.score:>6.3f}  {bar}")

    lines.append(f"{'─' * 60}\n")
    return "\n".join(lines)


def print_attribution_heatmap(
    token_attributions: list[TokenAttribution],
    **kwargs,
) -> None:
    """Print the attribution heatmap to stdout."""
    print(render_attribution_heatmap(token_attributions, **kwargs))


def plot_attribution_bar(
    token_attributions: list[TokenAttribution],
    title: str = "Token Attribution Scores",
    max_tokens: int = 60,
    save_path: str | None = None,
) -> None:
    """
    Plot a bar chart of attribution scores vs token index.

    Requires matplotlib. Tokens are shown as x-axis tick labels.
    If save_path is given, saves the figure instead of showing it.

    Args:
        token_attributions: From AnchorResult.token_attributions.
        title: Plot title.
        max_tokens: Truncate to first N tokens for readability.
        save_path: If set, save figure to this path instead of plt.show().
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot_attribution_bar] matplotlib not installed — skipping plot.")
        return

    if not token_attributions:
        print("[plot_attribution_bar] No token attributions to plot.")
        return

    attrs = token_attributions[:max_tokens]
    tokens = [ta.token for ta in attrs]
    scores = [ta.score for ta in attrs]

    # Colour bars by score
    colours = [
        tuple(c / 255 for c in _score_to_rgb(s)) for s in scores
    ]

    fig, ax = plt.subplots(figsize=(max(10, len(tokens) * 0.3), 4))
    bars = ax.bar(range(len(tokens)), scores, color=colours, edgecolor="none")
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=60, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Attribution Score")
    ax.set_title(title)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[plot_attribution_bar] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
