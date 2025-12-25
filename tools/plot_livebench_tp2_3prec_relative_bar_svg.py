#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def _fmt_pct(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}%"


def _fmt_delta_pp(x: float, digits: int = 2) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.{digits}f}pp"


def _escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    summary_path = repo_root / "artifacts/qwen3_32b_vllm_tp2_livebench_20251225/results/livebench_summary.json"
    out_path = repo_root / "images/qwen3_32b_tp2_livebench_relative_bar_20251225.svg"

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    models = data["models"]

    bf16 = models["qwen3-32b-bf16"]["mean"]
    fp8 = models["qwen3-32b-fp8"]["mean"]
    nvfp4 = models["qwen3-32b-nvfp4"]["mean"]

    series = [
        ("BF16", bf16, "#1f77b4"),
        ("FP8", fp8, "#ff7f0e"),
        ("NVFP4", nvfp4, "#2ca02c"),
    ]

    rel = [(name, (mean / bf16) * 100.0, color) for name, mean, color in series]
    deltas = [(name, r - 100.0) for name, r, _ in rel]

    # SVG layout
    width, height = 860, 420
    margin_l, margin_r, margin_t, margin_b = 90, 40, 70, 90
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b

    # y scale: show percentage axis in 0-100 ticks (as requested).
    # Keep a little headroom above 100 so values like 100.26% are not clipped.
    r_values = [r for _, r, _ in rel]
    y_min = 0.0
    y_max = max(105.0, max(r_values) + 1.0)

    def y_of(v: float) -> float:
        # higher v => smaller y (SVG origin at top)
        return margin_t + (y_max - v) / (y_max - y_min) * chart_h

    # x positions
    n = len(rel)
    gap = 28
    bar_w = (chart_w - gap * (n - 1)) / n

    def x_of(i: int) -> float:
        return margin_l + i * (bar_w + gap)

    # axis ticks (0-100)
    ticks = [0, 20, 40, 60, 80, 100]

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')

    # Title
    title = "Qwen3-32B TP=2 LiveBench（BF16=100% 相对分数）"
    subtitle = "release=2024-11-25 · n=1000/model · 2025-12-25"
    lines.append(f'<text x="{width/2}" y="32" text-anchor="middle" font-size="20" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Arial">{_escape_xml(title)}</text>')
    lines.append(f'<text x="{width/2}" y="54" text-anchor="middle" font-size="13" fill="#555" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Arial">{_escape_xml(subtitle)}</text>')

    # Grid + y labels
    for t in ticks:
        y = y_of(t)
        lines.append(f'<line x1="{margin_l}" y1="{y}" x2="{width-margin_r}" y2="{y}" stroke="#e6e6e6" stroke-width="1"/>')
        lines.append(f'<text x="{margin_l-10}" y="{y+4}" text-anchor="end" font-size="12" fill="#555" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Arial">{_escape_xml(_fmt_pct(float(t),0))}</text>')

    # Axes
    lines.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t+chart_h}" stroke="#333" stroke-width="1"/>')
    lines.append(f'<line x1="{margin_l}" y1="{margin_t+chart_h}" x2="{width-margin_r}" y2="{margin_t+chart_h}" stroke="#333" stroke-width="1"/>')

    # Bars
    for i, (name, r, color) in enumerate(rel):
        x = x_of(i)
        y = y_of(r)
        y0 = y_of(y_min)
        h = max(0.0, y0 - y)
        lines.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{color}"/>')

        # value label
        lines.append(
            f'<text x="{x + bar_w/2}" y="{y - 8}" text-anchor="middle" font-size="13" font-weight="600" fill="#111" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Arial">{_escape_xml(_fmt_pct(r,2))}</text>'
        )

        # x label + delta
        delta = dict(deltas)[name]
        lines.append(
            f'<text x="{x + bar_w/2}" y="{margin_t + chart_h + 28}" text-anchor="middle" font-size="14" fill="#111" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Arial">{_escape_xml(name)}</text>'
        )
        lines.append(
            f'<text x="{x + bar_w/2}" y="{margin_t + chart_h + 48}" text-anchor="middle" font-size="12" fill="#555" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Arial">{_escape_xml(_fmt_delta_pp(delta,2))} vs BF16</text>'
        )

    # y-axis label
    lines.append(
        f'<text x="{margin_l}" y="{margin_t-18}" text-anchor="start" font-size="12" fill="#555" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Arial">Relative score (BF16=100%)</text>'
    )

    lines.append("</svg>")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
