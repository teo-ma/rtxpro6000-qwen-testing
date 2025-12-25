#!/usr/bin/env python3
"""Generate the Qwen3-14B PPL-proxy relative-score bar chart PNG.

Goal: keep palette consistent with LiveBench chart:
- BF16:   #1f77b4
- Ours:   #ff7f0e
- NVIDIA: #2ca02c
- Ours (no calib): #7f7f7f

Output (default): images/qwen3_14b_nvfp4_ppl_proxy_score.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class BarSpec:
    label: str
    value: float
    color: str


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    out_path = os.environ.get(
        "OUT_PNG",
        os.path.join(repo_root, "images", "qwen3_14b_nvfp4_ppl_proxy_score.png"),
    )

    bars = [
        BarSpec("BF16", 100.00, "#1f77b4"),
        BarSpec("自制量化 NVFP4（有校准）", 99.31, "#ff7f0e"),
        BarSpec("NVIDIA (NVFP4)", 98.49, "#2ca02c"),
        BarSpec("自制量化 NVFP4（无校准）", 98.46, "#7f7f7f"),
    ]

    # Import lazily so the repo stays dependency-light; we only need matplotlib to regenerate the PNG.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Font fallback for Chinese labels (varies by OS/image). DejaVu Sans may miss CJK glyphs.
    # 1) Try to add well-known CJK font files directly (works even if font isn't registered).
    # 2) Fall back to common font family names.
    from matplotlib import font_manager

    candidate_font_files = [
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        # Linux (common locations)
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        # Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]

    added_font_names = []
    for font_path in candidate_font_files:
        if os.path.exists(font_path):
            try:
                font_manager.fontManager.addfont(font_path)
                font_name = font_manager.FontProperties(fname=font_path).get_name()
                if font_name:
                    added_font_names.append(font_name)
            except Exception:
                # Best-effort only.
                pass

    fallback_font_names = [
        "PingFang SC",  # macOS
        "Hiragino Sans GB",  # macOS
        "Heiti SC",  # macOS
        "Arial Unicode MS",  # macOS (often present)
        "Noto Sans CJK SC",  # Linux
        "Noto Sans CJK",  # Linux
        "Source Han Sans SC",  # Linux
        "Microsoft YaHei",  # Windows
        "SimHei",  # Windows
        "DejaVu Sans",  # fallback
    ]

    # Put dynamically added font names first.
    plt.rcParams["font.sans-serif"] = added_font_names + fallback_font_names
    plt.rcParams.setdefault("axes.unicode_minus", False)

    labels = [b.label for b in bars]
    values = [b.value for b in bars]
    colors = [b.color for b in bars]

    fig_w, fig_h = 6.4, 3.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    x = list(range(len(bars)))
    rects = ax.bar(x, values, color=colors, width=0.6)

    ax.set_title("Relative Score (BF16=100, PPL proxy)")
    ax.set_ylabel("Score")
    ax.set_xticks(x, labels, rotation=0)

    # Show ticks 0-100 like LiveBench; keep slight headroom to avoid clipping the 100.00 label.
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    # Value labels on top.
    for rect, v in zip(rects, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.3,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Keep visuals clean and similar to other charts.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"WROTE_PNG {out_path}")


if __name__ == "__main__":
    main()
