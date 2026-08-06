"""Generate 4 realistic-looking images for examples/input_real.json."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "images"
OUT.mkdir(parents=True, exist_ok=True)


def arch_diagram():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, label, color="#E8F1FB", edge="#3366CC"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05",
            linewidth=1.5, edgecolor=edge, facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)

    # input row
    box(0.5, 0.4, 9.0, 0.7, "Input Tokens  (32k context)", color="#F4F4F4", edge="#666666")

    # attention layers
    for i in range(4):
        box(0.5 + i * 2.4, 1.6, 2.0, 0.9, f"Attn Layer L{i+1}\nentropy h(L_i)")

    # arrows up
    for i in range(4):
        ax.annotate("", xy=(0.5 + i * 2.4 + 1.0, 1.55),
                    xytext=(0.5 + i * 2.4 + 1.0, 1.15),
                    arrowprops=dict(arrowstyle="->", color="#666"))

    # threshold layer
    box(1.5, 3.1, 7.0, 0.9, "Dynamic Threshold τ(h)  →  selective KV drop",
        color="#FFF4E0", edge="#FF8C00")
    for i in range(4):
        ax.annotate("", xy=(0.5 + i * 2.4 + 1.0, 3.05),
                    xytext=(0.5 + i * 2.4 + 1.0, 2.55),
                    arrowprops=dict(arrowstyle="->", color="#666"))

    # output
    box(2.0, 4.5, 6.0, 0.9, "Compressed KV Cache  (-21% memory)",
        color="#E0F7E9", edge="#2E7D5B")
    ax.annotate("", xy=(5.0, 4.45), xytext=(5.0, 4.05),
                arrowprops=dict(arrowstyle="->", color="#666"))

    ax.set_title("Adaptive KV Cache Selection via Attention Entropy",
                 fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "arch.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def throughput_curve():
    ctx = np.array([4, 8, 16, 24, 32])
    full_tp = np.array([220, 165, 95, 58, 35])
    ours_tp = np.array([235, 195, 130, 82, 49])
    h2o_tp = np.array([225, 180, 115, 70, 42])

    full_acc = np.array([41.7, 41.5, 41.0, 40.6, 39.8])
    ours_acc = np.array([41.1, 41.0, 40.8, 40.4, 39.5])
    h2o_acc = np.array([40.1, 39.8, 39.0, 38.0, 36.7])

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(ctx, full_tp, "o-", color="#666666", label="Full Cache (TP)", linewidth=2)
    ax1.plot(ctx, ours_tp, "o-", color="#3366CC", label="Ours (TP)", linewidth=2.5)
    ax1.plot(ctx, h2o_tp, "o-", color="#A02020", label="H2O (TP)", linewidth=2)
    ax1.set_xlabel("Context length (k tokens)", fontsize=11)
    ax1.set_ylabel("Throughput (tok/s)", fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(ctx, full_acc, "s--", color="#666666", alpha=0.6, label="Full (F1)")
    ax2.plot(ctx, ours_acc, "s--", color="#3366CC", alpha=0.7, label="Ours (F1)")
    ax2.plot(ctx, h2o_acc, "s--", color="#A02020", alpha=0.6, label="H2O (F1)")
    ax2.set_ylabel("LongBench F1", fontsize=11)
    ax2.set_ylim(34, 43)
    ax2.legend(loc="lower left", fontsize=9)

    ax1.set_title("Throughput & Accuracy vs Context Length", fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def heatmap_kv():
    rng = np.random.default_rng(42)
    layers, heads = 12, 32
    base = rng.beta(2, 2, size=(layers, heads))
    # add structure: lower layers keep more, some heads always keep
    bias = np.linspace(0.3, -0.2, layers)[:, None]
    base = np.clip(base + bias, 0, 1)
    base[:, 0] = np.clip(base[:, 0] + 0.5, 0, 1)
    base[:, 17] = np.clip(base[:, 17] + 0.4, 0, 1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(base, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("Attention head", fontsize=11)
    ax.set_ylabel("Layer (bottom → top)", fontsize=11)
    ax.set_title("KV Retention Heatmap (sample: NeedleInHaystack 32k)",
                 fontsize=12, pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("retention ratio", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def compare_quadrant():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("← prompt-aware           decode-aware →", fontsize=11)
    ax.set_ylabel("← layer-uniform           layer-adaptive →", fontsize=11)
    ax.axhline(5, color="#888", linewidth=1)
    ax.axvline(5, color="#888", linewidth=1)
    ax.grid(alpha=0.2)

    methods = [
        ("StreamingLLM", 1.5, 1.5, "#A02020"),
        ("H2O", 3.0, 3.0, "#A02020"),
        ("SnapKV", 1.8, 6.5, "#FF8C00"),
        ("PyramidKV", 2.2, 7.8, "#FF8C00"),
        ("Ours", 7.5, 8.2, "#3366CC"),
    ]
    for name, x, y, c in methods:
        ax.scatter(x, y, s=220, color=c, edgecolor="black", linewidth=1.0, zorder=3)
        ax.text(x + 0.25, y + 0.1, name, fontsize=11, fontweight="bold")

    ax.set_title("Design Space of KV Compression Methods", fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "compare.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    arch_diagram()
    throughput_curve()
    heatmap_kv()
    compare_quadrant()
    print("wrote 4 images to", OUT)
