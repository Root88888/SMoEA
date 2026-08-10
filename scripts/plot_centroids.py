#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/plot_centroids.py

【多質心結構圖】讀 assets/centroids_summary.json（build_router_assets
產出），把每個任務的分群結果畫成「標籤—扇形—質心點」圖：單質心任務
一條垂線一個點；多質心任務從標籤下方扇形展開 k 個點。

【執行】
  python scripts/plot_centroids.py \
      --summary assets/centroids_summary.json --out results/centroids_fan.png
【scale up】任務數與每行任務數自動適應；任務增多自動增行。
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORANGE = "#F5A623"
GREY = "#AAAAAA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="assets/centroids_summary.json")
    ap.add_argument("--out", default="results/centroids_fan.png")
    ap.add_argument("--per_row", type=int, default=13)
    ap.add_argument("--dot_gap", type=float, default=0.30)
    a = ap.parse_args()
    with open(a.summary, encoding="utf-8") as f:
        S = json.load(f)
    tasks = sorted(S, key=lambda x: int(x[1:]))
    rows = [tasks[i:i + a.per_row] for i in range(0, len(tasks), a.per_row)]

    row_h, pad = 2.0, 0.7

    def slotw(t):
        return max(S[t]["k"] * a.dot_gap, 0.8) + 0.35

    W = max(sum(slotw(t) for t in r) for r in rows) + 2 * pad
    H = len(rows) * row_h + pad
    fig, ax = plt.subplots(figsize=(W * 0.55, H * 0.62))
    for ri, row in enumerate(rows):
        y0 = (len(rows) - 1 - ri) * row_h
        y_lab, y_anchor, y_dot = y0 + 1.45, y0 + 1.28, y0 + 0.55
        x = pad
        for t in row:
            k = S[t]["k"]
            w = slotw(t)
            cx = x + w / 2
            ax.text(cx, y_lab, t, ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color="#1a1a1a")
            xs = ([cx] if k == 1 else
                  [cx + (i - (k - 1) / 2) * a.dot_gap for i in range(k)])
            for xd in xs:
                ax.plot([cx, xd], [y_anchor, y_dot + 0.13],
                        color=GREY, lw=0.9, zorder=1)
                ax.plot(xd, y_dot, "o", color=ORANGE, markersize=9,
                        zorder=2)
            x += w
    ax.set_xlim(0, W)
    ax.set_ylim(0, len(rows) * row_h + 0.4)
    ax.axis("off")
    n_multi = sum(1 for t in tasks if S[t]["multi"])
    ax.set_title(f"multi-centroid structure — {len(tasks)} tasks, "
                 f"{n_multi} expanded", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {a.out}  ({len(tasks)} tasks, {n_multi} multi)")


if __name__ == "__main__":
    main()
