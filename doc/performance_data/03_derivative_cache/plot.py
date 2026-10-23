#!/usr/bin/env python3
"""Plot archived process medians without rerunning the solver."""
from pathlib import Path
import json
import statistics
ROOT = Path(__file__).resolve().parent
records = [json.loads(line) for line in (ROOT / "timings.jsonl").read_text().splitlines()]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
metrics = ["residual", "jacobian", "derivative"]
cases = sorted({r["case"] for r in records})
fig, axes = plt.subplots(1, 3, figsize=(15, max(3, len(cases) * .55 + 1)))
summary = []
for ax, metric in zip(axes, metrics):
    for offset, variant, color in [(-.18, "before", "#666666"), (.18, "after", "#0072B2")]:
        values = [[1000 * r[f"{metric}_median_s"] for r in records
                   if r["case"] == case and r["variant"] == variant] for case in cases]
        centers = [statistics.median(v) for v in values]
        errors = [[c - min(v) for c, v in zip(centers, values)],
                  [max(v) - c for c, v in zip(centers, values)]]
        ax.barh([i + offset for i in range(len(cases))], centers, height=.34,
                xerr=errors, label=variant, color=color, capsize=3)
        summary.extend({"case": case, "metric": metric, "variant": variant,
                        "median_ms": c, "min_ms": min(v), "max_ms": max(v)}
                       for case, c, v in zip(cases, centers, values))
    ax.set_yticks(range(len(cases)), cases)
    ax.set_xlabel("ms / assembly; median of process medians (range)")
    ax.set_title(metric + " assembly")
    ax.legend()
for i in range(len(cases)):
    axes[2].annotate("<0.001 ms (cached)", xy=(0, i + .18), xytext=(5, 0),
                     textcoords="offset points", va="center", fontsize=8,
                     color="#0072B2")
fig.suptitle("Repeated calls after warm-up; lnx2 (valid derivative cache)")
fig.tight_layout(rect=(0, 0, 1, .93))
fig.savefig(ROOT / "comparison.svg")
svg = ROOT / "comparison.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
fig.savefig(ROOT / "comparison.png", dpi=150)
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
