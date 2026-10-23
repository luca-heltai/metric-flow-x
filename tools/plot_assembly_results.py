#!/usr/bin/env python3
"""Plot compact results.json from compare_assembly.py, without raw logs."""
import argparse
import json
from pathlib import Path


def plot_results(results, output, metrics, title=None):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "metric-flow-x"})
    import matplotlib.pyplot as plt

    rows = [r for r in results["rows"] if r["metric"] in metrics]
    workloads = sorted({r["workload"] for r in rows})
    if not workloads or not metrics:
        raise ValueError("No matching timing rows")
    figure, axes = plt.subplots(len(workloads), len(metrics), squeeze=False,
                                figsize=(6 * len(metrics), 4 * len(workloads)))
    for i, workload in enumerate(workloads):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            selected = sorted([r for r in rows if r["workload"] == workload and r["metric"] == metric],
                              key=lambda r: r["ranks"])
            ranks = [r["ranks"] for r in selected]
            if len(ranks) != len(set(ranks)) or any(p <= 0 for p in ranks):
                raise ValueError(f"Invalid or duplicate rank counts for {workload}/{metric}")
            for variant, color in [("before", "#666666"), ("after", "#0072B2")]:
                values = [r[variant + "_median_ms"] for r in selected]
                errors = [[r[variant + "_median_ms"] - r[variant + "_min_ms"] for r in selected],
                          [r[variant + "_max_ms"] - r[variant + "_median_ms"] for r in selected]]
                if selected:
                    ax.errorbar(ranks, values, yerr=errors, marker="o", capsize=3,
                                color=color, label=variant)
            if len(ranks) > 1:
                ax.set_xscale("log", base=2)
            ax.set_xticks(ranks, ranks)
            ax.set_xlabel("MPI ranks, one node")
            ax.set_ylabel("ms / assembly; median (range)")
            ax.set_title(workload + " — " + metric)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
            if selected:
                ax.legend()
    figure.suptitle(title or results.get("metadata", {}).get("title", "Assembly baseline versus candidate"))
    figure.tight_layout()
    figure.savefig(output, metadata={"Date": None} if output.suffix.lower() == ".svg" else None)
    plt.close(figure)
    if output.suffix.lower() == ".svg":
        output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metrics", nargs="+", default=["residual", "jacobian"])
    parser.add_argument("--title")
    args = parser.parse_args()
    output = args.output or args.results.with_suffix(".svg")
    if output.resolve() == args.results.resolve():
        parser.error("Figure output must differ from the input JSON")
    plot_results(json.loads(args.results.read_text()), output, args.metrics, args.title)


if __name__ == "__main__":
    main()
