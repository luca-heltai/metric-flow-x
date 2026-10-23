"""Regenerate cumulative latency and strong-scaling plots from archived timings."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
records = [json.loads(line) for line in (args.data / 'timings.jsonl').read_text().splitlines()]
assert len(records) == 48
ranks = [1, 2, 4, 8]
metrics = ['residual', 'jacobian']
workloads = ['adan56', 'network10k']
colors = {'before': '#666666', 'after': '#0072B2'}
summary = []
lookup = {}
for workload in workloads:
    for rank in ranks:
        case = f'{workload}_mpi{rank}'
        pairs = {run: {r['variant']: r for r in records
                       if r['case'] == case and r['run'] == run} for run in range(3)}
        assert all(set(pair) == {'before', 'after'} for pair in pairs.values())
        for metric in metrics:
            row = dict(case=case, workload=workload, ranks=rank, metric=metric)
            for variant in colors:
                values = [1000 * pair[variant][metric + '_median_s'] for pair in pairs.values()]
                row[variant + '_median_ms'] = statistics.median(values)
                row[variant + '_min_ms'] = min(values)
                row[variant + '_max_ms'] = max(values)
            row['median_paired_speedup'] = statistics.median(
                pair['before'][metric + '_median_s'] / pair['after'][metric + '_median_s']
                for pair in pairs.values())
            summary.append(row)
            lookup[workload, rank, metric] = row

for row in summary:
    for variant in colors:
        single = lookup[row['workload'], 1, row['metric']][variant + '_median_ms']
        speedup = single / row[variant + '_median_ms']
        row[variant + '_strong_scaling_speedup'] = speedup
        row[variant + '_parallel_efficiency'] = speedup / row['ranks']

for kind in ['latency', 'strong_scaling']:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, workload in enumerate(workloads):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            selected = [lookup[workload, rank, metric] for rank in ranks]
            for variant, color in colors.items():
                if kind == 'latency':
                    values = [r[variant + '_median_ms'] for r in selected]
                    errors = [[r[variant + '_median_ms'] - r[variant + '_min_ms'] for r in selected],
                              [r[variant + '_max_ms'] - r[variant + '_median_ms'] for r in selected]]
                    ax.errorbar(ranks, values, yerr=errors, marker='o', capsize=3,
                                color=color, label=variant)
                else:
                    values = [r[variant + '_strong_scaling_speedup'] for r in selected]
                    ax.plot(ranks, values, marker='o', color=color, label=variant)
            if kind == 'strong_scaling':
                ax.plot(ranks, ranks, '--', color='#999999', label='ideal')
            ax.set_xscale('log', base=2)
            ax.set_xticks(ranks, ranks)
            ax.set_xlabel('MPI ranks, one node')
            ax.set_ylabel('ms / assembly' if kind == 'latency' else 'T(1) / T(p) within each variant')
            ax.set_title(('ADAN56' if workload == 'adan56' else '10k SI network') + ' — ' + metric)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
            ax.legend()
    figure.suptitle('Initial baseline vs five-change stack — lnx3, three alternating process pairs')
    figure.tight_layout()
    for extension in ['svg', 'png']:
        target = args.data / (kind + '.' + extension)
        figure.savefig(target, dpi=150)
        if extension == 'svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
    plt.close(figure)
(args.data / 'scaling_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
