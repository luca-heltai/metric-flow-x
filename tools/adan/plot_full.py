#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plot completed ADAVN probe histories, timesteps and MPI assembly results.

Requires matplotlib. Figures are standalone PNG and vector PDF artifacts.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_full import MMHG_PA


def save(fig, output, name):
    output.mkdir(parents=True, exist_ok=True)
    fig.text(.5, -.015, "Network: Müller et al. (2023), doi:10.3389/fphys.2023.1162391, CC BY 4.0; modified model",
             ha="center", fontsize=7)
    for suffix in ('png', 'pdf'):
        fig.savefig(output/(name+'.'+suffix), dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_run(run, output):
    report = json.loads((run/'cycle_report.json').read_text())
    if report['final_time_s'] != 8.:
        raise ValueError('A validated ten-cycle report is required')
    with (run/'probes.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    series = {}
    for row in rows:
        series.setdefault(int(row['vessel_id']), []).append(row)
    time = [float(row['time_s']) for row in series[1630]]
    def average(ids, key, scale=1.):
        return [sum(float(series[vid][k][key]) for vid in ids)/len(ids)*scale
                for k in range(len(time))]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, layout='constrained')
    axes[0].plot(time, average((1630, 1631), 'pressure_Pa', 1/MMHG_PA), label='Aortic probes, source row 815')
    axes[0].axhline(92.58, color='tab:red', linestyle='--', label='Paper Table 3: MAP 92.58 mmHg')
    axes[0].set_ylabel('Pressure [mmHg]')
    axes[0].legend(fontsize=8)
    axes[1].plot(time, average((1630, 1631), 'flow_m3_s', 1e6), label='Aortic probes')
    for ids, label in (((8298, 8299), 'IVC'), ((8574, 8575), 'SVC'), ((8696, 8697), 'Coronary sinus')):
        axes[1].plot(time, average(ids, 'flow_m3_s', 1e6), label=label)
    axes[1].set(ylabel='Flow [mL/s]', xlabel='Time [s]')
    axes[1].legend(fontsize=8, ncols=2)
    for ax in axes:
        ax.axvspan(0, 4.8, color='0.8', alpha=.3)
        ax.axvline(4.8, color='0.4', linestyle=':')
        ax.grid(alpha=.2)
    fig.suptitle('Full ADAVN open-loop elastic reduction; shaded: tolerance ramp')
    save(fig, output, 'pressure_flow')
    with (run/'ida_internal_steps.csv').open() as stream:
        steps = list(csv.DictReader(stream))
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True, layout='constrained')
    time = [float(row['t']) for row in steps]
    axes[0].semilogy(time, [float(row['h']) for row in steps], linewidth=.8)
    axes[0].set_ylabel('Accepted IDA step [s]')
    axes[1].plot(time, [int(row['nonlin_iters']) for row in steps], linewidth=.8)
    axes[1].set(ylabel='Newton iterations / step', xlabel='Time [s]')
    for ax in axes:
        ax.axvspan(0, 4.8, color='0.8', alpha=.3)
        ax.axvline(2.4, color='0.5', linestyle=':')
        ax.axvline(4.8, color='0.5', linestyle=':')
        ax.grid(alpha=.2)
    save(fig, output, 'startup_steps')


def plot_scaling(performance, output):
    result = json.loads((performance/'results.json').read_text())
    ranks = sorted(map(int, result['summaries']))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), layout='constrained')
    for key, name in (('residual', 'Residual'), ('jacobian', 'Jacobian')):
        axes[0].plot(ranks, [result['summaries'][str(n)][key+'_median_s']*1000 for n in ranks], 'o-', label=name)
        axes[1].plot(ranks, [result['summaries'][str(n)][key+'_speedup'] for n in ranks], 'o-', label=name)
    axes[1].plot(ranks, [n/min(ranks) for n in ranks], '--', color='0.5', label='Ideal')
    axes[0].set(ylabel='Maximum-rank assembly time [ms]')
    axes[1].set(ylabel='Strong-scaling speedup')
    for ax in axes:
        ax.set(xlabel='MPI ranks', xticks=ranks)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('CPU/MPI assembly on GPU partition; three independent runs')
    save(fig, output, 'assembly_scaling')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--performance-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir is None and args.performance_dir is None:
        parser.error('Provide a run or performance directory')
    if args.run_dir:
        plot_run(args.run_dir, args.output_dir)
    if args.performance_dir:
        plot_scaling(args.performance_dir, args.output_dir)
