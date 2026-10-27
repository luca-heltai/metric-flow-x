#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plot frontal and sagittal views of an ADAVN VTK before starting a run.

Requires matplotlib. Atlas-derived figures retain the atlas's separate terms.
"""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_vtk_network import parse_vtk


def preview(vtk, output):
    mesh = parse_vtk(vtk)
    types = mesh.cell_arrays['source_vessel_type']
    fig, axes = plt.subplots(1, 2, figsize=(9, 9), sharey=True, layout='constrained')
    for ax, horizontal, title in zip(axes, (0, 1), ('Frontal (x,z)', 'Sagittal (y,z)')):
        for kind, color, label in ((1, '#bd2525', 'Arteries'), (0, '#185bba', 'Veins')):
            segments = [[(mesh.points[i][horizontal], mesh.points[i][2]) for i in cell]
                        for cell, vessel_type in zip(mesh.cells, types) if vessel_type == kind]
            ax.add_collection(LineCollection(segments, colors=color, linewidths=.45,
                                             alpha=.8, label=label))
        ax.autoscale()
        ax.set_aspect('equal')
        ax.set(title=title, xlabel=('x' if horizontal == 0 else 'y')+' [m]')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('z [m]')
    axes[0].legend(fontsize=9)
    fig.suptitle('ADAVN arteries and veins: geometry inspection')
    fig.text(.5, -.01, 'Atlas geometry: HeMoLab ADAVN-WEB, separate terms; adapted to the 2023 supplementary table.',
             ha='center', fontsize=7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vtk', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    preview(args.vtk, args.output)
