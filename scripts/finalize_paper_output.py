#!/usr/bin/env python3
"""Validate current-code runs and export machine-specific manuscript scaling tables.

Accept a reproduce_paper output root or a submit_paper_gpu campaign root.
Only the 50k/100k strong-scaling tables actually plotted by the manuscript are
reduced. Raw application/field outputs are preserved separately from timed runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile

from reproduce_paper import ROOT, sha256, timestamp

SECTIONS = ('assemble_jacobian', 'assemble_residual',
            'build_extended_sparsity_pattern', 'solve_with_jacobian')
NETWORKS = {'network50k': '50k', 'network100k': '100k'}
DIRECTORIES = {'network50k': '50k_vessels', 'network100k': '100k_vessels',
               'liver': '8k_vessels', 'cerebral': 'cerebral'}


def hardware_tex(hardware):
    fields = {row['field'].rstrip(':'): row['data']
              for row in (hardware.get('lscpu') or {}).get('lscpu', [])}
    parts = [f"compute node {hardware['hostname']}"]
    if fields.get('Model name'):
        parts.append(fields['Model name'])
    if fields.get('Socket(s)') and fields.get('Core(s) per socket'):
        parts.append(f"{fields['Socket(s)']} sockets with {fields['Core(s) per socket']} cores per socket")
    if hardware.get('memory_total_kib'):
        parts.append(f"{hardware['memory_total_kib'] / 1024**2:.1f} GiB RAM")
    if hardware.get('platform'):
        parts.append(hardware['platform'])
    escapes = {'\\': r'\textbackslash{}', '_': r'\_', '%': r'\%', '&': r'\&',
               '#': r'\#', '$': r'\$', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
               '^': r'\textasciicircum{}'}
    description = ''.join(escapes.get(char, char) for char in ', '.join(parts))
    return '\\newcommand{\\ScalingMachineDescription}{' + description + '}\n'


def read_table(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def positive(value, label, allow_zero=False):
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f'Invalid {label}: {value}')
    return number


def receipts(root):
    paths = set(root.glob('*/run.json'))
    for name in ('scaling', 'outputs'):
        paths.update((root / name).glob('*/run.json'))
    return [(p, json.loads(p.read_text())) for p in sorted(paths)]


def finalize(root, paper_root, machine, ranks, description=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', machine) or machine == 'raksha':
        raise ValueError('Choose a new machine profile; raksha is the historical reference')
    if not ranks or any(n < 1 for n in ranks) or len(set(ranks)) != len(ranks):
        raise ValueError('Expected unique positive MPI counts')
    destination = paper_root / 'scaling' / machine
    # Existing hardware-only profiles are allowed, but measurements are never overwritten.
    if destination.exists() and any(p.name != 'machine.tex' for p in destination.iterdir()):
        raise ValueError(f'Profile already contains results: {destination}; choose a new profile')
    machine_tex = destination / 'machine.tex'
    if description:
        machine_text = '\\newcommand{\\ScalingMachineDescription}{' + description + '}\n'
    elif machine_tex.is_file():
        machine_text = machine_tex.read_text()
        if 'AUTO_FROM_SCALING_HARDWARE' in machine_text:
            machine_text = None
    else:
        machine_text = None
    selected = []
    outputs = []
    for path, receipt in receipts(root):
        case = receipt['case']
        if case.get('suite') == 'scaling' and case.get('base') in NETWORKS:
            selected.append((path, receipt))
        elif case.get('base') in DIRECTORIES:
            outputs.append((path, receipt))
    if not selected:
        raise ValueError('No 50k/100k scaling receipts found')
    groups = {base: {} for base in NETWORKS}
    configurations = {base: set() for base in NETWORKS}
    identities, hosts = set(), set()
    sources, summaries = [], []
    for path, receipt in selected:
        case = receipt['case']
        label = str(path)
        if receipt['status'] != 'completed' or receipt.get('returncode') != 0:
            raise ValueError(f'Incomplete or failed scaling run: {label}')
        if receipt.get('final_time_override') is not None:
            raise ValueError(f'Smoke/overridden run cannot become paper scaling: {label}')
        if receipt.get('machine') != machine:
            raise ValueError(f'Machine profile mismatch in {label}')
        hardware = receipt.get('hardware') or {}
        if not hardware.get('hostname'):
            raise ValueError(f'Missing execution hostname in {label}; rerun with current runner')
        hosts.add(hardware['hostname'])
        revision, binary = receipt.get('executable_revision'), receipt.get('executable_sha256')
        if not re.fullmatch(r'[0-9a-f]{40}', revision or '') or not re.fullmatch(r'[0-9a-f]{64}', binary or ''):
            raise ValueError(f'Missing executable revision/SHA256 in {label}')
        build = receipt.get('build_provenance')
        if build and (build.get('executable_revision') != revision or build.get('executable_sha256') != binary):
            raise ValueError(f'Contradictory build provenance in {label}')
        identities.add((revision, binary, receipt['manifest_sha256']))
        parameters = path.parent / 'parameters.prm'
        if sha256(parameters) != receipt['parameters_sha256']:
            raise ValueError(f'Parameters changed since execution: {parameters}')
        settings = receipt['settings']
        if settings.get('Write solution output') != 'false' or settings.get('Print current statistics') != 'false':
            raise ValueError(f'Scaling must disable solution output and progress: {label}')
        parameter_text = parameters.read_text()
        for key in ('Write solution output', 'Print current statistics'):
            values = re.findall(rf'(?m)^\s*set {key}\s*=\s*([^\n#]+)', parameter_text)
            if [value.strip() for value in values] != ['false']:
                raise ValueError(f'{key} must be false in {parameters}')
        horizon = re.findall(r'(?m)^\s*set Final time\s*=\s*([^\n#]+)', parameter_text)
        if len(horizon) != 1 or float(horizon[0]) != 1.0:
            raise ValueError(f'Paper scaling requires final time 1 s: {parameters}')
        configurations[case['base']].add(re.sub(
            r'(?m)^(\s*set Output directory\s*=)[^\n]*', r'\1 <run-output>', parameter_text))
        for line in parameter_text.splitlines():
            if re.match(r'\s*set Output (terminal|network terminals|mass conservation|IDA step history|IDA internal steps|flow regime) CSV file\s*=', line):
                if line.split('=', 1)[1].split('#', 1)[0].strip():
                    raise ValueError(f'Per-step diagnostic enabled in {parameters}')
        for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                         'BLIS_NUM_THREADS', 'DEAL_II_NUM_THREADS'):
            if receipt.get('thread_environment', {}).get(variable) != '1':
                raise ValueError(f'{variable} must be explicitly 1: {label}')
        base, mpi = case['base'], receipt['mpi']
        if not isinstance(mpi, int) or mpi != case['mpi'] or mpi not in ranks or mpi in groups[base]:
            raise ValueError(f'Unexpected/duplicate MPI count {mpi}: {label}')
        if case['id'] != f'{base}-np{mpi}':
            raise ValueError(f'Noncanonical scaling case ID in {label}')
        results = path.parent / 'results'
        timings = results / 'scaling_timings.csv'
        rows = read_table(timings)
        reduced = {}
        for row in rows:
            if int(row['n_mpi_processes']) != mpi:
                raise ValueError(f'MPI count mismatch in {timings}')
            value = positive(row['wall_time'], timings, allow_zero=True)
            section = row['section']
            if section in reduced:
                raise ValueError(f'Duplicate timing section in {timings}: {section}')
            reduced[section] = value
        if not set(SECTIONS) <= reduced.keys():
            raise ValueError(f'Missing scaling sections in {timings}')
        statistics = results / 'ida_statistics.csv'
        stats = read_table(statistics)
        if len(stats) != 1:
            raise ValueError(f'Expected exactly one solve in {statistics}')
        positive(stats[0]['n_steps'], statistics)
        solve_time = positive(stats[0]['wall_time'], statistics)
        initialization = results / 'initialization_statistics.json'
        ic = json.loads(initialization.read_text())
        ic_time = positive(ic['ida_setup_and_ic_wall_s'], initialization, allow_zero=True)
        if ic_time > solve_time:
            raise ValueError(f'Initialization exceeds solve time in {initialization}')
        groups[base][mpi] = [mpi, *(reduced[name] for name in SECTIONS)]
        summaries.append({'case_id': case['id'], 'base': base, 'mpi': mpi, 'solve_wall_seconds': solve_time,
                          'ida_setup_and_ic_wall_seconds': ic_time,
                          'integration_wall_seconds': solve_time - ic_time,
                          'process_wall_seconds': receipt.get('process_wall_seconds')})
        for source in (path, parameters, timings, statistics, initialization):
            sources.append({'path': str(source), 'sha256': sha256(source)})
    if len(identities) != 1 or len(hosts) != 1:
        raise ValueError('Scaling sweeps must share one executable, manifest and execution host')
    for base, values in groups.items():
        if set(values) != set(ranks):
            raise ValueError(f'{base}: missing MPI counts {sorted(set(ranks) - values.keys())}')
        if len(configurations[base]) != 1:
            raise ValueError(f'{base}: input parameters differ across the scaling sweep')
        reference = next(run for run in summaries if run['base'] == base and run['mpi'] == min(ranks))
        for run in summaries:
            if run['base'] != base:
                continue
            run['reference_mpi'] = reference['mpi']
            ratio = run['mpi'] / reference['mpi']
            run['solve_speedup'] = reference['solve_wall_seconds'] / run['solve_wall_seconds']
            run['solve_parallel_efficiency'] = run['solve_speedup'] / ratio
            run['section_speedups'] = {}
            run['section_parallel_efficiencies'] = {}
            for column, section in enumerate(SECTIONS, start=1):
                timing = values[run['mpi']][column]
                speedup = values[reference['mpi']][column] / timing if timing > 0 else None
                run['section_speedups'][section] = speedup
                run['section_parallel_efficiencies'][section] = speedup / ratio if speedup is not None else None
    revision, binary, manifest_hash = next(iter(identities))
    if machine_text is None:
        machine_text = hardware_tex(selected[0][1]['hardware'])
    # Validate everything before creating any export. Stage on the destination filesystem.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.finalize-', dir=destination.parent) as temporary:
        stage = Path(temporary) / 'profile'
        stage.mkdir()
        (stage / 'machine.tex').write_text(machine_text)
        (stage / 'plot_data').mkdir()
        for base, values in groups.items():
            with (stage / 'plot_data' / f'scaling_{NETWORKS[base]}.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(('n_mpi_processes', *SECTIONS))
                writer.writerows(values[mpi] for mpi in sorted(values))
        copied = []
        metadata = stage / 'provenance'
        metadata.mkdir()
        for name in ('build-provenance.json', 'source-files.json', 'source.patch', 'source-status.txt',
                     'executable-revision.txt', 'executable.sha256', 'linked-libraries.txt',
                     'mpi-version.txt', 'scaling-allocation.txt', 'submission.json', 'manifest.json'):
            source = root / name
            if source.is_file():
                shutil.copyfile(source, metadata / name)
                sources.append({'path': str(source), 'sha256': sha256(source)})
        for path, receipt in selected + outputs:
            if receipt['status'] != 'completed':
                continue
            if receipt.get('machine') != machine or receipt.get('executable_sha256') != binary:
                raise ValueError(f'Output run belongs to another machine/build: {path}')
            if receipt.get('final_time_override') is not None:
                raise ValueError(f'Smoke output cannot become paper data: {path}')
            if not re.fullmatch(r'[a-z0-9_-]+', receipt['case']['id']):
                raise ValueError(f'Invalid case ID in {path}')
            if sha256(path.parent / 'parameters.prm') != receipt['parameters_sha256']:
                raise ValueError(f'Output parameters changed since execution: {path}')
            target = stage / DIRECTORIES[receipt['case']['base']] / 'runs' / receipt['case']['id']
            if target.exists():
                raise ValueError(f'Duplicate output case: {path}')
            shutil.copytree(path.parent, target)
            copied.append(str(target.relative_to(stage)))
        report = {'created_utc': timestamp(), 'machine': machine, 'hostname': next(iter(hosts)),
                  'hardware': selected[0][1]['hardware'], 'executable_revision': revision,
                  'executable_sha256': binary, 'manifest_sha256': manifest_hash,
                  'build_provenance': selected[0][1].get('build_provenance'),
                  'ranks': sorted(ranks), 'sources': sources, 'runs': summaries,
                  'raw_runs': copied, 'scaling_tables_ready': True,
                  'manuscript_profile_complete': False,
                  'remaining': ['Render pressure/network and mass-conservation images from new field runs.',
                                'Review cerebral IDA figure and manuscript numerical discussion/software versions.'],
                  'validation': 'Structural/provenance checks; no numerical validation against archived results.'}
        (stage / 'finalization.json').write_text(json.dumps(report, indent=2) + '\n')
        (stage / 'README.md').write_text(
            f'# {machine}: current-code scaling\n\n'
            'plot_data/scaling_50k.csv and scaling_100k.csv match the manuscript columns.\n'
            'Timings are accumulated solver section wall times, without solution output.\n'
            'Raw receipts, parameters, logs and results are in *_vessels/runs/ and cerebral/runs/.\n'
            'See finalization.json for hashes, solve/integration times and remaining figure work.\n'
            'This export does not replace raksha data or select the manuscript profile.\n')
        if destination.exists():
            # Only machine.tex was allowed above; preserve it until staging has succeeded.
            for child in stage.iterdir():
                child.replace(destination / child.name)
        else:
            stage.replace(destination)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--machine', required=True)
    parser.add_argument('--paper-root', type=Path, default=ROOT / 'paper',
                        help='Manuscript checkout (use ./papers if that is its actual location)')
    parser.add_argument('--ranks', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--description', help='LaTeX hardware/OS description for a new profile')
    args = parser.parse_args(argv)
    try:
        report = finalize(args.run_root.resolve(), args.paper_root.resolve(), args.machine,
                          args.ranks, args.description)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    print(f'Exported 50k/100k scaling tables to {args.paper_root}/scaling/{args.machine}')
    print('Remaining manuscript work: ' + ' '.join(report['remaining']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
