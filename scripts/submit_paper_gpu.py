#!/usr/bin/env python3
"""Submit CPU/MPI paper experiments to a configurable Slurm partition (stdlib only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from reproduce_paper import MANIFEST, ROOT, sha256


def quote(value) -> str:
    return shlex.quote(str(value))


def capture_source() -> dict:
    tracked = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
    return {'revision': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'files': {name: sha256(ROOT / name) for name in tracked}}


def snapshot_executable(executable: Path, root: Path, revision: str) -> None:
    """Freeze a reused binary and its application library before any solver runs."""
    if not executable.is_file():
        raise ValueError(f'Executable missing: {executable}')
    runtime = root / 'runtime'
    (runtime / 'lib').mkdir(parents=True, exist_ok=False)
    listing = subprocess.check_output(['ldd', str(executable)], text=True)
    (root / 'linked-libraries.txt').write_text(listing)
    if 'not found' in listing:
        raise ValueError('Reused executable has unresolved shared libraries')
    libraries = re.findall(r'(libmetric_flow_x\S*)\s+=>\s+(.+?)\s+\(0x', listing)
    copied = {}
    for source, target in [(executable, runtime / 'blood_flow')] + [
            (Path(path), runtime / 'lib' / name) for name, path in libraries]:
        expected = sha256(source)
        shutil.copy2(source, target)
        if sha256(target) != expected or sha256(source) != expected:
            raise ValueError(f'Executable/library changed while copying: {source}')
        if target.parent.name == 'lib':
            copied[str(target)] = expected
    (root / 'executable-revision.txt').write_text(revision + '\n')
    (root / 'build-provenance.json').write_text(json.dumps({
        'executable_revision': revision,
        'executable_sha256': sha256(runtime / 'blood_flow'),
        'runtime_library_sha256': copied,
        'source_executable': str(executable),
        'reused_executable': True,
        'revision_source': 'user-supplied build revision; runtime smoke checks recorded separately',
    }, indent=2) + '\n')


def prepare(args) -> list[tuple[str, list[str], Path]]:
    manifest = json.loads(MANIFEST.read_text())
    scaling = [case for case in manifest['runs'] if case['suite'] == 'scaling'
               and case['mpi'] in args.ranks]
    outputs = [] if args.scaling_only else [
        case for case in manifest['runs'] if case['suite'] != 'scaling']
    # Verify the requested matrix before submitting any jobs.
    for base in {case['base'] for case in scaling}:
        ranks = [case['mpi'] for case in scaling if case['base'] == base]
        if ranks != sorted(args.ranks):
            raise ValueError(f'Unexpected scaling ranks for {base}: {ranks}')
    for case in scaling + outputs:
        blocker = manifest['provenance'][case['base']].get('execution_blocker')
        if blocker:
            raise ValueError(f"{case['id']}: {blocker}")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'jobs').mkdir()
    (root / 'logs').mkdir()
    (root / 'output-cases.json').write_text(json.dumps([c['id'] for c in outputs], indent=2) + '\n')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Pending jobs must use the exact source files inspected at submission time.
    source_state = capture_source()
    (root / 'source-files.json').write_text(json.dumps(source_state, indent=2) + '\n')
    profile = root / 'paper-output/scaling' / args.machine
    profile.mkdir(parents=True)
    source_profile = ROOT / 'paper/scaling' / args.machine / 'machine.tex'
    if args.description:
        (profile / 'machine.tex').write_text(
            '\\newcommand{\\ScalingMachineDescription}{' + args.description + '}\n')
    elif source_profile.is_file():
        (profile / 'machine.tex').write_text(source_profile.read_text())
    else:
        (profile / 'machine.tex').write_text('% AUTO_FROM_SCALING_HARDWARE\n')
    # All MPI ranks stay on this node. PRRTE's local launcher does not require
    # Slurm's unavailable pmix plugin, and remains inside the job's CPU cgroup.
    launcher = root / 'jobs/mpirun-local.sh'
    launcher.write_text('''#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
exec mpirun --prtemca ras '^slurm' --host "localhost:${SLURM_NTASKS:?}" \\
  --map-by core --bind-to core "$@"
''')
    launcher.chmod(0o755)
    setup = f'''#!/usr/bin/env bash
set -eo pipefail
source {quote(args.env.resolve())}
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
cd {quote(ROOT)}
python3 - {quote(root / 'source-files.json')} <<'PYSOURCE'
import json, pathlib, subprocess, sys
sys.path.insert(0, 'scripts')
from reproduce_paper import sha256
state = json.loads(pathlib.Path(sys.argv[1]).read_text())
if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != state['revision']:
    sys.exit('Source revision changed since submission; submit a fresh campaign')
for name, expected in state['files'].items():
    if not pathlib.Path(name).is_file() or sha256(pathlib.Path(name)) != expected:
        sys.exit('Source changed since submission: ' + name)
PYSOURCE
'''
    if args.executable:
        setup += f'export LD_LIBRARY_PATH={quote(root / "runtime/lib")}:"${{LD_LIBRARY_PATH:-}}"\n'
    build = root / 'build'
    executable = root / 'runtime/blood_flow' if args.executable else build / 'blood_flow'
    runner = [sys.executable, str(ROOT / 'scripts/reproduce_paper.py'),
              '--dataset-root', str(args.dataset_root.resolve()),
              '--executable', str(executable),
              '--machine', args.machine,
              '--build-provenance', str(root / 'build-provenance.json'),
              '--mpi-launcher', str(launcher), '--mpi-arg=--report-bindings']
    run = shlex.join(runner)
    preparation = ('--suite scaling --ranks ' + ' '.join(map(str, args.ranks))
                   if args.scaling_only else '--suite all')
    build_body = f'''
git rev-parse HEAD > {quote(root / 'executable-revision.txt')}
git diff HEAD -- > {quote(root / 'source.patch')}
git status --porcelain > {quote(root / 'source-status.txt')}
python3 scripts/reproduce_paper.py {preparation} --prepare-only \
  --dataset-root {quote(args.dataset_root.resolve())} --output-root {quote(root / 'prepared-inputs')}
cmake -S {quote(ROOT)} -B {quote(build)} -G Ninja \\
  -DCMAKE_CXX_COMPILER="$(command -v mpicxx)" \
  -DCMAKE_BUILD_TYPE=Release -DMETRIC_FLOW_X_BACKEND=PETSc \\
  -DDEAL_II_DIR="$(spack location -i dealii)"
cmake --build {quote(build)} --target blood_flow --parallel 4
mpirun --version > {quote(root / 'mpi-version.txt')}
ldd {quote(build / 'blood_flow')} > {quote(root / 'linked-libraries.txt')}
sha256sum {quote(build / 'blood_flow')} > {quote(root / 'executable.sha256')}
python3 - {quote(root)} {quote(build / 'blood_flow')} <<'PYBUILD'
import json, pathlib, subprocess, sys
sys.path.insert(0, 'scripts')
from reproduce_paper import sha256
root, executable = map(pathlib.Path, sys.argv[1:])
revision = (root / 'executable-revision.txt').read_text().strip()
# A changing source tree cannot be attributed to the recorded build revision.
if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != revision or
        subprocess.check_output(['git', 'diff', 'HEAD', '--']) != (root / 'source.patch').read_bytes()):
    sys.exit('Source changed during compilation; submit a fresh campaign')
(root / 'build-provenance.json').write_text(json.dumps(dict(
    executable_revision=revision, executable_sha256=sha256(executable),
    source_patch_sha256=sha256(root / 'source.patch'),
    source_dirty=bool((root / 'source.patch').stat().st_size),
    backend='PETSc', build_type='Release'), indent=2) + '\\n')
PYBUILD
'''
    if args.executable:
        build_body = f'''
git diff HEAD -- > {quote(root / 'source.patch')}
git status --porcelain > {quote(root / 'source-status.txt')}
python3 - {quote(args.executable.resolve())} {quote(root)} {quote(args.executable_revision)} <<'PYREUSE'
import pathlib, sys
sys.path.insert(0, 'scripts')
from submit_paper_gpu import snapshot_executable
snapshot_executable(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3])
PYREUSE
python3 scripts/reproduce_paper.py {preparation} --prepare-only \\
  --dataset-root {quote(args.dataset_root.resolve())} --output-root {quote(root / 'prepared-inputs')}
python3 tests/test_output_schedule.py --executable {quote(executable)} \\
  --workdir {quote(root / 'validation')} > {quote(root / 'validation.log')} 2>&1
mpirun --version > {quote(root / 'mpi-version.txt')}
'''
    common = ['--partition=' + args.partition, '--nodes=1', '--hint=nomultithread', '--gres=none',
              '--export=ALL', '--chdir=' + str(ROOT)]
    if args.node:
        common.append('--nodelist=' + args.node)
    specs = []

    def job(name, options, body):
        if args.scaling_only and name == 'outputs':
            return
        script = root / f'jobs/{name}.sh'
        script.write_text(setup + body)
        specs.append((name, common + options + [f'--job-name=paper-{name}',
                     f'--output={root}/logs/{name}-%A_%a.log'], script))

    job('verify' if args.executable else 'build',
        ['--ntasks=1', '--cpus-per-task=4', '--mem=32G', '--time=04:00:00'], build_body)
    # One allocation holds the SAME exclusive node across both complete sweeps.
    body = f'\nscontrol show job "$SLURM_JOB_ID" > {quote(root / "scaling-allocation.txt")}\n'
    body += f'revision=$(cat {quote(root / "executable-revision.txt")})\n'
    for case in scaling:
        body += f'{run} {quote(case["id"])} --executable-revision "$revision" --output-root {quote(root / "scaling")}\n'
    job('scaling', ['--ntasks=' + str(max(args.ranks)), '--cpus-per-task=1', '--exclusive', '--mem=0',
                    '--time=' + args.time], body)
    # All array elements are independent and eligible together after scaling.
    body = f'''
case_id=$(python3 - {quote(root / 'output-cases.json')} "$SLURM_ARRAY_TASK_ID" <<'PYCASE'
import json, sys
print(json.load(open(sys.argv[1]))[int(sys.argv[2])])
PYCASE
)
revision=$(cat {quote(root / 'executable-revision.txt')})
ranks=4
case "$case_id" in mms-*|bifurcation) ranks=1 ;; esac
{run} "$case_id" --mpi "$ranks" --executable-revision "$revision" --output-root {quote(root / 'outputs')}
'''
    if args.smoke:
        # A short execution of the same scheduling path, explicitly recorded.
        body = body.rstrip() + ' --final-time 0.001\n'
        # Also shorten scaling without changing the persistent canonical matrix.
        _, _, script = specs[-1]
        lines = script.read_text().splitlines()
        script.write_text('\n'.join(line + ' --final-time 0.001' if line.startswith(run + ' ') else line
                                    for line in lines) + '\n')
    job('outputs', ['--ntasks=4', '--cpus-per-task=1', '--mem=16G', '--time=' + args.time,
                    f'--array=0-{len(outputs)-1}'], body)
    collection = '' if args.scaling_only else (
        f'\npython3 scripts/collect_paper_outputs.py {quote(root / "outputs")} '
        f'--output {quote(root / "plot-inputs")}\n')
    if not args.smoke:
        collection += (f'python3 scripts/finalize_paper_output.py {quote(root)} '
                       f'--machine {quote(args.machine)} --paper-root {quote(root / "paper-output")} '
                       '--ranks ' + ' '.join(map(str, args.ranks)) + '\n')
    job('collect', ['--ntasks=1', '--cpus-per-task=1', '--mem=16G', '--time=' + args.time], collection)
    return specs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True, help='Fresh shared directory for jobs, build and results')
    parser.add_argument('--dataset-root', type=Path, default=ROOT)
    parser.add_argument('--executable', type=Path, help='Reuse an existing binary instead of compiling; snapshot it with its application library')
    parser.add_argument('--executable-revision', help='Full source commit used for --executable')
    parser.add_argument('--env', type=Path, default=Path('/software/spackgpu-v2/env.sh'), help='Environment script providing Spack, CMake, Ninja and OpenMPI 5')
    parser.add_argument('--time', default='2-00:00:00', help='Time limit per scaling/outputs/collection job')
    parser.add_argument('--dry-run', action='store_true', help='Write scripts and print sbatch commands without submitting')
    parser.add_argument('--smoke', action='store_true', help='Override final time to 0.001 s for workflow validation, not paper results')
    parser.add_argument('--scaling-only', action='store_true', help='Build and measure 50k/100k sweeps, without field/application runs')
    parser.add_argument('--node', help='Optionally pin all jobs to this exact node; otherwise Slurm selects nodes')
    parser.add_argument('--partition', default='gpu', help='Slurm partition for the campaign')
    parser.add_argument('--machine', help='Paper profile label (default: pinned node or partition name)')
    parser.add_argument('--description', help='LaTeX hardware/OS description for a new machine profile')
    parser.add_argument('--ranks', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64],
                        help='Canonical MPI counts to measure on the reserved node')
    args = parser.parse_args(argv)
    args.machine = args.machine or args.node or args.partition
    try:
        if args.executable and not re.fullmatch(r'[0-9a-f]{40}', args.executable_revision or ''):
            raise ValueError('--executable requires --executable-revision with the full build commit')
        if args.executable_revision and not args.executable:
            raise ValueError('--executable-revision requires --executable')
        if not all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name)
                   for name in (args.node, args.machine) if name is not None):
            raise ValueError('Invalid node or machine name')
        if args.machine == 'raksha':
            raise ValueError('raksha is the historical reference; choose a new profile')
        if len(set(args.ranks)) != len(args.ranks) or not set(args.ranks) <= {1, 2, 4, 8, 16, 32, 64}:
            raise ValueError('Choose unique canonical MPI counts: 1 2 4 8 16 32 64')
        args.ranks.sort()
        if not args.env.is_file():
            raise ValueError(f'Environment script missing: {args.env}')
        if args.output_root.exists():
            raise ValueError(f'Output directory already exists: {args.output_root}; choose a fresh path')
        if not args.dry_run:
            manifest = json.loads(MANIFEST.read_text())
            dataset = args.dataset_root.resolve()
            meshes = {key: value for key, value in manifest['meshes'].items()
                      if not args.scaling_only or key in ('network50k', 'network100k')}
            missing = any(not (dataset / mesh['path']).is_file() or
                          sha256(dataset / mesh['path']) != mesh['sha256']
                          for mesh in meshes.values())
            if missing:
                print('Downloading pinned datasets on the submission host...', flush=True)
                subprocess.run(['bash', str(ROOT / 'scripts/download_datasets.sh'), str(dataset)], check=True)
            for mesh in meshes.values():
                if sha256(dataset / mesh['path']) != mesh['sha256']:
                    raise ValueError(f"Dataset checksum mismatch: {mesh['path']}")
        specs = prepare(args)
        submitted = {}
        record = args.output_root.resolve() / 'submission.json'
        for name, options, script in specs:
            command = ['sbatch', '--parsable', '--kill-on-invalid-dep=yes', *options]
            if submitted:
                previous = next(reversed(submitted.values()))
                command.append('--dependency=afterok:' + previous)
            command.append(str(script))
            print(shlex.join(command), flush=True)
            if args.dry_run:
                submitted[name] = f'<{name}-job-id>'
            else:
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                job_id = result.stdout.strip().split(';')[0]
                if not job_id.isdecimal():
                    raise ValueError(f'Unexpected sbatch result: {result.stdout!r}')
                submitted[name] = job_id
                print(f'{name}: {job_id}', flush=True)
            # Preserve successfully submitted IDs even if a later sbatch fails.
            record.write_text(json.dumps({'dry_run': args.dry_run, 'smoke': args.smoke,
                                          'machine': args.machine, 'node': args.node,
                                          'partition': args.partition,
                                          'source_executable': str(args.executable.resolve()) if args.executable else None,
                                          'ranks': args.ranks,
                                          'scaling_only': args.scaling_only,
                                          'jobs': submitted}, indent=2) + '\n')
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
