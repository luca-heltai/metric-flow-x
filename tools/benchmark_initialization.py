"""Prepare or run retained initialization comparisons using explicit input archives.

Preparation never runs a solver. Execution requires explicit expected per-rank
CPU affinities and a matching launcher. Use a new node-local output directory.
MPI1 selects the application's direct solver; MPI2 selects its iterative solver. All
inputs, ordering, effective controls, executable hashes and failures are retained.
"""
import argparse, datetime, hashlib, json, math, os, re, shlex, signal, socket, subprocess, sys, threading, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--legacy', type=Path, help='Legacy executable/library archive; required only for the legacy variant.')
p.add_argument('--native', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--source', type=Path, required=True, help='Source tree used for provenance and resolving repository-relative input assets.')
p.add_argument('--case-input', action='append', required=True, metavar='NAME=FILE', help='Named input parameter file; repeat for each case.')
p.add_argument('--cases', nargs='+', help='Selected names; default all supplied case inputs.')
p.add_argument('--mpi-command', default='mpirun -np {ranks}', help='Shell-free launcher template with {ranks}, e.g. mpirun --bind-to core --map-by pe-list=8,9:ordered -np {ranks}.')
p.add_argument('--rank-cpus', action='append', help='Expected OS CPU affinity for one rank, comma-separated; supply in rank order. Required with --execute.')
p.add_argument('--rank-env', default='OMPI_COMM_WORLD_RANK', help='MPI rank environment variable used by the independent affinity probe.')
p.add_argument('--maximum-busy-fraction', type=float, default=.05)
p.add_argument('--idle-sample-seconds', type=float, default=2.)
p.add_argument('--ranks', nargs='+', type=int, default=[1, 2])
p.add_argument('--file-output-interval', type=float, default=None, help='Written-file cadence in seconds; default .001 for startup horizons <=.02, otherwise .01. Use the horizon for initial/final output only.')
p.add_argument('--preserve-reference-tolerances', action='store_true', help='Prospective policy: retain each frozen case input absolute/relative tolerances; do not apply the default strict override.')
p.add_argument('--measurement-kind', choices=['pilot', 'formal'], default='formal')
p.add_argument('--horizons', nargs='+', type=float, default=[0.01, 0.02, 1.0, 3.0])
p.add_argument('--runs', type=int, default=3)
p.add_argument('--variants', nargs='+', choices=['legacy','native_rest','native_0d'], default=['legacy','native_rest','native_0d'])
p.add_argument('--expected-host', help='Optional exact hostname guard.')
p.add_argument('--execute', action='store_true')
p.add_argument('--timeout', type=float, default=21600)
a = p.parse_args()
assert a.runs > 0 and a.ranks and all(x in (1,2) for x in a.ranks)
assert (a.cases is None or len(set(a.cases)) == len(a.cases)) and len(set(a.ranks)) == len(a.ranks)
assert a.file_output_interval is None or (math.isfinite(a.file_output_interval) and a.file_output_interval > 0)
assert all(math.isfinite(x) and x > 0 for x in a.horizons)
source = a.source.resolve()
assert source.is_dir()
case_inputs = {}
for item in a.case_input:
    name, separator, value = item.partition('=')
    assert separator and re.fullmatch(r'[A-Za-z0-9_-]+',name) and name not in case_inputs
    case_inputs[name] = Path(value).resolve()
    assert case_inputs[name].is_file()
if a.cases is None:
    a.cases = list(case_inputs)
assert a.cases and all(name in case_inputs for name in a.cases)
assert 0 <= a.maximum_busy_fraction < 1 and math.isfinite(a.idle_sample_seconds) and a.idle_sample_seconds > 0
expected_cpus = [sorted(set(map(int,x.split(',')))) for x in (a.rank_cpus or [])]
assert all(cpus and min(cpus)>=0 for cpus in expected_cpus)
if a.execute:
    assert len(expected_cpus) >= max(a.ranks), 'Supply --rank-cpus explicitly for every requested rank'
def launcher(ranks):
    assert '{ranks}' in a.mpi_command
    return shlex.split(a.mpi_command.format(ranks=ranks))
out = a.output.resolve()
assert not out.exists(), 'Use a new output directory; existing results are never overwritten'
out.mkdir(parents=True)
archives = {'native_rest': a.native.resolve(), 'native_0d': a.native.resolve()}
if a.legacy is not None:
    archives['legacy'] = a.legacy.resolve()
variants = a.variants
assert variants and len(set(variants)) == len(variants) and all(x in archives for x in variants)
for archive in archives.values():
    assert (archive/'blood_flow').is_file() and (archive/'libmetric_flow_x.so').is_file()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run_checked_solver(command, folder, environment, timeout):
    """Abort this process group if the Release SDK prints a fatal solver error."""
    path = folder/'process.log'
    failed = []
    done = threading.Event()
    with path.open('w') as log:
        start = time.perf_counter()
        process = subprocess.Popen(command, cwd=folder, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        def monitor():
            # Read a separate descriptor: solver output remains direct-to-file.
            with path.open() as stream:
                carry = ''
                while not done.wait(.05):
                    text = carry + stream.read()
                    carry = text[-32:]
                    if '[IDAS ERROR]' in text or '[IDA ERROR]' in text:
                        failed.append(text)
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        return
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            elapsed = time.perf_counter() - start
            done.set()
            watcher.join()
    text = path.read_text()
    if failed or '[IDAS ERROR]' in text or '[IDA ERROR]' in text or code != 0:
        (folder/'failure.json').write_text(json.dumps(dict(
            returncode=code, whole_process_wall_s=elapsed,
            solver_error_detected=bool(failed or '[IDAS ERROR]' in text or '[IDA ERROR]' in text),
            diagnostic_tail=text[-8192:]), indent=2)+'\n')
        raise RuntimeError('Solver process failed; retained '+str(folder/'failure.json'))
    return elapsed

def replace(text, name, value):
    text, count = re.subn(r'(?m)^(\s*set ' + re.escape(name) + r'\s*=\s*)[^\n]*', lambda m: m[1] + str(value), text)
    assert count == 1, (name, count)
    return text

jobs = []
for case in a.cases:
    text = case_inputs[case].read_text()
    for asset_key in ['Vtk file path for mesh input', 'Inflow CSV file', 'Terminal parameters CSV file']:
        matches = re.findall(r'(?m)^\s*set '+re.escape(asset_key)+r'\s*=\s*([^\n#]*)',text)
        if matches and matches[0].strip() and not Path(matches[0].strip()).is_absolute():
            asset = (source/matches[0].strip()).resolve()
            assert asset.is_file(), asset
            text = replace(text,asset_key,asset)
    for ranks in a.ranks:
        for horizon in a.horizons:
            file_interval = a.file_output_interval if a.file_output_interval is not None else (0.001 if horizon <= 0.02 else 0.01)
            # Apply identical solver controls to every initialization policy.
            solver_policy = 'serial_direct' if ranks == 1 else 'iterative'
            for run in range(a.runs):
                # Each policy occupies each order position over three triplets.
                order = variants[run % len(variants):] + variants[:run % len(variants)]
                for policy in order:
                    tag = f'{case}_mpi{ranks}_t{horizon:g}_pair{run}_{policy}'
                    folder = out/tag; folder.mkdir()
                    prm = text
                    overrides = {'Output directory': folder/'output', 'Verbosity (console depth)': 0, 'Initial time': 0, 'Final time': horizon, 'Time interval between each output': '0.001', 'Use direct solver': 'true' if ranks == 1 else 'false'}
                    if not a.preserve_reference_tolerances:
                        overrides.update({'Absolute error tolerance': '1e-12', 'Relative error tolerance': '1e-7'})
                    for key, value in overrides.items():
                        prm = replace(prm, key, value)
                    extra = f'  set Write terminal CSV files = true\n  set Output time interval = {file_interval}\n'
                    if policy != 'legacy':
                        extra += ('  set Initial condition policy = native\n  set Initial state source = ' + ('rest' if policy=='native_rest' else 'steady network') + '\n  set Check Jacobian at startup = false\n')
                    prm = prm.replace('subsection MetricFlowSystem<1, 3>\n', 'subsection MetricFlowSystem<1, 3>\n'+extra, 1)
                    (folder/'input.prm').write_text(prm)
                    archive = archives[policy]
                    command = launcher(ranks)+[str(archive/'blood_flow'),str(folder/'input.prm')]
                    actual_controls = {}
                    for key in ['Absolute error tolerance','Relative error tolerance','Initial step size','Minimum step size','Maximum order of BDF']:
                        matches = re.findall(r'(?m)^\s*set '+re.escape(key)+r'\s*=\s*([^\n#]+)',prm)
                        assert len(matches)==1, (key,matches)
                        actual_controls[key]=float(matches[0])
                    jobs.append(dict(actual_solver_controls=actual_controls,case=case,ranks=ranks,horizon=horizon,run=run,policy=policy,solver_policy=solver_policy,use_direct_solver=(ranks == 1),file_output_interval=file_interval,folder=str(folder),command=command,input_sha256=sha(folder/'input.prm')))
protocol = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), expected_host=a.expected_host, actual_host=socket.gethostname(), source=str(source), case_input_sha256={name:sha(path) for name,path in case_inputs.items()}, mpi_command=a.mpi_command, expected_rank_cpus=expected_cpus, maximum_busy_fraction=a.maximum_busy_fraction, idle_sample_seconds=a.idle_sample_seconds, script_sha256=sha(Path(__file__)), runs=a.runs, variants=variants, cases=a.cases, horizons=a.horizons, ranks=a.ranks, preserve_reference_tolerances=a.preserve_reference_tolerances, solver_error_monitor="abort own process group on IDA/IDAS ERROR, including Release ignored statuses", measurement_kind=a.measurement_kind, exclude_from_formal_summary=(a.measurement_kind == 'pilot'), file_output_interval_override=a.file_output_interval, solver_policy_by_ranks={'1':'serial_direct','2':'iterative'}, archive_sha256={policy:{n:sha(archive/n) for n in ['blood_flow','libmetric_flow_x.so']} for policy,archive in archives.items()}, wall_definition='perf_counter around complete mpirun subprocess including startup, setup, writes and teardown', phase_rank='rank 0 wall from initialization_statistics.json', ic_equivalence='native_rest vs legacy must pass separate solver-weight validation; native_0d differs physically at startup', late_accuracy='0D ramp versus uniform and tight reference is separate, mandatory before ramp claims', no_filtering=True, jobs=jobs)
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
print(out/'protocol.json', flush=True)
if not a.execute:
    raise SystemExit(0)
assert a.expected_host is None or socket.gethostname() == a.expected_host
# Independent affinity probe before solver measurements, retained verbatim.
env = dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMPI_ALLOW_RUN_AS_ROOT='1',OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1')
probe_code = 'import json,os; print(json.dumps({"rank":int(os.environ['+repr(a.rank_env)+']),"affinity":sorted(os.sched_getaffinity(0))}))'
probe = subprocess.check_output(launcher(max(a.ranks))+[sys.executable,'-c',probe_code],env=env,text=True)
(out/'affinity.jsonl').write_text(probe)
assert {x['rank']:x['affinity'] for x in map(json.loads,probe.splitlines())} == {rank:expected_cpus[rank] for rank in range(max(a.ranks))}
selected_cpus = {'cpu'+str(cpu) for cpus in expected_cpus[:max(a.ranks)] for cpu in cpus}
def cpu_sample():
    data = {}
    for line in Path('/proc/stat').read_text().splitlines():
        fields=line.split()
        if fields and fields[0] in selected_cpus:
            values=list(map(int,fields[1:9]));data[fields[0]]=(sum(values),values[3]+values[4])
    assert set(data)==selected_cpus
    return data
before=cpu_sample();time.sleep(a.idle_sample_seconds);after=cpu_sample()
busy={cpu:1-(after[cpu][1]-v[1])/(after[cpu][0]-v[0]) for cpu,v in before.items()}
(out/'idle_preflight.json').write_text(json.dumps(busy,indent=2)+'\n')
assert max(busy.values()) <= a.maximum_busy_fraction, busy
for job in jobs:
    folder=Path(job['folder']);archive=archives[job['policy']]
    run_env=dict(env,LD_LIBRARY_PATH=str(archive)+':'+os.environ.get('LD_LIBRARY_PATH',''))
    ldd=subprocess.check_output(['ldd',str(archive/'blood_flow')],env=run_env,text=True)
    assert f'libmetric_flow_x.so => {archive}/libmetric_flow_x.so' in ldd
    (folder/'ldd.txt').write_text(ldd)
    elapsed=run_checked_solver(job['command'],folder,run_env,a.timeout)
    phases=json.loads((folder/'output/initialization_statistics.json').read_text())
    for key in ['preparation_wall_s','ida_setup_and_ic_wall_s','first_ida_output_interval_wall_s']:
        assert math.isfinite(phases[key]) and phases[key]>=0, (job,key,phases)
    record=dict(job,whole_process_wall_s=elapsed,initialization_statistics=phases)
    with (out/'records.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n');stream.flush();os.fsync(stream.fileno())
    print(job['folder'],elapsed,flush=True)
(out/'complete.json').write_text(json.dumps(dict(records=len(jobs),returncode=0))+'\n')
