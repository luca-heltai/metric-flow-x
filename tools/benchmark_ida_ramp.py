#!/usr/bin/env python3
"""Prepare, then explicitly execute the predeclared long-run ramp comparison."""
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import time

TOOLS = Path(__file__).resolve().parent
SOURCE_ROOT = TOOLS.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_revision(source):
    git = shutil.which('git')
    if not git:
        return None
    result = subprocess.run([git,'-C',str(source),'rev-parse','HEAD'],capture_output=True,text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def parameters(source, output, policy, variant):
    p, r = policy["physical"], policy["ramp"]
    atol = policy["reference_atol"] if variant == "reference" else policy["base_atol"]
    rtol = policy["reference_rtol"] if variant == "reference" else policy["base_rtol"]
    return f'''subsection Metric Flow Parameters
 set Density (rho) = {p['rho']}
 set Viscosity coefficient (mu) = {p['mu']}
 set Profile constant for friction term (xi) = {p['xi']}
end
subsection Functions
 set Inflow function = {policy['waveform']}
end
subsection MetricFlowSystem<1, 3>
 set Vtk file path for mesh input = {source/'parameters/single_vessel.vtk'}
 set Finite element degree = 1
 set Number of global refinement = 0
 set Number of refinement cycles = 1
 set Initial state source = steady network
 set Initial condition policy = native
 set Use componentwise absolute tolerances = false
 set Use direct solver = true
 set Numerical flux type = {policy['numerical_flux']}
 set Theta (penalty parameter) = {policy['theta']}
 set Theta Boundary (stability parameter) = {policy['theta_boundary']}
 set Outlet boundary condition type = RCR
 set Terminal parameter source = Parameters
 set Default terminal R1 = {p['R1']}
 set Default terminal R2 = {p['R2']}
 set Default terminal C = {p['C']}
 set Default terminal P_out = {p['P_out']}
 set Output terminal CSV file = terminals.csv
 set Output directory = {output}
 set Output time interval = {policy['file_interval']}
 set Verbosity (console depth) = 0
 set Check Jacobian at startup = false
 set Maximum IDA internal steps per output = {policy['maximum_ida_internal_steps_per_output']}
 set Tolerance ramp enabled = {'true' if variant == 'ramp' else 'false'}
 set Tolerance ramp initial relative factor = {r['relative_factor']}
 set Tolerance ramp initial absolute factor = {r['absolute_factor']}
 set Tolerance ramp hold duration = {r['hold_duration']}
 set Tolerance ramp end time = {r['end_time']}
 subsection IDA parameters
  set Initial time = {policy['initial_time']}
  set Final time = {policy['final_time']}
  set Time interval between each output = {policy['callback_interval']}
  subsection Error control
   set Absolute error tolerance = {atol}
   set Relative error tolerance = {rtol}
  end
  subsection Running parameters
   set Initial step size = {policy['initial_step_size']}
   set Minimum step size = {policy['minimum_step_size']}
   set Maximum order of BDF = {policy['maximum_bdf_order']}
   set Maximum number of nonlinear iterations = 10
  end
 end
end
'''


def campaign_plan(policy, pilot):
    pairs = 1 if pilot else policy['runs_per_variant']
    names = policy['performance_order'][:2*pairs] + ['reference']
    return names, pairs, 'correctness_only' if pilot else 'formal_performance'


def acceptance_summary(validations, policy, n_pairs):
    """Separate baseline equivalence from tighter-run accuracy diagnostics.

    Missing acceptance_target denotes the original v3 all-comparison gate.
    Structural/physics validation exceptions must be handled by the caller as
    campaign failures; a diagnostic designation never suppresses those checks.
    """
    mode = policy.get('acceptance_target', 'all_comparisons')
    if mode not in ('all_comparisons', 'uniform_baseline_equivalence'):
        raise ValueError('Unknown acceptance target: '+str(mode))
    if not isinstance(n_pairs, int) or isinstance(n_pairs, bool) or n_pairs < 1:
        raise ValueError('Expected a positive integer pair count')
    if mode == 'uniform_baseline_equivalence':
        expected = {'mandatory_comparisons':'every_uniform_ramp_pair',
                    'diagnostic_comparisons':'each_production_run_vs_tighter_run',
                    'absolute_accuracy_certified':False}
        if any(policy.get(key) != value for key, value in expected.items()):
            raise ValueError('Incomplete equivalence acceptance policy')
    pairs = {f'pair-{i}' for i in range(n_pairs)}
    references = {f'{variant}-{i}' for i in range(n_pairs) for variant in ('uniform', 'ramp')}
    if set(validations) != pairs | references:
        raise ValueError('Missing or unexpected physical comparisons')
    metric_names = {'area', 'velocity', 'pressure', 'flow', 'terminal_area',
                    'terminal_velocity', 'terminal_pressure', 'terminal_flow',
                    'reconstructed_Pc', 'volume', 'dVdt', 'net_flux', 'imbalance'}
    for comparison in validations.values():
        metrics = comparison['metrics']
        if set(metrics) != metric_names:
            raise ValueError('Missing or unexpected physical metrics')
        for metric in metrics.values():
            error = metric['max_normalized_error']
            if not math.isfinite(error) or error < 0 or metric['accepted'] is not (error < 1):
                raise ValueError('Invalid physical metric or inconsistent strict acceptance')
        if comparison['accepted'] is not all(metric['accepted'] for metric in metrics.values()):
            raise ValueError('Inconsistent comparison acceptance')
    equivalence = all(validations[name]['accepted'] for name in pairs)
    diagnostics = all(validations[name]['accepted'] for name in references)
    return {'accepted':equivalence and (diagnostics if mode == 'all_comparisons' else True),
            'acceptance_target':mode, 'equivalence_passed':equivalence,
            'tighter_reference_diagnostics_passed':diagnostics,
            'absolute_accuracy_certified':False}


def check_solver_log(path):
    if any(token in path.read_text() for token in ('[IDAS ERROR]', '[IDA ERROR]')):
        raise RuntimeError('Native IDA error in retained '+str(path))


def cpu_usage(cpus):
    result = {}
    for line in Path('/proc/stat').read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in {'cpu'+str(cpu) for cpu in cpus}:
            numbers = list(map(int, fields[1:]))
            result[fields[0]] = (sum(numbers), numbers[3]+numbers[4])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE_ROOT, help='Source checkout containing parameters/single_vessel.vtk')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy', type=Path, default=TOOLS/'ida_ramp_policy.json')
    parser.add_argument('--physics-helper', type=Path, default=TOOLS/'compare_inflow_trajectories.py')
    parser.add_argument('--mpirun', type=Path, help='OpenMPI launcher; defaults to mpirun on PATH')
    parser.add_argument('--map-by', help='OpenMPI placement override, recorded in the protocol')
    parser.add_argument('--library-directory', type=Path, help='Matching MetricFlowX library directory; defaults to executable directory')
    parser.add_argument('--execute', action='store_true', help='Run only after root review and exclusive host ownership')
    parser.add_argument('--pilot', action='store_true', help='Only uniform/ramp/tighter-reference correctness runs; all timings excluded from formal performance')
    args = parser.parse_args()
    executable, source, output = (getattr(args, name).resolve() for name in ('executable', 'source', 'output'))
    policy = json.loads(args.policy.read_text())
    names, actual_pairs, measurement_kind = campaign_plan(policy,args.pilot)
    spec = importlib.util.spec_from_file_location('late_comparison',TOOLS/'compare_late_ramp_outputs.py')
    validator = importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)
    physics = validator.load_physics(args.physics_helper,policy)
    if args.pilot and policy.get('pilot') != {'measurement_kind':'correctness_only',
                                               'actual_pair_count':1,
                                               'all_timings_excluded_from_formal_performance':True}:
        raise RuntimeError('The policy does not declare the correctness-only pilot')
    placement = args.map_by or policy['map_by']
    library_directory = (args.library_directory or executable.parent).resolve()
    launcher_path = str(args.mpirun.resolve()) if args.mpirun else shutil.which('mpirun')
    if output.exists():
        if not args.execute:
            raise RuntimeError('Preparation output already exists; choose a fresh directory')
        receipt = json.loads((output/'protocol.json').read_text())
        if receipt.get('measurement_kind') != measurement_kind or receipt.get('placement') != placement or receipt.get('library_directory') != str(library_directory) or receipt.get('physics_helper_sha256') != sha(args.physics_helper) or receipt['executed'] or receipt['policy_sha256'] != sha(args.policy) or receipt['executable_sha256'] != sha(executable) or receipt['mesh_sha256'] != sha(source/'parameters/single_vessel.vtk'):
            raise RuntimeError('Existing preparation was executed or its reviewed artifacts changed')
        for name in names:
            case = output/name
            expected = parameters(source,case/'output',policy,name.split('-')[0])
            if (case/'parameters.prm').read_text() != expected or sha(case/'parameters.prm') != receipt['parameter_sha256'][name]:
                raise RuntimeError('Reviewed parameter bytes changed: '+name)
    else:
        output.mkdir(parents=True, exist_ok=False)
        for name in names:
            case = output/name
            case.mkdir()
            (case/'parameters.prm').write_text(parameters(source, case/'output', policy, name.split('-')[0]))
        receipt = {'declared_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'host':socket.gethostname(),
                   'policy':policy, 'policy_sha256':sha(args.policy), 'executable':str(executable),
                   'executable_sha256':sha(executable), 'mesh_sha256':sha(source/'parameters/single_vessel.vtk'),
                   'source_revision':source_revision(source),
                   'script_sha256':sha(Path(__file__)), 'cases':names,
                   'measurement_kind':measurement_kind,'actual_pair_count':actual_pairs,
                   'formal_pair_count':policy['runs_per_variant'],
                   'placement':placement,'library_directory':str(library_directory),
                   'physics_helper_sha256':sha(args.physics_helper),
                   'all_timings_excluded_from_formal_performance':args.pilot,
                   'parameter_sha256':{name:sha(output/name/'parameters.prm') for name in names},
                   'executed':False}
        (output/'protocol.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if not args.execute:
        print('Prepared only; no solver processes launched: '+str(output))
        return 0
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env['LD_LIBRARY_PATH']=str(library_directory)+os.pathsep+env.get('LD_LIBRARY_PATH','')
    env['OMPI_ALLOW_RUN_AS_ROOT']='1'; env['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM']='1'
    launcher = launcher_path
    if not launcher:
        raise RuntimeError('MPI launcher unavailable')
    prefix = [launcher,'--bind-to',policy['bind_to'],'--map-by',placement,'-n','1']
    check = subprocess.run(prefix+['python3','-c','import os,json;print(json.dumps(sorted(os.sched_getaffinity(0))))'],env=env,capture_output=True,text=True,check=True)
    affinity = json.loads(next(line for line in check.stdout.splitlines() if line.startswith('[')))
    if not affinity:
        raise RuntimeError('Rank affinity is empty')
    topology = {cpu: ((Path('/sys/devices/system/cpu')/f'cpu{cpu}/topology/physical_package_id').read_text().strip(),
                      (Path('/sys/devices/system/cpu')/f'cpu{cpu}/topology/core_id').read_text().strip()) for cpu in affinity}
    if len(set(topology.values())) != 1:
        raise RuntimeError('Expected binding to one physical core: '+str(affinity))
    before = cpu_usage(affinity); time.sleep(2); after = cpu_usage(affinity)
    utilization = {cpu:100*(1-(after[cpu][1]-before[cpu][1])/(after[cpu][0]-before[cpu][0])) for cpu in before}
    if max(utilization.values()) > policy['preflight_maximum_busy_percent']:
        raise RuntimeError('Selected cores busy: '+str(utilization))
    process_list = subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,comm=,args='],text=True)
    forbidden = {'ninja','cc1plus','blood_flow','blood_flow_debug','benchmark_assembly',executable.name}
    if any(line.split()[1] in forbidden for line in process_list.splitlines() if len(line.split()) >= 2):
        raise RuntimeError('Competing own build/solver process exists')
    linked = subprocess.check_output(['ldd',str(executable)],env=env,text=True)
    own_libraries = [(line.split('=>',1)[0].strip(),line.split('=>',1)[1].strip().split()[0]) for line in linked.splitlines()
                     if any(name+' =>' in line for name in ('libmetric_flow_x.so','libmetric_flow_x_debug.so'))]
    own_name, own_library = own_libraries[0] if len(own_libraries) == 1 else ('', '')
    if not own_library or not Path(own_library).is_file() or Path(own_library).resolve() != (library_directory/own_name).resolve():
        raise RuntimeError('Cannot verify own MetricFlowX library')
    artifacts = {str(path):sha(path) for path in (executable,Path(own_library))}
    receipt.update(executed=True,affinity=affinity,topology=topology,preflight_utilization_percent=utilization,
                   own_processes_before=process_list,ldd=linked,artifacts_sha256=artifacts,records=[])
    protocol = output/'protocol.json'
    protocol.write_text(json.dumps(receipt,indent=2)+'\n')
    try:
        for name in names:
            case = output/name
            command = prefix+[str(executable),'parameters.prm']
            started = time.perf_counter()
            with (case/'run.log').open('w') as log:
                subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,
                               check=True,timeout=policy['timeout_per_process_s'])
            elapsed = time.perf_counter()-started
            check_solver_log(case/'run.log')
            if any(sha(Path(path)) != digest for path,digest in artifacts.items()):
                raise RuntimeError('Executable/library changed during campaign')
            initialization = json.loads((case/'output/initialization_statistics.json').read_text())
            if any(initialization[key] != 0 for key in ('ic_residual_calls','ic_jacobian_setups','ic_linear_solves')):
                raise RuntimeError('Native mode unexpectedly invoked IDACalcIC work')
            if not initialization['native_pair_verified'] or initialization['initial_state_source'] != 'steady network':
                raise RuntimeError('Initial-state policy differs from the declared verified 0D pair')
            if name.startswith('ramp') and (initialization['ramp_registrations'] != 1 or initialization['ramp_stop_arms'] != 1 or initialization['ramp_stop_releases'] != 1 or initialization['ramp_last_relative_factor'] != 1 or initialization['ramp_last_absolute_factor'] != 1):
                raise RuntimeError('Ramp did not reach exact final weights')
            with (case/'output/ida_statistics.csv').open(newline='') as stream:
                statistics_rows = list(csv.DictReader(stream))
            if len(statistics_rows) != 1:
                raise RuntimeError('IDA summary is incomplete')
            ida_statistics = {key:float(value) for key,value in statistics_rows[0].items()}
            if not all(math.isfinite(value) for value in ida_statistics.values()):
                raise RuntimeError('IDA summary contains nonfinite values')
            with (case/'output/ida_step_history.csv').open(newline='') as stream:
                history = list(csv.DictReader(stream))
            interval_fields = [key for key in history[0] if key.endswith('_interval')]
            through_ramp_end = {key:sum(float(row[key]) for row in history if float(row['output_time']) <= policy['ramp']['end_time']+1e-14) for key in interval_fields}
            receipt['records'].append({'case':name,'command':command,'process_wall_s':elapsed,
                                       'included_in_performance':not args.pilot and name!='reference','initialization':initialization,
                                       'ida_statistics':ida_statistics,'history_counters_through_ramp_end':through_ramp_end})
            protocol.write_text(json.dumps(receipt,indent=2)+'\n')
        validations = {name:validator.compare(output/'reference',output/name,policy,physics) for name in names if name!='reference'}
        validations.update({f'pair-{i}':validator.compare(output/f'uniform-{i}',output/f'ramp-{i}',policy,physics) for i in range(actual_pairs)})
        (output/'physical_equivalence.json').write_text(json.dumps(validations,indent=2,allow_nan=False)+'\n')
        times = {row['case']:row['process_wall_s'] for row in receipt['records']}
        integration = {row['case']:row['ida_statistics']['wall_time'] for row in receipt['records']}
        ratios = [times[f'uniform-{i}']/times[f'ramp-{i}'] for i in range(actual_pairs)]
        ida_ratios = [integration[f'uniform-{i}']/integration[f'ramp-{i}'] for i in range(actual_pairs)]
        report = {**acceptance_summary(validations, policy, actual_pairs),
                  'measurement_kind':measurement_kind,'actual_pair_count':actual_pairs,
                  'formal_pair_count':policy['runs_per_variant'],
                   'placement':placement,'library_directory':str(library_directory),
                   'physics_helper_sha256':sha(args.physics_helper),
                  'all_timings_excluded_from_formal_performance':args.pilot,
                  'all_pairs_retained':True,'reference_excluded_from_timing_comparison':True,
                  'physical_equivalence':'physical_equivalence.json','policy_sha256':receipt['policy_sha256']}
        if not args.pilot:
            report.update(median_paired_process_speedup=statistics.median(ratios),paired_process_speedups=ratios,
                          median_paired_ida_speedup=statistics.median(ida_ratios),paired_ida_speedups=ida_ratios,
                          median_process_performance_improved=statistics.median(ratios)>1)
        (output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        receipt.update(returncode=0 if report['accepted'] else 1)
        print(json.dumps(report))
    except BaseException as error:
        receipt.update(returncode=1,error=repr(error))
        raise
    finally:
        receipt['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        protocol.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt['returncode']


if __name__ == '__main__':
    raise SystemExit(main())
