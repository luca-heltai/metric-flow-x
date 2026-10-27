#!/usr/bin/env python3
"""Audit every tracked parameter source and paper variant without running a solver."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from reproduce_paper import replace_setting, smoke_ramp_settings

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'f922927df89e58a0bd316aca26d971f955d6c24e'
SYSTEM = 'MetricFlowSystem<1, 3>'
IDA_KEYS = {
    'Initial time', 'Final time', 'Time interval between each output',
    'Absolute error tolerance', 'Relative error tolerance', 'Initial step size',
    'Maximum number of nonlinear iterations', 'Maximum order of BDF',
    'Minimum step size', 'Maximum step size', 'Nonlinear convergence coefficient',
}


def parse(text):
    """Retain subsection paths so duplicate or misplaced settings cannot hide."""
    stack, result = [], {}
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith('subsection '):
            stack.append(line.removeprefix('subsection '))
        elif line == 'end':
            assert stack, 'Unmatched end'
            stack.pop()
        else:
            match = re.fullmatch(r'set (.+?)\s*=\s*(.*)', line)
            assert match, f'Invalid parameter line: {line}'
            key = (*stack, match[1].strip())
            assert key not in result, f'Duplicate setting: {key}'
            result[key] = match[2].strip()
    assert not stack, f'Unclosed subsections: {stack}'
    return result


def check_transient(entries):
    d = {k[-1]: v for k, v in entries.items() if k[0] == SYSTEM}
    assert d['Initial condition policy'] == 'native'
    assert int(d['Maximum IDA internal steps per output']) == 10000
    assert d['Initial state source'] in ('rest', 'steady network')
    assert d['Use componentwise absolute tolerances'] in ('true', 'false')
    initial, final = float(d['Initial time']), float(d['Final time'])
    period = float(d['Time interval between each output'])
    minimum, maximum = float(d['Minimum step size']), float(d['Maximum step size'])
    assert final > initial and period > 0 and 0 <= minimum <= maximum
    assert minimum <= float(d['Initial step size']) <= maximum
    if d['Use componentwise absolute tolerances'] == 'true':
        assert float(d['Area absolute tolerance fraction']) == 1e-6
        assert float(d['Velocity absolute tolerance']) == 1e-6
        assert float(d['Capacitor pressure absolute tolerance']) == .01
        assert float(d['Nonlinear convergence coefficient']) == 1e-4
    if d['Initial state source'] == 'steady network':
        assert d['Outlet boundary condition type'] == 'RCR'
        mu = entries.get(('Metric Flow Parameters', 'Viscosity coefficient (mu)'), '.004')
        assert float(mu) > 0
    if d['Tolerance ramp enabled'] == 'true':
        hold, end = float(d['Tolerance ramp hold duration']), float(d['Tolerance ramp end time'])
        assert 0 <= hold < end - initial and end < final
        steps = (end - initial) / period
        assert round(steps) >= 1 and math.isclose(steps, round(steps), abs_tol=1e-9)
        assert float(d['Tolerance ramp initial relative factor']) == 100
        assert float(d['Tolerance ramp initial absolute factor']) == 100
    else:
        assert d['Tolerance ramp enabled'] == 'false'


def main():
    known = set()
    for source in ('source/blood_flow_system.cc', 'apps/blood_flow.cc',
                   'include/metric_flow_x/tolerance_ramp.h', 'tools/adan/mean_flow.cc'):
        known.update(re.findall(r'add_parameter\(\s*"([^"]+)"', (ROOT / source).read_text()))
    known |= IDA_KEYS
    library_keys = set(re.findall(
        r'add_parameter\(\s*"([^"]+)"',
        (ROOT / 'source/blood_flow_system.cc').read_text()))
    library_keys |= IDA_KEYS - {'Maximum step size', 'Nonlinear convergence coefficient'}
    physical = {'Density (rho)', 'Viscosity coefficient (mu)',
                'Profile constant for friction term (xi)', 'Tube law exponent (m)',
                'Reflection coefficient at outflow boundary (Rt)'}
    functions = {'Exact solution', 'Inflow function', 'RHS expression'}
    files = subprocess.check_output(
        ['git', 'ls-files', '*.prm', '*.prm.in'], cwd=ROOT, text=True).splitlines()
    inventory = []
    for name in files:
        text = (ROOT / name).read_text()
        try:
            entries = parse(text)
            if name.startswith('parameters/') and name.endswith('.prm.in'):
                twin = (ROOT / name).with_suffix('')
                if twin.exists():
                    assert twin.read_text() == text, 'Conflicting sources for one CMake destination'
            for key in entries:
                if key[0] == 'Metric Flow Parameters':
                    assert len(key) == 2 and key[-1] in physical, key
                elif key[0] == 'Functions':
                    assert len(key) == 2 and key[-1] in functions, key
                else:
                    assert key[-1] in known, f'Unregistered setting: {key}'
                    assert len(key) == 1 or key[0] == SYSTEM, key
            if name.startswith('tests/parameters/'):
                for key in entries:
                    if key[0] == SYSTEM:
                        assert key[-1] in library_keys, f'Runner setting in library fixture: {key}'
                category = 'library regression fixture'
            elif name.endswith('/mean.prm.in'):
                category = 'resistive mean driver'
            elif name.endswith('/assembly.prm.in'):
                category = 'assembly benchmark'
            elif entries.get((SYSTEM, 'Run mode'), 'transient').startswith('steady'):
                category = 'stationary'
            else:
                if '@RUN_MODE@' in text:
                    text = text.replace('@RUN_MODE@', 'transient').replace('@FINAL_TIME@', '8')
                    text = text.replace('@INITIAL_SOURCE@', 'steady network')
                    entries = parse(text)
                check_transient(entries)
                if entries[(SYSTEM, 'Use componentwise absolute tolerances')] == 'false':
                    category = 'accuracy; uniform scalar tolerances'
                elif entries[(SYSTEM, 'Tolerance ramp enabled')] == 'true':
                    category = 'transient; ramp enabled'
                else:
                    category = 'short transient; uniform component tolerances'
        except (AssertionError, KeyError, ValueError) as error:
            raise ValueError(f'{name}: {error}') from error
        inventory.append({'path': name, 'policy': category,
                          'sha256': hashlib.sha256((ROOT / name).read_bytes()).hexdigest()})

    manifest = json.loads((ROOT / 'reproducibility/manifest.json').read_text())
    assert manifest['code_baseline'] == BASELINE
    for case in manifest['runs']:
        template = (ROOT / f"reproducibility/parameters/{case['base']}.prm.in").read_text()
        for final in (None, .001, .1, .5):
            text = template
            settings = dict(case['settings'])
            if final is not None:
                settings.update(smoke_ramp_settings(text, final))
                settings['Final time'] = str(final)
            for key, value in settings.items():
                text = replace_setting(text, key, value)
            check_transient(parse(text))
    print(json.dumps({'baseline': BASELINE, 'source_count': len(files),
                      'paper_variants': len(manifest['runs']),
                      'paper_horizons_checked': ['configured', .001, .1, .5],
                      'categories': dict(Counter(row['policy'] for row in inventory)),
                      'files': inventory}, indent=2))


if __name__ == '__main__':
    main()
