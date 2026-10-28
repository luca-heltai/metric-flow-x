"""Analytical unit/propagation checks and conservative rejection tests."""
import csv
import importlib.util
import math
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('gate',ROOT/'tools/compare_inflow_trajectories.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class InflowComparison(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.before,self.after=self.root/'before',self.root/'after'
        for folder in [self.before,self.after]:
            (folder/'output').mkdir(parents=True)
            mesh=folder/'single_vessel.vtk'
            shutil.copyfile(ROOT/'parameters/single_vessel.vtk',mesh)
            (folder/'last_used_parameters.prm').write_text('''subsection MetricFlowSystem<1, 3>
set Use componentwise absolute tolerances = false
set Inflow CSV file =
set Output directory = output
set Theta (penalty parameter) = 0.001
set Use direct solver = true
set Vtk file path for mesh input = '''+str(mesh)+'''
subsection IDA parameters
set Initial time = 0
set Final time = 0.001
set Time interval between each output = 0.001
subsection Error control
set Absolute error tolerance = 1e-12
set Relative error tolerance = 1e-7
end
end
end
subsection Metric Flow Parameters
set Density (rho) = 1060
set Viscosity coefficient (mu) = 0
end
subsection Functions
set Inflow function = t
end
''')
            self.write_outputs(folder)
    def tearDown(self):self.temp.cleanup()
    def write_csv(self,folder,name,fields,rows):
        with (folder/'output'/name).open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(fields);writer.writerows(rows)
    def write_outputs(self,folder,delta_a=0.,delta_u=0.):
        material=m.read_vtk(folder/'single_vessel.vtk')
        area=material['a_d']+delta_a;velocity=.01+delta_u
        pressure=material['p0']+material['p_d']+material['beta']/material['a_d']*(math.sqrt(area)-math.sqrt(material['a_d']))
        flow=area*velocity
        self.write_csv(folder,'HDG_IDA_Vessel_0.csv',['time_s','P_dynpcm2','Q_cm3ps','A_cm2','U_cmps','boundary_1_label'],[[t,pressure*10,flow*1e6,area*1e4,velocity*100,'boundary_1'] for t in [0,.001]])
        for boundary,sign in [(0,-1),(1,1)]:
            self.write_csv(folder,f'terminal_{boundary}.csv',['time_s','pressure_Pa','flow_m3_s','boundary_id','vessel_id','label'],[[t,pressure,sign*flow,boundary,0,f'boundary_{boundary}'] for t in [0,.001]])
        self.write_csv(folder,'network_terminals.csv',['boundary_id','vessel_id','label','R1','R2','C','P_out'],[[1,0,'boundary_1',11752000,111670000,1.0163e-8,0]])
        self.write_csv(folder,'mass_conservation.csv',['t','total_volume','dVdt','net_flux','imbalance','rel_imbalance'],[[t,material['L']*area,0,0,0,0] for t in [0,.001]])
        self.write_csv(folder,'ida_internal_steps.csv',['step','t','h','order'],[[1,.001,.001,1]])
        self.write_csv(folder,'ida_statistics.csv',['n_steps','wall_time'],[[1,.5]])
        self.write_csv(folder,'ida_step_history.csv',['output_time','ida_time','last_step_size'],[[0,0,.00001],[.001,.001,.001]])
    def edit(self,name,key,value,row=0,both=False):
        for folder in [self.before,self.after] if both else [self.after]:
            fields,rows=m.read_csv(folder/'output'/name)
            rows[row][key]=value
            self.write_csv(folder,name,fields,[[r[k] for k in fields] for r in rows])
    def test_exact_physical_outputs(self):
        r=m.compare_case(self.before,self.after)
        self.assertTrue(r['accepted'])
        self.assertTrue(all(v['max_normalized_error']==0 for v in r['metrics'].values()))
    def consolidate_vessels(self, folder):
        legacy = folder/'output/HDG_IDA_Vessel_0.csv'
        _, rows = m.read_csv(legacy)
        self.write_csv(folder, 'vessel_probes.csv',
            ['time_s','vessel_id','pressure_Pa','flow_m3_s','area_m2','velocity_m_s'],
            [[row['time_s'],0,float(row['P_dynpcm2'])/10,float(row['Q_cm3ps'])*1e-6,
              float(row['A_cm2'])*1e-4,float(row['U_cmps'])*.01] for row in rows])
        legacy.unlink()

    def test_consolidated_vessel_probes_match_historical_units(self):
        expected = m.csv_output_files(self.before/'output')
        self.consolidate_vessels(self.after)
        self.assertEqual(expected, m.csv_output_files(self.after/'output'))
        self.assertTrue(m.compare_case(self.before,self.after)['accepted'])
        self.consolidate(self.after)
        self.assertTrue(m.compare_case(self.before,self.after)['accepted'])

    def test_duplicate_consolidated_vessel_rows_rejected(self):
        self.consolidate_vessels(self.after)
        path = self.after/'output/vessel_probes.csv'
        lines = path.read_text().splitlines()
        path.write_text('\n'.join(lines+[lines[-1]])+'\n')
        with self.assertRaisesRegex(ValueError, 'vessel rows'):
            m.compare_case(self.before,self.after)

    def test_mixed_vessel_layout_rejected(self):
        self.consolidate_vessels(self.after)
        shutil.copyfile(self.before/'output/HDG_IDA_Vessel_0.csv', self.after/'output/HDG_IDA_Vessel_0.csv')
        with self.assertRaisesRegex(ValueError, 'Mixed'):
            m.compare_case(self.before,self.after)
    def consolidate(self, folder):
        fields, rows = m.read_csv(folder/'output/terminal_0.csv')
        _, outlet = m.read_csv(folder/'output/terminal_1.csv')
        rows += outlet
        rows.sort(key=lambda row: (float(row['time_s']), int(row['boundary_id'])))
        self.write_csv(folder, 'terminals.csv', fields, [[row[key] for key in fields] for row in rows])
        for bid in (0, 1):
            (folder/'output'/f'terminal_{bid}.csv').unlink()

    def test_configured_diagnostic_filenames(self):
        receipt = self.after/'last_used_parameters.prm'
        controls = []
        for key, default in m.CSV_DEFAULTS.items():
            old = self.after/'output'/default
            name = 'nested/' + default
            target = self.after/'output'/name
            target.parent.mkdir(exist_ok=True)
            if old.exists():
                old.rename(target)
            controls.append(f'set Output {key} CSV file = {name}')
        receipt.write_text(receipt.read_text().replace('set Output directory = output',
                           'set Output directory = output\n' + '\n'.join(controls)))
        self.assertTrue(m.compare_case(self.before, self.after)['accepted'])
        receipt.write_text(receipt.read_text().replace('set Output mass conservation CSV file = nested/mass_conservation.csv',
                                                      'set Output mass conservation CSV file ='))
        with self.assertRaises(ValueError):
            m.compare_case(self.before, self.after)

    def test_consolidated_and_historical_layouts_agree(self):
        old_files = m.csv_output_files(self.before/'output')
        self.consolidate(self.after)
        self.assertEqual(old_files, m.csv_output_files(self.after/'output'))
        self.assertTrue(m.compare_case(self.before, self.after)['accepted'])
        self.consolidate(self.before)
        self.assertTrue(m.compare_case(self.before, self.after)['accepted'])

    def test_configured_terminal_filename(self):
        self.consolidate(self.after)
        default = self.after/'output/terminals.csv'
        data = default.read_text()
        default.unlink()
        receipt = self.after/'last_used_parameters.prm'
        original = receipt.read_text()
        for filename in ('renamed.csv', 'sub/renamed.csv', str(self.root/'external.csv'), 'terminal_1.csv'):
            with self.subTest(filename=filename):
                receipt.write_text(original.replace('subsection MetricFlowSystem<1, 3>\n',
                    f'subsection MetricFlowSystem<1, 3>\nset Output terminal CSV file = {filename}\n'))
                path = self.after/'output'/filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(data)
                self.assertTrue(m.compare_case(self.before, self.after)['accepted'])
                path.unlink()

    def test_empty_terminal_filename_disables_output(self):
        receipt = self.after/'last_used_parameters.prm'
        receipt.write_text(receipt.read_text().replace('subsection MetricFlowSystem<1, 3>\n',
            'subsection MetricFlowSystem<1, 3>\nset Output terminal CSV file =\n'))
        with self.assertRaises(ValueError):
            m.compare_case(self.before, self.after)

    def test_mixed_terminal_layout_rejected(self):
        self.consolidate(self.after)
        (self.after/'output/terminal_0.csv').write_text('stale output')
        with self.assertRaises(ValueError):
            m.compare_case(self.before, self.after)

    def test_duplicate_consolidated_row_rejected(self):
        self.consolidate(self.after)
        fields, rows = m.read_csv(self.after/'output/terminals.csv')
        rows.insert(0, rows[0])
        self.write_csv(self.after, 'terminals.csv', fields, [[row[key] for key in fields] for row in rows])
        with self.assertRaises(ValueError):
            m.compare_case(self.before, self.after)

    def test_small_subsolver_state_and_derived_changes(self):
        self.write_outputs(self.after,delta_a=1e-13,delta_u=1e-14)
        r=m.compare_case(self.before,self.after)
        self.assertTrue(r['accepted'])
        self.assertGreater(r['metrics']['area']['max_normalized_error'],0)
        self.assertLess(max(v['max_normalized_error'] for v in r['metrics'].values()),1)
    def test_larger_than_state_tolerance_rejected(self):
        self.write_outputs(self.after,delta_a=1e-8)
        self.assertFalse(m.compare_case(self.before,self.after)['accepted'])
    def test_strict_normalized_one_rejected(self):
        metrics={};m.record_metric(metrics,'velocity',0.,1.,1.)
        self.assertFalse(metrics['velocity']['accepted'])
        self.assertEqual(metrics['velocity']['max_normalized_error'],1.)
    def test_pressure_mean_value_bound_analytical(self):
        self.assertEqual(m.pressure_bound(1.,4.,.01,{'beta':2.,'a_d':1.}),.01)
    def test_flow_identity_bound_analytical(self):
        self.assertAlmostEqual(m.flow_bound(2.,4.,5.,7.,.1,.2),1.2)
    def test_positive_pressure_branch_inverse(self):
        material={'a_d':4.,'beta':6.,'p0':10.,'p_d':20.}
        self.assertEqual(m.area_from_pressure(33.,material),16.)
        with self.assertRaisesRegex(ValueError,'branch'):m.area_from_pressure(20.,material)
    def test_actual_scalar_solver_weight(self):
        self.assertAlmostEqual(m.state_bound(2.,4.,.1,.01),.14)
    def test_diagnostics_can_change_and_are_recorded(self):
        self.edit('ida_internal_steps.csv','t','.0012')
        self.edit('ida_statistics.csv','n_steps','7')
        self.edit('ida_step_history.csv','ida_time','.0013',row=1)
        self.edit('mass_conservation.csv','rel_imbalance','1e-5')
        r=m.compare_case(self.before,self.after)
        self.assertTrue(r['accepted'])
        self.assertIn('t',r['diagnostics']['ida_internal_steps.csv']['fields'])
        self.assertIn('rel_imbalance',r['diagnostics'])
    def test_nonfinite_diagnostics_rejected_even_identical(self):
        self.edit('ida_internal_steps.csv','h','nan',both=True)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):m.compare_case(self.before,self.after)
    def test_changed_identifiers_or_labels_rejected(self):
        self.edit('terminal_0.csv','boundary_id','9')
        with self.assertRaisesRegex(ValueError,'identifiers'):m.compare_case(self.before,self.after)
    def test_changed_labels_rejected(self):
        self.edit('HDG_IDA_Vessel_0.csv','boundary_1_label','other')
        with self.assertRaisesRegex(ValueError,'labels'):m.compare_case(self.before,self.after)
    def test_fixed_physical_time_bound(self):
        self.edit('HDG_IDA_Vessel_0.csv','time_s','2e-14')
        with self.assertRaisesRegex(ValueError,'times'):m.compare_case(self.before,self.after)
    def test_componentwise_tolerances_rejected(self):
        p=self.after/'last_used_parameters.prm';p.write_text(p.read_text().replace('= false','= true'))
        with self.assertRaisesRegex(ValueError,'Componentwise'):m.compare_case(self.before,self.after)
    def test_solver_tolerance_changes_rejected(self):
        p=self.after/'last_used_parameters.prm';p.write_text(p.read_text().replace('= 1e-12','= 2e-12'))
        with self.assertRaisesRegex(ValueError,'Solver inputs'):m.compare_case(self.before,self.after)
    def test_unsupported_vtk_layout_rejected(self):
        p=self.after/'single_vessel.vtk';p.write_text(p.read_text().replace('CELLS 1 3','CELLS 2 6'))
        with self.assertRaises(ValueError):m.compare_case(self.before,self.after)
    def test_initial_volume_derivative_must_be_zero(self):
        self.edit('mass_conservation.csv','dVdt','1e-15')
        with self.assertRaisesRegex(ValueError,'Initial volume derivative'):m.compare_case(self.before,self.after)
    def test_missing_physical_rows_rejected(self):
        name='HDG_IDA_Vessel_0.csv';fields,_=m.read_csv(self.after/'output'/name)
        self.write_csv(self.after,name,fields,[])
        with self.assertRaisesRegex(ValueError,'row counts'):m.compare_case(self.before,self.after)
    def test_pure_unsupported_data_not_silently_ignored(self):
        self.write_csv(self.after,'extra_physical.csv',['x'],[[1]])
        with self.assertRaisesRegex(ValueError,'file set'):m.compare_case(self.before,self.after)

    def test_velocity_csv_centimeters_converted_to_solver_si_units(self):
        self.write_outputs(self.before,delta_u=-.01)
        self.write_outputs(self.after,delta_u=-.01+9e-13)
        result=m.compare_case(self.before,self.after)
        self.assertTrue(result['accepted'])
        self.assertAlmostEqual(result['metrics']['velocity']['max_normalized_error'],.9,delta=1e-5)

    def test_pressure_and_area_units_have_consistent_mean_value_ratio(self):
        self.write_outputs(self.after,delta_a=1e-13)
        metrics=m.compare_case(self.before,self.after)['metrics']
        self.assertAlmostEqual(metrics['pressure']['max_normalized_error']/metrics['area']['max_normalized_error'],1.,delta=1e-4)

    def test_capacitor_pressure_reconstruction_uses_outlet_resistance(self):
        self.write_outputs(self.after,delta_u=1e-8)
        metrics=m.compare_case(self.before,self.after)['metrics']
        expected=11752000*4.5239e-4*1e-8
        self.assertAlmostEqual(metrics['reconstructed_Pc']['max_abs_error'],expected,delta=64*math.ulp(9460.))

    def test_volume_derivative_uses_mass_units_not_direct_state_units(self):
        fields,rows=m.read_csv(self.after/'output/mass_conservation.csv')
        rows[1]['total_volume']=str(float(rows[1]['total_volume'])+1e-13)
        rows[1]['dVdt']='1e-10'
        rows[1]['imbalance']='1e-10'
        self.write_csv(self.after,'mass_conservation.csv',fields,[[r[k] for k in fields] for r in rows])
        result=m.compare_case(self.before,self.after)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['metrics']['dVdt']['max_abs_error'],1e-10)
        self.assertLess(result['metrics']['dVdt']['max_normalized_error'],1)

    def test_net_flux_uses_terminal_flow_units(self):
        self.edit('mass_conservation.csv','net_flux','1e-10',row=1)
        self.edit('mass_conservation.csv','imbalance','-1e-10',row=1)
        result=m.compare_case(self.before,self.after)
        self.assertFalse(result['accepted'])
        self.assertFalse(result['metrics']['net_flux']['accepted'])

    def test_nonfinite_execution_diagnostic_not_silently_dropped(self):
        self.edit('ida_statistics.csv','wall_time','inf',both=True)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):m.compare_case(self.before,self.after)

    def test_both_empty_physical_outputs_rejected(self):
        name='HDG_IDA_Vessel_0.csv'
        for folder in [self.before,self.after]:
            fields,_=m.read_csv(folder/'output'/name)
            self.write_csv(folder,name,fields,[])
        with self.assertRaisesRegex(ValueError,'Incomplete'):m.compare_case(self.before,self.after)


    def test_global_gate_rejects_extra_case_outputs(self):
        before,after=self.root/'all_before',self.root/'all_after'
        for case in m.CASES:
            shutil.copytree(self.before,before/case)
            shutil.copytree(self.after,after/case)
        self.assertTrue(m.compare(before,after)['accepted'])
        extra=after/'unexpected/output';extra.mkdir(parents=True)
        (extra/'extra.csv').write_text('x\n1\n')
        with self.assertRaisesRegex(ValueError,'unexpected'):m.compare(before,after)


    def test_nonfinite_derived_segment_length_rejected(self):
        p=self.after/'single_vessel.vtk'
        p.write_text(p.read_text().replace('0.24137 0.0 0.0','1.7e308 1.7e308 1.7e308'))
        with self.assertRaisesRegex(ValueError,'Nonfinite'):m.read_vtk(p)

    def test_underflowed_stiffness_rejected(self):
        p=self.after/'single_vessel.vtk'
        p.write_text(p.read_text().replace('4e5','1e-300').replace('1.2e-3','1e-300'))
        with self.assertRaisesRegex(ValueError,'stiffness'):m.read_vtk(p)


    def parameter(self, folder, key, value):
        path = folder/'last_used_parameters.prm'
        lines = path.read_text().splitlines()
        matches = [i for i, line in enumerate(lines) if line.startswith('set '+key+' =')]
        self.assertEqual(len(matches), 1)
        lines[matches[0]] = 'set '+key+' = '+value
        path.write_text('\n'.join(lines)+'\n')

    def test_entire_effective_parameter_map_is_preserved(self):
        values = m.read_parameters(self.before)['parameters']
        self.assertEqual(values[('Metric Flow Parameters', 'Density (rho)')], '1060')
        self.assertEqual(values[('Functions', 'Inflow function')], 't')

    def test_changed_physical_parameters_rejected_with_identical_outputs(self):
        changes = [('Density (rho)', '1061'), ('Viscosity coefficient (mu)', '0.001'),
                   ('Theta (penalty parameter)', '0.002'), ('Inflow function', '2*t'),
                   ('Use direct solver', 'false')]
        path = self.after/'last_used_parameters.prm'
        original = path.read_text()
        for key, value in changes:
            with self.subTest(parameter=key):
                path.write_text(original)
                self.parameter(self.after, key, value)
                with self.assertRaisesRegex(ValueError, 'effective physical parameters'):
                    m.compare_case(self.before, self.after)

    def test_different_output_and_verified_input_paths_accepted(self):
        for folder in (self.before, self.after):
            waveform = folder/'waveform.csv'
            waveform.write_bytes(b'time,flow\n0,0\n1,1\n')
            self.parameter(folder, 'Inflow CSV file', str(waveform))
        self.parameter(self.after, 'Output directory', '/different/output')
        self.assertTrue(m.compare_case(self.before, self.after)['accepted'])

    def test_changed_waveform_bytes_rejected_even_with_same_path_string(self):
        for folder in (self.before, self.after):
            self.parameter(folder, 'Inflow CSV file', 'waveform.csv')
            (folder/'waveform.csv').write_bytes(b'time,flow\n0,0\n1,1\n')
        (self.after/'waveform.csv').write_bytes(b'time,flow\n0,0\n1,2\n')
        with self.assertRaisesRegex(ValueError, 'waveform bytes'):
            m.compare_case(self.before, self.after)

    def test_empty_and_nonempty_waveform_rejected(self):
        (self.after/'waveform.csv').write_bytes(b'time,flow\n0,0\n1,1\n')
        self.parameter(self.after, 'Inflow CSV file', 'waveform.csv')
        with self.assertRaisesRegex(ValueError, 'waveform bytes'):
            m.compare_case(self.before, self.after)

    def test_nonpositive_total_volume_rejected_even_identical(self):
        for value in ('0', '-1e-6'):
            with self.subTest(volume=value):
                self.edit('mass_conservation.csv', 'total_volume', value, both=True)
                with self.assertRaisesRegex(ValueError, 'Nonpositive total volume'):
                    m.compare_case(self.before, self.after)


if __name__=='__main__':unittest.main(verbosity=2)
