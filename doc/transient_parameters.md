# Current transient parameter policy

All tracked `.prm` and `.prm.in` sources were reviewed against
`f922927df89e58a0bd316aca26d971f955d6c24e`, the head of PR #80 in
`metric-flow-x-adan-full`. These are inputs for the forthcoming paper runs,
including the Raksha/Müller experiment matrix. Earlier local measurements must
be regenerated; none is treated as a published result that freezes an input.

## Startup and IDA controls

Physical transients use native consistent initialization, componentwise absolute
error tolerances (area fraction `1e-6`, velocity `1e-6` m/s, capacitor pressure
`0.01` Pa), a maximum of 10,000 internal IDA steps per output, a `1e-12` s minimum
step, Newton convergence coefficient `1e-4`, and at least 20 nonlinear iterations.
The internal maximum step is `0.02` s (or the entire duration for shorter runs),
independent of the output callback period. Existing file and callback cadences
are retained. The relative tolerance retains each case's configured final value.
The scalar absolute tolerance remains in the input but componentwise weights
replace it when enabled; this changes the final error budget as well as startup.

The resistive graph supplies the initial state for RCR networks with positive
viscosity. Inviscid cases and verification cases start from rest; graph tube
resistances require strictly positive viscosity. The physical waveform, wall
model, spatial degree/refinement, flux, and existing BDF order are retained.
In particular, the paper's BDF 1–5 sweep remains a sweep. Uniformly forcing
BDF2 or changing the flux would change the experiment being measured.

Both tolerance multipliers start at 100. The loose hold lasts 30% of the run;
the geometric tightening ends at the last IDA output time at or before 60%.
If grid snapping leaves too short a tightening interval, the hold is reduced to
half the cutoff duration. The remaining trajectory uses final tolerances.
This generalizes the full ADAVN tutorial's three/six-cycle schedule for a
ten-cycle simulation. The ADAVN template and its current rerun copy retain
exactly the original 2.4/4.8 s schedule over 8 s.

For the three wall-model inputs, final time is 0.001 s and the IDA output period
is 0.005 s. There is no interior output time on which to end a ramp. They use
native initialization and component tolerances with the ramp explicitly off.

The paper MMS trajectory retains scalar tolerances throughout, with the ramp
explicitly off, so startup relaxation cannot contaminate the error being measured.
Its template keeps the requested BDF5 and `1e-12` scalar tolerances, with native
initialization, IDA work limits and explicit step controls.

## Removed legacy cases and schema cleanup

The sixteen historical ARKode parameter cases have been removed, together with
three tutorial-01 templates derived from the same obsolete verification setup.
Renaming their sections to IDA did not reproduce their former vessel constants,
initial-condition interface or manufactured fields. They are no longer offered
as runnable or pending-migration inputs. Their earlier versions remain in Git
history. Symbolic derivations and convergence reports remain reference artifacts.
Use `reproducibility/parameters/mms.prm.in` and the paper runner for the current
supported MMS experiment and degree/refinement sweep.

Old `BloodFlowSystem`/`Blood Flow Parameters` names are normalized. Superseded
inflow assignments are comments, preserving the previously effective last
waveform. The missing final `end` in `network_60k.prm` is repaired. Paths into a
sibling `blood-flow` checkout now use `@SOURCE_DIR@`; CMake expands both `.prm`
and `.prm.in` parameter sources. Where both forms target the same build file,
the source copies now agree with the canonical `.prm.in` (including the 15 s
bifurcation template). Dataset filenames and physical input values are retained.
This audit does not claim that every external dataset is already downloaded.

The two initialization benchmark inputs and
`tutorials/05_adavn_full/reference/executed_parameters.prm` are updated too.
Despite its historical filename, the latter is now a portable **current rerun
input**, not a receipt certifying an earlier execution. Substitute `@DATA_DIR@`
with a fresh directory prepared by tutorial 05. Existing numerical JSON/CSV and
plots describe earlier executions and must be regenerated before finalizing the
paper. No values have been invented to represent the updated inputs.

## Generated experiments and static verification

`scripts/reproduce_paper.py` applies the policy to all 43 manifest variants.
`--final-time` rescales hold/cutoff times, snaps the cutoff to the callback grid,
and disables relaxation when no interior cutoff exists. Its effective overrides
are recorded in the existing run receipt. The manifest records PR #80's source
baseline and the changed tolerance policy for each imported case.

`tools/benchmark_initialization.py` defaults to the two native variants using the
current executable. It updates existing initial-state settings instead of
inserting duplicates and fits the ramp to each requested horizon. An explicit
legacy variant remains available for independently supplied archival inputs and
an executable supporting their parameter schema.

Run the static inventory without launching simulations:

```sh
python3 scripts/audit_parameters.py > parameter-audit.json
```

The report lists every tracked parameter source, its policy and SHA-256. It
checks balanced sections, duplicate/unregistered settings, native initialization,
component budgets, graph applicability, step limits and ramp-grid alignment.
It also renders every paper variant at its configured duration and at 0.001,
0.1 and 0.5 s. These checks establish input coverage and consistency, not numerical
accuracy or speed. Simulations, comparisons and new performance measurements
are left to the author for the final rerun.

## Complete parameter-source inventory

The active input set contains 41 tracked parameter sources.

| Source | Policy |
|---|---|
| `doc/performance_data/initialization/adan56.prm` | transient; ramp enabled |
| `doc/performance_data/initialization/network10k.prm` | transient; ramp enabled |
| `parameters/aortic.prm.in` | transient; ramp enabled |
| `parameters/benchmark_parameters/37arteries_network/37_vessel_network.prm` | transient; ramp enabled |
| `parameters/benchmark_parameters/56_adnr/56_adnr.prm` | transient; ramp enabled |
| `parameters/benchmark_parameters/aortic_bifurcation/aortic_bifurcation.prm` | transient; ramp enabled |
| `parameters/benchmark_parameters/aortic_bifurcation/aortic_bifurcation.prm.in` | transient; ramp enabled |
| `parameters/benchmark_parameters/binary_tree_255/tree_255.prm` | transient; ramp enabled |
| `parameters/benchmark_parameters/common_carotid_artery/commom_carotid_artery.prm.in` | transient; ramp enabled |
| `parameters/benchmark_parameters/large_network_tree/cco_network.prm` | transient; ramp enabled |
| `parameters/benchmark_parameters/large_network_tree/network_tree.prm` | transient; ramp enabled |
| `parameters/benchmark_parameters/upper_thoracic_aorta/upper_thoracic_aorta.prm.in` | transient; ramp enabled |
| `parameters/constant.prm.in` | transient; ramp enabled |
| `parameters/multi_vessel.prm` | transient; ramp enabled |
| `parameters/multi_vessel.prm.in` | transient; ramp enabled |
| `parameters/networks_by_svv/network_100k.prm` | transient; ramp enabled |
| `parameters/networks_by_svv/network_10k.prm` | transient; ramp enabled |
| `parameters/networks_by_svv/network_200k.prm` | transient; ramp enabled |
| `parameters/networks_by_svv/network_20k.prm` | transient; ramp enabled |
| `parameters/networks_by_svv/network_30k.prm` | transient; ramp enabled |
| `parameters/networks_by_svv/network_40k.prm` | transient; ramp enabled |
| `parameters/networks_by_svv/network_50k.prm` | transient; ramp enabled |
| `parameters/networks_by_svv/network_60k.prm` | transient; ramp enabled |
| `parameters/stationary.prm.in` | stationary |
| `parameters/wall_arterial.prm.in` | short transient; uniform component tolerances |
| `parameters/wall_mixed.prm.in` | short transient; uniform component tolerances |
| `parameters/wall_venous.prm.in` | short transient; uniform component tolerances |
| `reproducibility/parameters/bifurcation.prm.in` | transient; ramp enabled |
| `reproducibility/parameters/cerebral.prm.in` | transient; ramp enabled |
| `reproducibility/parameters/liver.prm.in` | transient; ramp enabled |
| `reproducibility/parameters/mms.prm.in` | accuracy; uniform scalar tolerances |
| `reproducibility/parameters/network100k.prm.in` | transient; ramp enabled |
| `reproducibility/parameters/network37.prm.in` | transient; ramp enabled |
| `reproducibility/parameters/network50k.prm.in` | transient; ramp enabled |
| `tutorials/02_y_junction/parameters.prm.in` | transient; ramp enabled |
| `tutorials/03_37_arteries/parameters.prm.in` | transient; ramp enabled |
| `tutorials/04_adan56/parameters.prm.in` | transient; ramp enabled |
| `tutorials/05_adavn_full/assembly.prm.in` | assembly benchmark |
| `tutorials/05_adavn_full/hdg.prm.in` | transient; ramp enabled |
| `tutorials/05_adavn_full/mean.prm.in` | resistive mean driver |
| `tutorials/05_adavn_full/reference/executed_parameters.prm` | transient; ramp enabled |
