# Dataset sources and generators

All solver-ready paper VTK inputs are available with
`scripts/download_datasets.sh [dataset-root]`. A normal solver run does not
need the generator dependencies.

| Network | Frozen source / generator | Distribution |
| --- | --- | --- |
| MMS single vessel | `2026_1d_blood_flow/convergence_results/mms_single_vessel.vtk` | `paper_inputs_v1.tar.gz` |
| Bifurcation and 37 segments | Manuscript VTK, also tracked under `notebooks/` | `paper_inputs_v1.tar.gz` |
| Liver, 7999 vessels | Manuscript `scaling_results_on_my_computer/8k_vessels/liver_vascular_tree.vtk`; OpenCCO origin described in manuscript | `paper_inputs_v1.tar.gz`; original OpenCCO generation recipe is not available |
| Cerebral, 19999 vessels | `notebooks/cerebral_data/cco_parietal_hc_w_data.vtk`; converter `notebooks/cerebral_data/convert_cco_to_stand_vtk.py` | `cerebral_datasets_v2.tar.gz` |
| Synthetic, 50001 and 100001 vessels | Trusted `.tree.npz` sources with recorded unit metadata; converter `notebooks/networks_by_svv/convert_to_standard_vtk.py` | `metric_flow_x_datasets_si_v1.tar.gz`; hashes/settings in `datasets/datasets_si_v1.json` |

The SVV generator `notebooks/networks_by_svv/complex_network.py` grows a tree
with external `svVascularize` and PyVista dependencies. It does not record a
seed or dependency lock, and is not an exact reconstruction of the published
geometries. Use the frozen NPZ source to regenerate an existing SI mesh:

```bash
python3 notebooks/networks_by_svv/convert_to_standard_vtk.py \
  notebooks/networks_by_svv/trees/network_50000.tree.npz \
  /tmp/network_100k.vtk
```

Use each manifest entry's `source` and `settings_SI`, not its filename, to
select the source geometry and conversion options. In particular the
`network_50000.tree.npz` source yields **100001 segments**, because a growing
binary tree's terminal count is different from its segment count. The
converter reads source length/time units from the archive or requires them
explicitly with `--length-unit` and `--time-unit`; it never infers them.
It requires NumPy and svVascularize; only load trusted NPZ files because the
upstream tree format stores Python objects. The raw VTK/VTP visualisation
exports in the network archive retain their source units and are not solver
inputs.

The cerebral converter has a separate CLI:

```bash
python3 notebooks/cerebral_data/convert_cco_to_stand_vtk.py --help
```

Record its outlet-pressure and terminal-pressure options explicitly when
creating a new dataset. A newly converted network is a new input: update its
manifest digest and release version before using it as a canonical case.

For all generators, the immutable downloaded VTK is the input for a recorded
run. Generator provenance alone is not evidence that the resulting simulation
matches a paper figure.
