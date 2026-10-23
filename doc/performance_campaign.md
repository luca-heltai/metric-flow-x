# Measured performance campaign

The five-intervention campaign is tracked in
[issue #36](https://github.com/luca-heltai/metric-flow-x/issues/36), following the
assembly investigation in issue #9. Each accepted intervention has a separate
pull request based on the preceding intervention. Candidates must demonstrate
an improvement and preserve the measured numerical results.

## Measurement and numerical comparison

`benchmark_assembly` measures setup, residual assembly, the state Jacobian, and
the derivative Jacobian separately. Three timed samples follow a warm-up. Each
sample reports elapsed time per call on the slowest MPI rank. The benchmark
does not solve a circulation trajectory or include physical output costs.

An optional third argument supplies a snapshot filename prefix. After timing,
the benchmark assembles three nonuniform states at different times, including
reversed velocity, and writes every locally owned residual entry and every
stored entry of both Jacobian matrices as hexadecimal doubles. Comparisons use
the same MPI process count and numbering before and after an intervention.
Neither a norm nor a weighted checksum alone is the acceptance criterion.

`tools/compare_assembly.py` runs archived before/after executables and libraries
in alternating order with one OpenMP/OpenBLAS thread and MPI processes bound
to cores. It checks exact snapshot equality and records the hashes and sizes
before removing the large snapshot files. It preserves raw JSON timing records,
command lines, build/input hashes, host and CPU metadata, process logs, and SVG
and PNG comparison plots. Plot error bars show the range of independent process
medians, not a confidence interval. Timing is a local experiment, not a CI gate.

To archive a build, copy `benchmark_assembly` and `libmetric_flow_x.so` into a
directory and add a `revision.txt` describing the source revision and any
uncommitted source hash. Archive before rebuilding. `LD_LIBRARY_PATH` selects
the corresponding library; confirm the selection with `ldd` on Linux. Use the
same measurement driver in both archives.

```sh
python3 tools/prepare_assembly_benchmarks.py \
  --adan56-mesh notebooks/56_adnr_new.vtk \
  --network10k-mesh notebooks/networks_by_svv/trees/network_10000_adnr.vtk \
  --output-dir build/performance_campaign/inputs
python3 tools/compare_assembly.py \
  --before build/performance_campaign/baseline \
  --after build/performance_campaign/candidate1 \
  --inputs build/performance_campaign/inputs/adan56.prm \
           build/performance_campaign/inputs/network10k.prm \
  --ranks 1 2 --runs 5 --repetitions 20 \
  --output build/performance_campaign/step1
```

The reporting script requires matplotlib. The solver and its tests do not.
Keep working artifacts on persistent storage when execution may move between
nodes. Do not combine timings from different machines into a paired comparison.

Full trajectory regressions, finite-difference Jacobian tests, external-pressure
oracles, taper tests and MPI tests complement these synthetic snapshots. Exact
agreement on finite workloads does not establish all possible configurations
or physiological validity. Assembly speedups and end-to-end speedups must be
reported separately.
