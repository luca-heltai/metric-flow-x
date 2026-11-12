# Native checkpoint and restart

The transient standalone application can save complete native flow checkpoints
at its selected output times:

```sh
blood_flow input.prm --checkpoint-directory /path/to/history
blood_flow --restart /path/to/history/checkpoint-0x1.0624dd2f1a9fcp-8 \
  --final-time 0.02 --output-directory /path/to/restarted-output \
  --checkpoint-directory /path/to/continued-history
```

The parameter `Checkpoint directory` enables the same feature. Empty disables
it. Each checkpoint is a separate immutable directory; its hexadecimal floating
point name preserves the recorded time. Select a directory that actually exists
in the history rather than copying the illustrative name above. Restart needs
neither a parameter file nor the original mesh, coefficient CSVs or inflow CSV.
The complete history directory can be moved to another location.

Checkpoint output follows the selected fixed output schedule, including its
initial and final outputs. It also works when `Write solution output = false`.
When checkpoint output is enabled, the standalone runner arms native IDA stop
times at the output callbacks, coordinating with tolerance-ramp stops. It checks
that IDA's internal time matches the physical callback time, so checkpoints use
accepted endpoints instead of dense interpolants. It records all five residual
component norms in `checkpoint_residuals.csv` alongside the history. No extra
algebraic projection is applied before writing. The library save API preserves
whatever pair its caller supplies; coupled callers must enforce and verify their
own endpoint/constraint policy.

## Contents and native API

`BloodFlowSystem<1, 3>` exposes collective calls on its own communicator:

```cpp
flow.save_checkpoint(directory, time, state, derivative);

// Construct a fresh flow object, and register any application parameters first.
restored.load_checkpoint(directory);
// restored.state(), restored.state_derivative(), restored.get_time() are available.
```

Application parameters already registered by the caller are restored. A native
replay reader may omit unrelated standalone runner/output controls; undefined
entries are skipped using `ParameterHandler::parse_input`'s native facility.

The checkpoint contains:

- The resolved parameter tree, including parsed function expressions, constants,
  boundary policies, material policies and numerical settings.
- The fully distributed mesh and native finite element state using deal.II
  `Triangulation::save/load` and `SolutionTransfer` serialization. Mesh geometry,
  topology, material and boundary identifiers are restored; local CellId/active
  finite element associations are checked after rebuilding the discretization.
- Both native state and derivative vectors: cell area and velocity, area and
  velocity traces, and every capacitive RCR pressure. Data transfer uses the
  native cell/finite element association rather than assuming an implicit
  global DoF order.
- Resolved vessel, wall and terminal coefficients, terminal resistive links and
  sealed endpoints; tabulated inflow samples and periodicity are embedded.
- The exact physical checkpoint timestamp and compatibility metadata.

Original input paths can remain in the archived parameter text for provenance.
They are not opened during restore: geometry and resolved coefficients come
from the archives, and inflow samples are deserialized. The original configured
initial time is retained in `ida_parameters.initial_time`; `get_time()` returns
the saved physical time. The standalone runner uses the latter to reset IDA while
preserving the original output-grid and tolerance-ramp origins.

## Restart semantics and restrictions

The runner starts a new IDA integration from the restored pair. It does not
rerun the original initial-state construction or reset capacitor pressures.
IDA's internal BDF history, adaptive controller history and cumulative solver
statistics are not serialized. A restarted trajectory is therefore compared
numerically within tolerance, not required to match bit for bit. Output files
and statistics start a new segment; historical files are not appended.

The current format requires a shared filesystem, the same MPI rank count,
deal.II version and linear algebra backend. Mesh partitioning is recovered from the native mesh archive;
changing the process count or repartitioning at load is unsupported. Native
binary archives are intended for compatible builds, not long-term portable
exchange. The standalone checkpoint feature supports transient mode with one
refinement cycle.

The optional `prepare_additional_data(directory)` hook runs collectively after
native flow transfer registration and before `mesh.save()`. An owning coupled
application can register additional `SolutionTransfer` objects (for example
multiplier and reference area), keeping them alive until the call returns,
and write additional metadata into the supplied staging directory. The matching
`restore_additional_data(directory)` hook runs after native setup and flow
state deserialization; deserialize all extra objects in registration order.
A checkpoint written with an extension hook requires a restore hook. The final
manifest includes additional files, so the enclosing commit remains atomic.

A caller-supplied external-pressure `std::function` cannot be serialized and is
rejected by the native writer. A coupled application must archive its own
configuration, geometry, solid/multiplier states and feedback policy, and
reconstruct any pressure provider before advancing. Merely
saving the flow archive does not checkpoint an arbitrary coupled application.

Writing uses a sibling `.incomplete` staging directory. Payload streams close
on every rank before the `COMPLETE` manifest is written and the directory is
renamed to its final name. Existing destinations are rejected. An interrupted
staging directory is not a valid checkpoint; move or remove that incomplete
directory before retrying the same destination. The manifest checks file
presence and sizes, but is not a cryptographic integrity check or a guarantee
against storage-device power loss.

## Native deal.II MPI serialization dependency

The installed deal.II 9.9.0-pre revision `99dd5779c6bd882205f1a3110ded7d294dae7354`
contains an offset error in `parallel::fullydistributed::Triangulation::save/load`:
it converts the first-cell index to bytes although the attached-data serializer
already performs the byte conversion. With multiple ranks, variable-size vector
payloads overwrite their size table and loading can crash. The same two offset
conversions are present in the inspected newer source used by the coupled build.

The minimal [dependency patch](patches/dealii-fully-distributed-checkpoint-offsets.patch)
removes those two conversions. Apply it to a separate compatible deal.II checkout
and rebuild through that workspace's normal build entry point. No system or
shared installation is modified by this project. Existing checkpoints written
with the affected MPI serializer are corrupt and must be regenerated.

The serial continuation test passes on the original installed library. MPI
round-trip and continuation validation require the corrected native serializer;
results with the isolated corrected dependency are being collected. This is a
dependency defect, so the application continues to use the native API rather
than introducing an alternate vector format.

## Verification

The C++ `test_checkpoint` regression checks two distinct time-dependent native
pairs (including nonzero serialization sentinels in the second), all components and derivatives, mesh/material/boundary/DoF association,
embedded inflow and parsed functions, residual preservation, relocation after
deleting original inputs, and collective rejection of duplicate, missing and
incompatible checkpoints. It runs with one, two and four ranks.

The standalone integration regression runs a nonzero transient, removes the
original parameter, mesh and inflow files, relocates its checkpoints, and
starts a fresh process from an interior checkpoint. It compares continued
vessel and terminal observables with an uninterrupted reference using explicit
absolute/relative tolerances that allow IDA's restarted BDF adaptation:

```sh
python3 tests/test_checkpoint_restart.py --executable /path/to/blood_flow \
  --ranks 1 --ranks 2 --workdir /path/to/isolated-results
```

The C++ test is a serialization/equation identity test; the Python test is the
integration-continuation test. Neither establishes physical validation of a
particular vascular model or constraint-preserving interpolation between saved
states.
