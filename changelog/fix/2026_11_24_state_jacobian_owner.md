PR: pending

- Expose an immutable owner for the assembled global state Jacobian. Detach before reassembly or sparsity rebuild when a solver snapshot retains the previous version, allowing coupled applications to avoid an extra full sparse-matrix clone.
