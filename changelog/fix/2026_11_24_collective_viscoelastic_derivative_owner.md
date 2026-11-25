PR: #116
- Keep retained viscoelastic derivative-Jacobian snapshots immutable across reassembly and sparsity rebuilds, with collective copy-on-write detachment when any MPI rank retains an owner.
