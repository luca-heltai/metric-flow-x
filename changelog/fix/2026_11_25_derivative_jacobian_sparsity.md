PR: https://github.com/luca-heltai/metric-flow-x/pull/117

Reduce the cached derivative-Jacobian sparsity pattern to the globally assembled cell-mass and terminal-capacitance terms when viscoelastic walls are disabled. This avoids reserving the full nonlinear state-Jacobian structure while retaining the existing full pattern for viscoelastic contributions.
