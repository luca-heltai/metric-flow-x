PR: https://github.com/luca-heltai/metric-flow-x/pull/118

Boundary and junction conservation equations, RCR balances, terminal output, and mass diagnostics now use the same numerical area flux as the cell residuals. This includes the HLL-HDG stabilization contribution.

The junction mass assembly reads face-trace values from the cell DoF indices, so MPI ranks can assemble owned junction contributions without relying on a rank-local face cache. Regression tests now compare the RCR Jacobian rows against finite differences of the assembled residual and report the configured scaled trace-consistency tolerance.
