PR: pending
- Allocate the global native linear solve matrix only when a native solve needs it, avoiding an unused Jacobian-pattern matrix in coupled ImmersX runs.
