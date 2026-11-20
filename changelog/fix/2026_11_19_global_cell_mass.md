PR: #109

- Replace persistent dense per-cell mass storage with a global sparse cell-mass operator for residual and derivative-Jacobian assembly. Keep residual updates local after the matrix action so the result does not depend on backend-specific compression modes. Construct the sparse inverse of differential cell blocks lazily within coherent initialization and release it on return or exception; checkpoint loading does not allocate it, and repeated initialization remains supported.
