PR: #109

- Replace persistent dense per-cell mass storage with a global sparse cell-mass operator for residual and derivative-Jacobian assembly. Accumulate residual terms in an owned-row buffer and compress the output once, avoiding backend-specific matrix/vector assembly-state errors. Construct the sparse inverse of differential cell blocks lazily within coherent initialization and release it on return or exception; checkpoint loading does not allocate it, and repeated initialization remains supported.
