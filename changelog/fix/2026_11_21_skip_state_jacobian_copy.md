PR: https://github.com/luca-heltai/metric-flow-x/pull/110

- Avoid copying the full assembled state Jacobian on every refresh; append the derivative block directly to the matrix produced by state assembly.
