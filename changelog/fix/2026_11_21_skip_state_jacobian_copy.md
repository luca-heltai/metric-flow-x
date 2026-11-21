PR: pending

- Avoid copying the full assembled state Jacobian on every refresh; append the derivative block directly to the matrix produced by state assembly.
