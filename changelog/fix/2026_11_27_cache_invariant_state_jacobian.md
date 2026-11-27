PR: https://github.com/luca-heltai/metric-flow-x/pull/119

- Cache state-independent boundary, junction, and capacitor coefficients in a reduced-pattern global sparse matrix assembled once per mesh and DoF setup. Reuse it after Jacobian reassembly while retaining only global sparse entries.
