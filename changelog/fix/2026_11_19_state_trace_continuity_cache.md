PR: #108

- Assemble the state-transmission trace-continuity block once per mesh/DoF setup in a global sparse matrix, reusing it in repeated Jacobians while retaining state-dependent flux transmission and runtime mode switching.
