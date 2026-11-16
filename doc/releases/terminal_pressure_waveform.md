# Optional terminal reservoir waveform

Terminal RCR/resistive reservoirs can now use an embedded, sampled pressure
series in Pa. An empty series retains the constant `P_out` behavior. The CSV
reader reuses strict named-column validation and non-extrapolating interpolation.
Native checkpoints preserve samples and read old constant terminal archives.
This changes the public `RCRPhysics` layout, requiring matching application
headers and libraries.
