# Testing

Tests are configured by the top-level CMake project and `tests/CMakeLists.txt`. The test directory includes `DEAL_II_PICKUP_TESTS()`, which discovers deal.II test sources and associates their expected-output files with generated build-tree test executables.

## Configure and run

Configure and build first:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

CTest must be pointed at the build tree: the test executables and generated parameter copies are not in the source tree.

