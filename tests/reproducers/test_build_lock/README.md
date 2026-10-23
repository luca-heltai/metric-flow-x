# Parallel CTest build lock reproducer

Run from the repository root with Python 3, CMake 3.23 or newer, and Ninja:

```bash
python3 tests/reproducers/test_build_lock/run.py
```

Use `--cmake`, `--ctest` and `--ninja` to specify tool paths if needed.
The script creates temporary builds and removes them afterward. It requires
neither deal.II nor a compiler and does not build or execute the solver.

The mock `deal_II_pickup_tests()` registers four CTest jobs. Each job starts
its own Ninja process in the same build directory, targeting a shared dummy
library writer. The writer appends to a shared file and records its execution
interval. The locked configuration includes the repository's actual
`tests/CMakeLists.txt`, so the test exercises the production resource-lock
assignment rather than copying it.

Both runs use `ctest -j4`. Expected and observed output:

```text
unlocked: 4/4 passed; maximum concurrent shared-output writers: 4
locked: 4/4 passed; maximum concurrent shared-output writers: 1
```

The script checks the CTest resource properties and measured writer overlap;
an unexpected result exits with an error. The unlocked check also confirms
that this environment schedules all four jobs concurrently.

This verifies serialization of nested Ninja invocations. Full parallel CTest
validation with the real solver and output comparisons remains pending.
The resource lock coordinates tests within one CTest invocation; separate
builds or CTest invocations still require external scheduling.
