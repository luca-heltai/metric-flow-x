#include <deal.II/base/mpi.h>

#include <metric_flow_x/blood_flow_system.h>

#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

void
test()
{
  BloodFlowSystem<1, 3> flow;
  flow.initialize_params(PRM_DIR "multi_vessel.prm");
  flow.use_direct_solver = false;
  flow.setup();
  auto state = flow.make_state();
  flow.initialize_state(state, 0.0);
  AssertThrow(flow.initial_state_diagnostics().algebraic_defect <=
                flow.initial_consistency_tolerance,
              ExcMessage(
                "Iterative initialization must satisfy the configured trace tolerance."));
  deallog.depth_file(10);
  deallog << "Iterative trace initialization with nonzero inflow: PASSED"
          << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
