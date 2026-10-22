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
  flow.update_ghosted_vectors(state);
  auto residual = flow.make_state();
  residual      = 0.0;
  flow.assemble_trace_interior_equations(flow.y_relevant, residual);
  flow.assemble_trace_boundary_equations(0.0, flow.y_relevant, residual);
  flow.assemble_trace_junction_equations(flow.y_relevant, residual);
  flow.assemble_trace_continuity_equations(flow.y_relevant, residual);
  residual.compress(VectorOperation::add);
  double local = 0.0;
  for (const auto i : flow.trace_dofs_owned)
    local += residual(i) * residual(i);
  AssertThrow(
    std::sqrt(Utilities::MPI::sum(local, flow.mpi_communicator())) < 1e-10,
    ExcMessage("Iterative initialization must satisfy trace equations."));
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
