#include <deal.II/base/mpi.h>

#include <metric_flow_x/blood_flow_system.h>

#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

void
test()
{
  BloodFlowSystem<1, 3> flow;
  flow.initialize_params(PRM_DIR "constant.prm");
  flow.setup();

  auto y     = flow.make_state();
  auto y_dot = flow.make_state();
  auto F     = flow.make_state();

  flow.initialize_state(y, 0.0);
  flow.initialize_state_derivative(y_dot, 0.0);
  flow.assemble_residual(0.0, y, y_dot, F);
  flow.assemble_state_jacobian(0.0, y, y_dot);
  flow.assemble_derivative_jacobian(0.0, y, y_dot);

  AssertThrow(flow.triangulation().n_active_cells() > 0, ExcInternalError());
  AssertThrow(flow.dof_handler().n_dofs() > 0, ExcInternalError());
  AssertThrow(flow.finite_element().n_components() == 4, ExcInternalError());
  AssertThrow(flow.differential_dofs().n_elements() > 0, ExcInternalError());
  AssertThrow(flow.algebraic_dofs().n_elements() > 0, ExcInternalError());
  AssertThrow(flow.component_dofs(BloodFlowSystem<1, 3>::Component::area)
                  .n_elements() > 0,
              ExcInternalError());
  AssertThrow(flow.component_dofs(
                    BloodFlowSystem<1, 3>::Component::velocity_trace)
                  .n_elements() > 0,
              ExcInternalError());
  AssertThrow(flow.state_jacobian_matrix().m() ==
                flow.locally_owned_dofs().n_elements(),
              ExcInternalError());
  AssertThrow(flow.derivative_jacobian_matrix().m() ==
                flow.locally_owned_dofs().n_elements(),
              ExcInternalError());

  const auto &vessel = flow.vessel_properties(0);
  AssertThrow(flow.pressure(vessel.a_d, 0) == flow.pressure(vessel.a_d, 0),
              ExcInternalError());
  AssertThrow(flow.pressure_derivative(vessel.a_d, 0) > 0.0,
              ExcInternalError());
  AssertThrow(flow.wave_speed(vessel.a_d, 0) > 0.0, ExcInternalError());

  deallog << "MetricFlowX coupling readiness: PASSED" << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  initlog();
  test();
}
