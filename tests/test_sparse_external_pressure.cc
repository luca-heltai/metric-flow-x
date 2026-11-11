// SPDX-License-Identifier: MIT
#include <metric_flow_x/blood_flow_system.h>

#include "pressure_jacobian_checks.h"
#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  for (const std::string fixture : {"constant", "multi_vessel"})
    for (const std::string flux : {"HLL", "LAX_FRIEDRICHS", "HLL_HDG"})
      {
        ParameterAcceptor::clear();
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          {
            std::ofstream prm("sparse_pressure.prm");
            prm << "include " << PRM_DIR << fixture << ".prm\n"
                << "subsection MetricFlowSystem<1, 3>\n"
                << "set Number of global refinement = 1\n"
                << "set Numerical flux type = " << flux << "\n"
                << "set Verbosity (console depth) = 0\nend\n";
          }
        MPI_Barrier(MPI_COMM_WORLD);
        BloodFlowSystem<1, 3> flow;
        flow.initialize_params("sparse_pressure.prm", "");
        flow.setup();
        auto state = flow.make_state();
        flow.compute_initial_solution(state, .375);
        for (const double speed : {.2, 1000.})
          {
            for (const auto component :
                 {BloodFlowSystem<1, 3>::Component::velocity,
                  BloodFlowSystem<1, 3>::Component::velocity_trace})
              for (const auto index : flow.component_dofs(component))
                if (flow.locally_owned_dofs().is_element(index))
                  state[index] = speed * (index % 3 == 0 ? -1. : 1.);
            state.compress(VectorOperation::insert);
            check_sparse_pressure_jacobian(flow, state);
          }
      }
  deallog.depth_file(10);
  deallog << "Sparse external-pressure columns and products: PASSED"
          << std::endl;
}
