// A linear radius has a quadratic rest area and constant rest pressure.
// Check that equilibrium and midpoint probes survive refinement and MPI.
#include <deal.II/base/mpi.h>

#include <metric_flow_x/blood_flow_system.h>

#include <cmath>
#include <filesystem>

#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

void
test()
{
  for (unsigned int refinement = 0; refinement < 3; ++refinement)
    {
      // ParameterHandler actions in deal.II 9.7 retain callbacks to the
      // previous instance. All acceptors from that iteration are destroyed.
      ParameterAcceptor::clear();
      BloodFlowSystem<1, 3> flow;
      flow.initialize_params(PRM_DIR "constant.prm");
      flow.fe_degree            = 2;
      flow.n_global_refinements = refinement;
      flow.output_directory     = "tapered_" + std::to_string(refinement);
      flow.setup();
      for (auto &[vid, vessel] : flow.vessel_map)
        {
          vessel.r_in  = 0.012;
          vessel.r_out = 0.006;
        }
      auto y = flow.make_state();
      // Test the seeded rest state before the terminal initialization solve
      // adjusts traces to the fixture's non-equilibrium outlet conditions.
      flow.compute_initial_solution(y, 0.0);
      auto residual = flow.make_state();
      flow.update_ghosted_vectors(y);
      flow.assemble_cell_residuals(0.0, flow.y_relevant, residual);
      residual.compress(VectorOperation::add);
      AssertThrow(residual.linfty_norm() < 1e-10, ExcInternalError());

      auto pressure = flow.make_state();
      flow.compute_pressure(y, pressure);
      for (const auto &cell : flow.dof_handler_.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            std::vector<types::global_dof_index> dofs(
              flow.fe_->n_dofs_per_cell());
            cell->get_dof_indices(dofs);
            const auto &vessel = flow.vessel_map.at(cell->material_id());
            for (unsigned int i = 0; i < dofs.size(); ++i)
              if (flow.fe_->system_to_component_index(i).first == 0)
                AssertThrow(std::abs(pressure(dofs[i]) - vessel.p0 -
                                     vessel.p_d) < 1e-8,
                            ExcInternalError());
          }
      flow.open_csv_files();
      unsigned int local_probes = 0;
      for (const auto &[vid, cell, xi] : flow.probe_targets)
        {
          const auto x =
            cell->vertex(0) + xi * (cell->vertex(1) - cell->vertex(0));
          AssertThrow(std::abs(x[0] - 0.24137 / 2.0) < 1e-14,
                      ExcInternalError());
          ++local_probes;
        }
      AssertThrow(Utilities::MPI::sum(local_probes, MPI_COMM_WORLD) == 1,
                  ExcInternalError());
      flow.close_csv_files();
    }
  deallog.depth_file(10);
  deallog << "Tapered rest equilibrium, nodal pressure and midpoint: PASSED"
          << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
