// Independently check the rigid single-tube seed, RCR equilibrium, tube-law
// inversion with external pressure, and the complete native consistent pair.
#include <deal.II/base/mpi.h>

#include <metric_flow_x/blood_flow_system.h>
#include <metric_flow_x/steady_network.h>

#include <cmath>
#include <fstream>
#include <iomanip>

#include "tests.h"
using namespace dealii;
using namespace MetricFlowX;

void
test()
{
  const MPI_Comm comm = MPI_COMM_WORLD;
  for (const unsigned int degree : {1u, 2u})
    for (const double compliance : {0., 1.e-8})
      {
        if (Utilities::MPI::this_mpi_process(comm) == 0)
          {
            std::ofstream file("consistent_initialization.prm");
            file
              << std::setprecision(17)
              << "subsection Metric Flow Parameters\n set Viscosity coefficient (mu) = 0.004\n set Profile constant for friction term (xi) = 2\nend\n"
              << "subsection Functions\n set Inflow function = 1e-6\nend\n"
              << "subsection MetricFlowSystem<1, 3>\n set Vtk file path for mesh input = "
              << METRIC_FLOW_X_SOURCE_DIR << "/parameters/single_vessel.vtk\n"
              << " set Finite element degree = " << degree
              << "\n set Number of global refinement = 0\n"
              << " set Outlet boundary condition type = RCR\n set Terminal parameter source = Parameters\n"
              << " set Default terminal R1 = 1e7\n set Default terminal R2 = 1e8\n"
              << " set Default terminal C = " << compliance
              << "\n set Default terminal P_out = 1000\n"
              << " set Initial state source = steady network\n set Use componentwise absolute tolerances = true\nend\n";
          }
        MPI_Barrier(comm);
        ParameterAcceptor::clear();
        BloodFlowSystem<1, 3> flow(comm);
        flow.initialize_params("consistent_initialization.prm");
        const double t0 = .17;
        flow.set_external_pressure_provider(
          [](const auto &point) { return 200. + 10. * point.time; });
        for (const unsigned int refinement : {0u, 2u, 1u})
          {
            flow.dof_handler_.clear();
            flow.triangulation_.clear();
            flow.n_global_refinements = refinement;
            flow.setup();
            auto state = flow.make_state(), derivative = flow.make_state(),
                 residual = flow.make_state();
            flow.compute_initial_solution(state, t0);
            const auto  &v      = flow.vessel_map.at(0);
            const double r      = std::sqrt(v.a_d / numbers::PI);
            const double length = .24137;
            const double resistance =
              8. * .004 * length / (numbers::PI * std::pow(r, 4));
            std::vector<types::global_dof_index> dofs(
              flow.fe_->n_dofs_per_cell());
            for (const auto &cell : flow.dof_handler_.active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  cell->get_dof_indices(dofs);
                  for (unsigned int i = 0; i < dofs.size(); ++i)
                    if (flow.fe_->system_to_component_index(i).first == 0)
                      {
                        const auto   point = flow.support_point(cell, i);
                        const double expected =
                          1110. + resistance * 1e-6 * (1. - point[0] / length);
                        const double p = flow.compute_physical_pressure(
                          state(dofs[i]), 0, v.a_d, t0, point, cell->id());
                        AssertThrow(
                          std::abs(p - expected) < 1.e-8,
                          ExcMessage(
                            "Rigid graph pressure or compliant/external-pressure reconstruction is wrong."));
                        const auto j = flow.fe_->component_to_system_index(
                          1, flow.fe_->system_to_component_index(i).second);
                        AssertThrow(
                          std::abs(state(dofs[i]) * state(dofs[j]) - 1.e-6) <
                            1.e-18,
                          ExcMessage(
                            "The nodal cell seed must conserve graph flow."));
                      }
                }
            for (const auto &[bid, index] : flow.rcr_pc_dof)
              if (flow.locally_owned_dofs_.is_element(index))
                AssertThrow(std::abs(state(index) - 1100.) < 1.e-9,
                            ExcMessage("Pc must start at P_out+R2*Q."));
            AssertThrow(flow.n_rcr_dofs == (compliance > 0. ? 1u : 0u),
                        ExcInternalError());
            flow.initialize_consistent_state(state, derivative, t0);
            const auto &diagnostics = flow.initial_state_diagnostics();
            AssertThrow(
              diagnostics.algebraic_defect <=
                  flow.initial_consistency_tolerance &&
                diagnostics.differential_defect <=
                  flow.initial_consistency_tolerance,
              ExcMessage(
                "The full pair must satisfy native consistency bounds."));
            flow.assemble_residual(t0, state, derivative, residual);
            AssertThrow(
              std::isfinite(residual.l2_norm()) &&
                derivative.linfty_norm() > 0.,
              ExcMessage(
                "The compliant graph warm start needs finite native residuals and nonzero differential derivatives."));
            AssertThrow(
              diagnostics.largest_trace_block <= 4,
              ExcMessage(
                "A single tube must use independent small trace blocks."));
          }
      }
  deallog.depth_file(10);
  deallog
    << "Rigid seed and native consistent pair with RCR, external pressure, refinement and polynomial degrees: PASSED"
    << std::endl;
}
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  mpi_initlog();
  test();
}
