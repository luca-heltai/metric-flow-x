// Repeated derivative Jacobians remain independent of state. Rebuilding the
// geometry, mass matrices, or terminal data must refresh the cached matrix.
#include <deal.II/base/mpi.h>

#include <deal.II/grid/grid_tools.h>

#include <metric_flow_x/blood_flow_system.h>

#include <map>

#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

using Entries =
  std::map<std::pair<types::global_dof_index, types::global_dof_index>, double>;

Entries
entries(const MatrixType &matrix, const IndexSet &owned)
{
  Entries result;
  for (const auto row : owned)
    for (auto entry = matrix.begin(row); entry != matrix.end(row); ++entry)
      result[{row, entry->column()}] = entry->value();
  return result;
}

void
test()
{
  BloodFlowSystem<1, 3> flow;
  flow.initialize_params(PRM_DIR "constant.prm");
  flow.terminal_parameter_source = "Parameters";
  flow.default_terminal_c        = 1e-8;
  flow.n_global_refinements      = 0;
  flow.setup();
  for (unsigned int cycle = 0; cycle < 2; ++cycle)
    {
      auto state      = flow.make_state();
      auto derivative = flow.make_state();
      state           = 0.;
      derivative      = 0.;
      flow.assemble_derivative_jacobian(0., state, derivative);
      const auto original =
        entries(flow.derivative_jacobian_matrix(), flow.locally_owned_dofs());
      for (const auto &[boundary, index] : flow.rcr_pc_dof)
        if (flow.locally_owned_dofs().is_element(index))
          AssertThrow(original.at({index, index}) == flow.default_terminal_c,
                      ExcMessage("Terminal capacitance must be refreshed."));
      state      = 2.;
      derivative = -3.;
      flow.assemble_derivative_jacobian(1., state, derivative);
      AssertThrow(entries(flow.derivative_jacobian_matrix(),
                          flow.locally_owned_dofs()) == original,
                  ExcMessage("Derivative Jacobian must be state independent."));

      GridTools::scale(2., flow.triangulation());
      flow.build_per_cell_mass_inv();
      flow.assemble_derivative_jacobian(2., state, derivative);
      const auto scaled =
        entries(flow.derivative_jacobian_matrix(), flow.locally_owned_dofs());
      for (const auto &[index, value] : original)
        {
          const double expected =
            index.first < flow.dof_handler().n_dofs() ? 2 * value : value;
          AssertThrow(std::abs(scaled.at(index) - expected) <=
                        1e-14 * std::abs(expected),
                      ExcMessage("Mass rebuild must refresh geometry values."));
        }

      if (cycle == 0)
        {
          flow.dof_handler_.clear();
          flow.triangulation().clear();
          flow.n_global_refinements = 2;
          flow.default_terminal_c   = 2e-8;
          flow.setup();
        }
    }
  deallog.depth_file(10);
  deallog << "Derivative Jacobian cache refresh: PASSED" << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
