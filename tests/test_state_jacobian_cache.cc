// Retained state Jacobians remain immutable while new states are assembled.
#include <deal.II/base/mpi.h>

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

  auto state      = flow.make_state();
  auto derivative = flow.make_state();
  state           = 1.;
  derivative      = 0.;
  flow.assemble_state_jacobian(0., state, derivative);

  const auto old_owned   = flow.locally_owned_dofs();
  auto       old_owner   = flow.state_jacobian_matrix_owner();
  const auto old_entries = entries(*old_owner, old_owned);

  state = 2.;
  flow.assemble_state_jacobian(0., state, derivative);
  auto new_owner = flow.state_jacobian_matrix_owner();
  AssertThrow(old_owner.get() != new_owner.get(),
              ExcMessage(
                "A retained state Jacobian must detach on reassembly."));
  AssertThrow(entries(*old_owner, old_owned) == old_entries,
              ExcMessage("Reassembly changed a retained state Jacobian."));
  const unsigned int local_entries_changed =
    entries(*new_owner, old_owned) != old_entries ? 1u : 0u;
  AssertThrow(Utilities::MPI::max(local_entries_changed, MPI_COMM_WORLD) == 1,
              ExcMessage("Changing the state must reassemble its Jacobian."));

  old_owner.reset();
  new_owner.reset();
  const auto *current_owner = flow.state_jacobian_matrix_owner().get();
  flow.assemble_state_jacobian(0., state, derivative);
  AssertThrow(flow.state_jacobian_matrix_owner().get() == current_owner,
              ExcMessage("Unretained state Jacobians should reuse storage."));

  auto       setup_owner   = flow.state_jacobian_matrix_owner();
  const auto setup_owned   = flow.locally_owned_dofs();
  const auto setup_entries = entries(*setup_owner, setup_owned);
  flow.dof_handler_.clear();
  flow.triangulation().clear();
  flow.n_global_refinements = 1;
  flow.setup();
  AssertThrow(flow.state_jacobian_matrix_owner().get() != setup_owner.get(),
              ExcMessage("A retained state Jacobian must detach on setup."));
  AssertThrow(entries(*setup_owner, setup_owned) == setup_entries,
              ExcMessage(
                "Sparsity rebuild changed a retained state Jacobian."));

  deallog.depth_file(10);
  deallog << "State Jacobian cache ownership: PASSED" << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
