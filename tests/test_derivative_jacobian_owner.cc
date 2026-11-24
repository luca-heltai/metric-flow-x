// SPDX-License-Identifier: MIT
// Retained viscoelastic derivative Jacobians remain immutable across
// reassembly and sparsity rebuilds, including when one MPI rank retains a copy.
#include <metric_flow_x/blood_flow_system.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>

#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

namespace
{
  using EntryKey = std::pair<types::global_dof_index, types::global_dof_index>;
  using Entries  = std::map<EntryKey, double>;

  Entries
  snapshot_entries(const MatrixType &matrix, const IndexSet &owned)
  {
    Entries entries;
    for (const auto row : owned)
      for (auto entry = matrix.begin(row); entry != matrix.end(row); ++entry)
        entries[{row, entry->column()}] = entry->value();
    return entries;
  }

  double
  maximum_difference(const Entries &left, const Entries &right)
  {
    AssertThrow(left.size() == right.size(), ExcInternalError());
    double difference = 0.;
    auto   l          = left.begin();
    auto   r          = right.begin();
    for (; l != left.end(); ++l, ++r)
      {
        AssertThrow(l->first == r->first, ExcInternalError());
        difference = std::max(difference, std::abs(l->second - r->second));
      }
    return difference;
  }
} // namespace

void
test()
{
  const auto comm = MPI_COMM_WORLD;
  ParameterAcceptor::clear();
  BloodFlowSystem<1, 3> flow(comm);
  flow.initialize_params(PRM_DIR "wall_arterial.prm");
  flow.setup();
  AssertThrow(flow.viscoelastic_wall, ExcInternalError());

  for (auto &[id, vessel] : flow.vessel_map)
    {
      (void)id;
      vessel.wall.km = 2.;
    }

  auto state = flow.make_state();
  auto rate  = flow.make_state();
  flow.compute_initial_solution(state, 0.);
  flow.initialize_trace_unknowns(state, 0.);
  rate = 0.;
  rate.compress(VectorOperation::insert);

  flow.assemble_derivative_jacobian(0., state, rate);
  const auto before_reassembly =
    snapshot_entries(flow.derivative_jacobian_matrix(),
                     flow.locally_owned_dofs_);
  auto previous_owner = flow.derivative_jacobian_matrix_owner();
  std::weak_ptr<const MatrixType> previous_owner_observer = previous_owner;

  // Retain the previous distributed matrix only on rank zero. Every rank must
  // detach its current owner before the next reassembly, or the old version on
  // rank zero would be mixed with newly assembled rows on other ranks.
  std::shared_ptr<const MatrixType> retained_owner;
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    retained_owner = previous_owner;
  previous_owner.reset();

  auto changed_state = state;
  for (const auto index : flow.locally_owned_dofs_)
    if (flow.component_dofs_[0].is_element(index))
      changed_state(index) *= 1.4;
  changed_state.compress(VectorOperation::insert);

  flow.assemble_derivative_jacobian(0., changed_state, rate);
  auto current_owner = flow.derivative_jacobian_matrix_owner();
  AssertThrow(
    Utilities::MPI::max(
      static_cast<unsigned int>(
        maximum_difference(
          before_reassembly,
          snapshot_entries(*current_owner, flow.locally_owned_dofs_)) > 1.e-12),
      comm) == 1,
    ExcMessage("Viscoelastic derivative Jacobian did not change with state."));

  unsigned int retained_unchanged = 1;
  const bool   is_root            = Utilities::MPI::this_mpi_process(comm) == 0;
  if (retained_owner)
    retained_unchanged =
      maximum_difference(before_reassembly,
                         snapshot_entries(*retained_owner,
                                          flow.locally_owned_dofs_)) == 0. ?
        1u :
        0u;
  const unsigned int observer_lifetime_matches_expectation =
    previous_owner_observer.expired() != is_root ? 1u : 0u;
  AssertThrow(Utilities::MPI::min(retained_unchanged, comm) == 1,
              ExcMessage(
                "Derivative reassembly changed a retained owner snapshot."));
  AssertThrow(
    Utilities::MPI::min(observer_lifetime_matches_expectation, comm) == 1,
    ExcMessage(
      "Derivative reassembly did not detach the matrix on every MPI rank."));
  unsigned int detached = 1;
  if (retained_owner)
    detached = retained_owner.get() != current_owner.get() ? 1u : 0u;
  AssertThrow(Utilities::MPI::min(detached, comm) == 1,
              ExcMessage(
                "Derivative reassembly did not detach the retained owner."));

  current_owner.reset();
  retained_owner.reset();

  // Retain the current matrix only on rank zero while setup rebuilds its
  // sparsity pattern. The old local rows must remain available unchanged.
  Entries setup_entries;
  auto    setup_previous_owner = flow.derivative_jacobian_matrix_owner();
  std::weak_ptr<const MatrixType> setup_previous_owner_observer =
    setup_previous_owner;
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    {
      retained_owner = setup_previous_owner;
      setup_entries =
        snapshot_entries(*retained_owner, flow.locally_owned_dofs_);
    }
  const auto setup_owned = flow.locally_owned_dofs_;
  setup_previous_owner.reset();
  flow.dof_handler_.clear();
  flow.triangulation().clear();
  flow.n_global_refinements = 1;
  flow.setup();

  current_owner      = flow.derivative_jacobian_matrix_owner();
  retained_unchanged = 1;
  detached           = 1;
  if (retained_owner)
    {
      retained_unchanged =
        maximum_difference(setup_entries,
                           snapshot_entries(*retained_owner, setup_owned)) ==
            0. ?
          1u :
          0u;
      detached = retained_owner.get() != current_owner.get() ? 1u : 0u;
    }
  AssertThrow(Utilities::MPI::min(retained_unchanged, comm) == 1,
              ExcMessage(
                "Sparsity rebuild changed a retained derivative owner."));
  const unsigned int setup_observer_lifetime_matches_expectation =
    setup_previous_owner_observer.expired() != is_root ? 1u : 0u;
  AssertThrow(
    Utilities::MPI::min(setup_observer_lifetime_matches_expectation, comm) == 1,
    ExcMessage(
      "Sparsity rebuild did not detach the matrix on every MPI rank."));
  AssertThrow(
    Utilities::MPI::min(detached, comm) == 1,
    ExcMessage(
      "Sparsity rebuild did not detach the retained derivative owner."));

  deallog.depth_file(10);
  deallog << "Viscoelastic derivative Jacobian owner: PASSED" << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
