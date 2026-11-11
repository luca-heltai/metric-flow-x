// SPDX-License-Identifier: MIT
#ifndef METRIC_FLOW_X_TEST_PRESSURE_JACOBIAN_CHECKS_H
#define METRIC_FLOW_X_TEST_PRESSURE_JACOBIAN_CHECKS_H

#include <metric_flow_x/blood_flow_system.h>

#include <map>
#include <tuple>

inline void
check_sparse_pressure_jacobian(MetricFlowX::BloodFlowSystem<1, 3> &flow,
                               const MetricFlowX::VectorType      &state)
{
  using namespace dealii;
  using Flow            = MetricFlowX::BloodFlowSystem<1, 3>;
  using Index           = types::global_dof_index;
  using Entry           = std::pair<Index, Index>;
  const auto  comm      = flow.mpi_communicator();
  const Index n_columns = 2 * flow.triangulation().n_global_active_cells();
  std::map<CellId, std::tuple<Index, Point<3>, Point<3>>> visible;
  for (const auto &cell : flow.dof_handler().active_cell_iterators())
    if (!cell->is_artificial())
      visible.emplace(cell->id(),
                      std::make_tuple(2 * cell->global_active_cell_index(),
                                      cell->vertex(0),
                                      cell->vertex(1)));
  const Flow::ExternalPressureBasisProvider basis = [&](const auto &point) {
    const auto &[column, p0, p1] = visible.at(point.cell_id);
    const auto   direction       = p1 - p0;
    const double x = ((point.point - p0) * direction) / direction.norm_square();
    AssertThrow(column + 1 < n_columns, ExcInternalError());
    // Repeated columns exercise the visitor's additive-entry contract.
    return Flow::ExternalPressureBasis{{column, .5 * (1. - x)},
                                       {column, .5 * (1. - x)},
                                       {column + 1, x}};
  };
  std::map<Entry, double> matrix;
  flow.visit_external_pressure_jacobian(
    .375, state, basis, [&](Index row, Index column, double value) {
      AssertThrow(flow.locally_owned_dofs().is_element(row) &&
                    column < n_columns && std::isfinite(value),
                  ExcMessage(
                    "Sparse pressure visitor emitted an invalid entry."));
      matrix[{row, column}] += value;
    });

  // Exhaustive global-column probing is deliberately confined to this tiny
  // independent oracle. It also checks field-backed owner/ghost sampling.
  for (Index column = 0; column < n_columns; ++column)
    {
      auto expected = flow.make_state(), actual = flow.make_state();
      flow.add_external_pressure_residual(
        .375,
        state,
        [&](const auto &point) {
          double value = 0.;
          for (const auto &[index, weight] : basis(point))
            if (index == column)
              value += weight;
          return value;
        },
        expected);
      for (const auto &[entry, value] : matrix)
        if (entry.second == column)
          actual[entry.first] += value;
      actual.compress(VectorOperation::add);
      actual -= expected;
      AssertThrow(
        actual.l2_norm() <= 2.e-11 * std::max(expected.l2_norm(), 1.),
        ExcMessage(
          "Sparse pressure column differs from native residual action."));
    }

  const auto coefficient = [](Index column) {
    return 71. * std::sin(.37 * static_cast<double>(column + 1)) +
           3. * static_cast<double>(column % 5);
  };
  auto expected = flow.make_state(), actual = flow.make_state();
  flow.add_external_pressure_residual(
    .375,
    state,
    [&](const auto &point) {
      double value = 0.;
      for (const auto &[column, weight] : basis(point))
        value += coefficient(column) * weight;
      return value;
    },
    expected);
  for (const auto &[entry, value] : matrix)
    actual[entry.first] += value * coefficient(entry.second);
  actual.compress(VectorOperation::add);
  actual -= expected;
  AssertThrow(
    actual.l2_norm() <= 2.e-11 * std::max(expected.l2_norm(), 1.),
    ExcMessage("Sparse pressure product differs from native residual action."));
}

#endif
