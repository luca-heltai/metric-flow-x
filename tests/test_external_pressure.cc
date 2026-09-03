// ---------------------------------------------------------------------
// External-pressure provider tests.
//
// The provider is prescribed data: it is held fixed while differentiating
// with respect to the BloodFlowSystem state.  The test uses the single-vessel
// fixture so that the constant-pressure oracle is exact at the discrete level:
// a constant pressure has zero weak cell contribution and cancels across
// interior fluxes, leaving only the RCR pressure relation at the outlet.
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <metric_flow_x/blood_flow_system.h>

#include <cmath>
#include <map>

#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

namespace
{
  constexpr double tolerance = 2.0e-10;

  double
  max_difference(const VectorType &a, const VectorType &b, const MPI_Comm comm)
  {
    double local = 0.0;
    for (const auto i : a.locally_owned_elements())
      local = std::max(local, std::abs(a(i) - b(i)));
    return Utilities::MPI::max(local, comm);
  }

  void
  assert_close(const VectorType &a,
               const VectorType &b,
               const MPI_Comm    comm,
               const double      tol = tolerance)
  {
    AssertThrow(max_difference(a, b, comm) < tol, ExcInternalError());
  }

  using MatrixSnapshot =
    std::map<std::pair<types::global_dof_index, types::global_dof_index>,
             double>;

  MatrixSnapshot
  snapshot(const MatrixType &matrix, const IndexSet &owned)
  {
    MatrixSnapshot result;
    for (const auto row : owned)
      for (auto it = matrix.begin(row); it != matrix.end(row); ++it)
        result[{row, it->column()}] = it->value();
    return result;
  }

  double
  snapshot_max_difference(const MatrixSnapshot &a,
                          const MatrixSnapshot &b,
                          const MPI_Comm        comm)
  {
    double local = 0.0;
    for (const auto &[key, value] : a)
      local = std::max(local, std::abs(value - b.at(key)));
    return Utilities::MPI::max(local, comm);
  }

  void
  assemble(BloodFlowSystem<1, 3> &problem,
           const double           t,
           const VectorType      &y,
           const VectorType      &ydot,
           VectorType            &F)
  {
    problem.assemble_residual(t, y, ydot, F);
  }

  void
  add_external_pressure(
    BloodFlowSystem<1, 3>                                 &problem,
    const double                                           t,
    const VectorType                                      &y,
    const BloodFlowSystem<1, 3>::ExternalPressureProvider &provider,
    VectorType                                            &F)
  {
    problem.add_external_pressure_residual(t, y, provider, F);
  }
} // namespace

void
test()
{
  BloodFlowSystem<1, 3> problem;
  problem.initialize_params(PRM_DIR "constant.prm");
  deallog.depth_file(10);
  problem.setup();

  auto y = problem.make_state();
  problem.compute_initial_solution(y, 0.0);
  auto ydot = problem.make_state();
  ydot      = 0.0;

  // No provider and an explicitly zero provider are identical, including both
  // native Jacobian blocks.
  problem.clear_external_pressure_provider();
  auto F0 = problem.make_state();
  assemble(problem, 0.0, y, ydot, F0);
  auto F_zero_native = problem.make_state();
  add_external_pressure(
    problem,
    0.0,
    y,
    [](const BloodFlowSystem<1, 3>::PressureEvaluationPoint &) { return 0.0; },
    F_zero_native);
  AssertThrow(F_zero_native.l2_norm() < tolerance, ExcInternalError());
  problem.assemble_state_jacobian(0.0, y, ydot);
  const auto J0 =
    snapshot(problem.state_jacobian_matrix(), problem.locally_owned_dofs_);
  problem.assemble_derivative_jacobian(0.0, y, ydot);
  const auto M0 =
    snapshot(problem.derivative_jacobian_matrix(), problem.locally_owned_dofs_);

  problem.set_external_pressure_provider(
    [](const BloodFlowSystem<1, 3>::PressureEvaluationPoint &) { return 0.0; });
  auto F_zero = problem.make_state();
  assemble(problem, 0.0, y, ydot, F_zero);
  problem.assemble_state_jacobian(0.0, y, ydot);
  const auto J_zero =
    snapshot(problem.state_jacobian_matrix(), problem.locally_owned_dofs_);
  problem.assemble_derivative_jacobian(0.0, y, ydot);
  const auto M_zero =
    snapshot(problem.derivative_jacobian_matrix(), problem.locally_owned_dofs_);

  assert_close(F0, F_zero, problem.mpi_communicator());
  AssertThrow(snapshot_max_difference(J0, J_zero, problem.mpi_communicator()) <
                tolerance,
              ExcInternalError());
  AssertThrow(snapshot_max_difference(M0, M_zero, problem.mpi_communicator()) <
                tolerance,
              ExcInternalError());

  // A constant external pressure is added to physical pressure.  In the
  // residual F = M*ydot - R, its discrete weak contribution cancels in every
  // cell and interior face.  The only remaining contribution in this fixture
  // is -delta_p in the outlet RCR trace-pressure row.
  constexpr double delta_p = 321.0;
  problem.set_external_pressure_provider(
    [](const BloodFlowSystem<1, 3>::PressureEvaluationPoint &) {
      return delta_p;
    });
  auto F_constant = problem.make_state();
  assemble(problem, 0.0, y, ydot, F_constant);
  auto D_constant = F_constant;
  D_constant -= F0;

  auto F_constant_native = problem.make_state();
  add_external_pressure(
    problem,
    0.0,
    y,
    [](const BloodFlowSystem<1, 3>::PressureEvaluationPoint &) {
      return delta_p;
    },
    F_constant_native);
  assert_close(D_constant, F_constant_native, problem.mpi_communicator());

  types::global_dof_index outlet_area_row = numbers::invalid_dof_index;
  for (const auto &cell : problem.dof_handler_.active_cell_iterators())
    if (cell->is_locally_owned())
      for (unsigned int f = 0; f < GeometryInfo<1>::faces_per_cell; ++f)
        if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 1)
          outlet_area_row =
            problem.face_dof_map.at(problem.canonical_face_key(cell, f))
              .a_hat_dof;

  outlet_area_row =
    Utilities::MPI::min(outlet_area_row, problem.mpi_communicator());
  AssertThrow(outlet_area_row != numbers::invalid_dof_index,
              ExcInternalError());
  double constant_error_local = 0.0;
  for (const auto i : problem.locally_owned_dofs_)
    {
      const double expected = (i == outlet_area_row) ? -delta_p : 0.0;
      constant_error_local =
        std::max(constant_error_local, std::abs(D_constant(i) - expected));
    }
  AssertThrow(Utilities::MPI::max(constant_error_local,
                                  problem.mpi_communicator()) < tolerance,
              ExcInternalError());

  const Point<3> sample_point(0.123, 0.0, 0.0);
  AssertThrow(std::abs((problem.pressure(1.0, 0, 0.0, sample_point) -
                        problem.pressure(1.0, 0)) -
                       delta_p) < tolerance,
              ExcInternalError());

  // Spatial evaluation uses the physical cell center, and the provider also
  // receives the vessel id.  The pressure output is a convenient direct
  // oracle for the location at which this data is sampled.
  problem.set_external_pressure_provider(
    [](const BloodFlowSystem<1, 3>::PressureEvaluationPoint &evaluation) {
      return 17.0 + 100.0 * evaluation.point[0] +
             3.0 * static_cast<double>(evaluation.vessel_id);
    });
  auto F_spatial_full = problem.make_state();
  assemble(problem, 0.0, y, ydot, F_spatial_full);
  auto F_spatial_native = problem.make_state();
  add_external_pressure(
    problem,
    0.0,
    y,
    [](const BloodFlowSystem<1, 3>::PressureEvaluationPoint &evaluation) {
      return 17.0 + 100.0 * evaluation.point[0] +
             3.0 * static_cast<double>(evaluation.vessel_id);
    },
    F_spatial_native);
  F_spatial_full -= F0;
  assert_close(F_spatial_full,
               F_spatial_native,
               problem.mpi_communicator(),
               4.0e-10);
  auto p_with_spatial = problem.make_state();
  problem.compute_pressure(y, p_with_spatial, 0.0);
  for (const auto &cell : problem.dof_handler_.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        std::vector<types::global_dof_index> ldofs(
          problem.fe_->n_dofs_per_cell());
        cell->get_dof_indices(ldofs);
        const double expected_external =
          17.0 + 100.0 * cell->center()[0] +
          3.0 * static_cast<double>(cell->material_id());
        for (unsigned int i = 0; i < problem.fe_->n_dofs_per_cell(); ++i)
          if (problem.fe_->system_to_component_index(i).first == 0)
            {
              const double base =
                problem.pressure(y(ldofs[i]), cell->material_id());
              AssertThrow(std::abs(p_with_spatial(ldofs[i]) - base -
                                   expected_external) < tolerance,
                          ExcInternalError());
            }
      }

  // Time is part of the same provider context and does not require a time
  // step.  The constant-pressure residual oracle above gives the exact sign.
  problem.set_external_pressure_provider(
    [](const BloodFlowSystem<1, 3>::PressureEvaluationPoint &evaluation) {
      return 11.0 * evaluation.time;
    });
  auto Ft0 = problem.make_state();
  auto Ft1 = problem.make_state();
  assemble(problem, 0.0, y, ydot, Ft0);
  assemble(problem, 2.0, y, ydot, Ft1);
  auto Dt = Ft1;
  Dt -= Ft0;
  auto Dt_native = problem.make_state();
  add_external_pressure(
    problem,
    2.0,
    y,
    [](const BloodFlowSystem<1, 3>::PressureEvaluationPoint &evaluation) {
      return 11.0 * evaluation.time;
    },
    Dt_native);
  assert_close(Dt, Dt_native, problem.mpi_communicator());
  double time_error_local = 0.0;
  for (const auto i : problem.locally_owned_dofs_)
    {
      const double expected = (i == outlet_area_row) ? -22.0 : 0.0;
      time_error_local = std::max(time_error_local, std::abs(Dt(i) - expected));
    }
  AssertThrow(Utilities::MPI::max(time_error_local,
                                  problem.mpi_communicator()) < tolerance,
              ExcInternalError());

  // The residual contribution is linear in prescribed pressure, including
  // spatial and vessel-id dependence.  This is the native F_ext oracle for a
  // future coupling layer.
  const auto p1 = [](const auto &evaluation) {
    return 4.0 + 8.0 * evaluation.point[0] + evaluation.time;
  };
  const auto p2 = [](const auto &evaluation) {
    return -3.0 + 2.0 * evaluation.point[0] - 0.5 * evaluation.time;
  };
  problem.set_external_pressure_provider(p1);
  auto F1 = problem.make_state();
  assemble(problem, 0.75, y, ydot, F1);
  auto F1_native = problem.make_state();
  add_external_pressure(problem, 0.75, y, p1, F1_native);
  problem.set_external_pressure_provider(p2);
  auto F2 = problem.make_state();
  assemble(problem, 0.75, y, ydot, F2);
  auto F2_native = problem.make_state();
  add_external_pressure(problem, 0.75, y, p2, F2_native);
  problem.set_external_pressure_provider([p1, p2](const auto &evaluation) {
    return 2.0 * p1(evaluation) - 0.5 * p2(evaluation);
  });
  auto Fcombo = problem.make_state();
  assemble(problem, 0.75, y, ydot, Fcombo);
  auto Fcombo_native = problem.make_state();
  add_external_pressure(
    problem,
    0.75,
    y,
    [p1, p2](const auto &evaluation) {
      return 2.0 * p1(evaluation) - 0.5 * p2(evaluation);
    },
    Fcombo_native);
  auto native_difference = F1_native;
  native_difference *= 2.0;
  native_difference.add(-0.5, F2_native);
  native_difference -= Fcombo_native;
  AssertThrow(native_difference.l2_norm() < 1.0e-9, ExcInternalError());
  auto linearity_error = Fcombo;
  linearity_error -= F0;
  auto D1 = F1;
  auto D2 = F2;
  D1 -= F0;
  D2 -= F0;
  linearity_error.add(-2.0, D1);
  linearity_error.add(0.5, D2);
  AssertThrow(linearity_error.l2_norm() < 1.0e-9, ExcInternalError());

  // The state Jacobian remains the derivative at fixed external pressure.
  problem.set_external_pressure_provider(
    [](const BloodFlowSystem<1, 3>::PressureEvaluationPoint &evaluation) {
      return 250.0 + 13.0 * evaluation.point[0] + 5.0 * evaluation.time;
    });
  problem.assemble_state_jacobian(0.0, y, ydot);
  auto direction = problem.make_state();
  direction      = 0.0;
  for (const auto i : problem.locally_owned_dofs())
    direction(i) = 0.2 + 0.01 * static_cast<double>(i % 7);
  direction.compress(VectorOperation::insert);

  constexpr double epsilon = 1.0e-7;
  auto             yp      = y;
  auto             ym      = y;
  yp.add(epsilon, direction);
  ym.add(-epsilon, direction);
  auto Fp = problem.make_state();
  auto Fm = problem.make_state();
  assemble(problem, 0.0, yp, ydot, Fp);
  assemble(problem, 0.0, ym, ydot, Fm);
  Fp -= Fm;
  Fp /= 2.0 * epsilon;
  auto Jdirection = problem.make_state();
  problem.state_jacobian_matrix().vmult(Jdirection, direction);
  Fp -= Jdirection;
  AssertThrow(Fp.l2_norm() / std::max(1.0, Jdirection.l2_norm()) < 2.0e-6,
              ExcInternalError());

  // The provider has no ydot dependence, so the derivative Jacobian is the
  // same native mass block as in the zero-pressure case.
  problem.assemble_derivative_jacobian(0.0, y, ydot);
  const auto M_nonzero =
    snapshot(problem.derivative_jacobian_matrix(), problem.locally_owned_dofs_);
  AssertThrow(snapshot_max_difference(M0,
                                      M_nonzero,
                                      problem.mpi_communicator()) < tolerance,
              ExcInternalError());

  if (Utilities::MPI::this_mpi_process(problem.mpi_communicator()) == 0)
    deallog << "External pressure tests: PASSED" << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
