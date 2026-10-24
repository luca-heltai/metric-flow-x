// Independently check the rigid single-tube seed, RCR equilibrium, tube-law
// inversion with external pressure, and the complete native consistent pair.
#include <deal.II/base/mpi.h>

#include <deal.II/lac/full_matrix.h>

#include <metric_flow_x/blood_flow_system.h>
#include <metric_flow_x/ida_error_guard.h>
#include <metric_flow_x/steady_network.h>
#include <metric_flow_x/tolerance_ramp.h>

#include <cmath>
#include <fstream>
#include <iomanip>

#include "tests.h"
using namespace dealii;
using namespace MetricFlowX;


// Use the real application residual/Jacobian and an independent dense solve.
// CalcIC counters are inspected before IDASolve does any integration work.
void
check_calc_ic(BloodFlowSystem<1, 3> &flow,
              const VectorType      &initial,
              const VectorType      &initial_derivative,
              const double           t0,
              const double           relative_tolerance,
              const MatrixType      &jacobian)
{
  using Integrator      = SUNDIALS::IDA<VectorType>;
  const double h        = 1e-8;
  auto         absolute = flow.make_absolute_tolerances();
  for (const bool perturb : {false, true})
    {
      auto state = flow.make_state(), derivative = flow.make_state();
      state      = initial;
      derivative = initial_derivative;
      if (perturb)
        {
          // Negative control: inconsistent traces must actually be corrected.
          for (const auto i : flow.algebraic_dofs())
            state(i) *= 1.01;
          state.compress(VectorOperation::insert);
        }
      const VectorType           before(state);
      Integrator::AdditionalData data;
      data.initial_time       = t0;
      data.final_time         = t0 + h;
      data.initial_step_size  = h;
      data.output_period      = h;
      data.relative_tolerance = relative_tolerance;
      data.ic_type            = Integrator::AdditionalData::use_y_diff;
      Integrator    ida(data, MPI_COMM_SELF);
      IDAErrorGuard errors(MPI_COMM_SELF);
      ida.reinit_vector        = [&](VectorType &v) { v = flow.make_state(); };
      ida.get_local_tolerances = [&]() -> VectorType & { return absolute; };
      ida.differential_components = [&] {
        errors.install(MetricFlowX::internal::native_ida_memory(ida), 500);
        return flow.differential_dofs();
      };
      ida.residual = [&](double            t,
                         const VectorType &y,
                         const VectorType &yp,
                         VectorType       &r) {
        flow.assemble_residual(t, y, yp, r);
      };
      FullMatrix<double> inverse;
      unsigned int       setups = 0, linear_solves = 0;
      unsigned int       initial_outputs = 0, final_outputs = 0;
      ida.setup_jacobian =
        [&](double t, const VectorType &y, const VectorType &yp, double alpha) {
          ++setups;
          flow.assemble_jacobian(t, y, yp, alpha);
          inverse.reinit(y.size(), y.size());
          for (unsigned int i = 0; i < y.size(); ++i)
            for (unsigned int j = 0; j < y.size(); ++j)
              inverse(i, j) = jacobian.el(i, j);
          inverse.gauss_jordan();
        };
      ida.solve_with_jacobian =
        [&](const VectorType &rhs, VectorType &dst, double) {
          ++linear_solves;
          dealii::Vector<double> b(rhs.size()), x(rhs.size());
          for (unsigned int i = 0; i < rhs.size(); ++i)
            b(i) = rhs(i);
          inverse.vmult(x, b);
          for (unsigned int i = 0; i < rhs.size(); ++i)
            dst(i) = x(i);
          dst.compress(VectorOperation::insert);
        };
      ida.output_step = [&](double,
                            const VectorType &y,
                            const VectorType &yp,
                            unsigned int      output) {
        errors.check();
        auto    *memory = MetricFlowX::internal::native_ida_memory(ida);
        long int steps = -1, iterations = -1, backtracks = -1;
        long int error_fails = -1, convergence_fails = -1;
        AssertThrow(IDAGetNumSteps(memory, &steps) == 0 &&
                      IDAGetNumNonlinSolvIters(memory, &iterations) == 0 &&
                      IDAGetNumBacktrackOps(memory, &backtracks) == 0 &&
                      IDAGetNumErrTestFails(memory, &error_fails) == 0 &&
                      IDAGetNumNonlinSolvConvFails(memory,
                                                   &convergence_fails) == 0,
                    ExcInternalError());
        if (output == 0)
          {
            ++initial_outputs;
            AssertThrow(steps == 0 && setups > 0,
                        ExcMessage("CalcIC must run before integration."));
            double correction = 0., derivative_correction = 0.;
            for (const auto i : flow.locally_owned_dofs())
              {
                const double weight =
                  absolute(i) + data.relative_tolerance * std::abs(initial(i));
                correction =
                  std::max(correction, std::abs(y(i) - before(i)) / weight);
                if (flow.differential_dofs().is_element(i))
                  derivative_correction =
                    std::max(derivative_correction,
                             h * std::abs(yp(i) - initial_derivative(i)) /
                               weight);
              }
            // IDACalcIC performs two weight-evaluation passes, each with
            // a Jacobian setup and an initial linear solve even for F=0.
            if (!perturb)
              AssertThrow(
                correction < 1. && derivative_correction < 1. &&
                  iterations <= 2 && setups <= 2 && linear_solves <= 4 &&
                  backtracks == 0 && convergence_fails == 0,
                ExcMessage(
                  "Consistent pair required substantial CalcIC correction/work: " +
                  std::to_string(correction) + ", " +
                  std::to_string(derivative_correction) +
                  ", iterations=" + std::to_string(iterations) +
                  ", setups=" + std::to_string(setups)));
            else
              AssertThrow(correction > 1.,
                          ExcMessage(
                            "CalcIC failed to repair perturbed traces."));
          }
        else
          {
            ++final_outputs;
            if (!perturb)
              {
                double first_step = 0.;
                AssertThrow(
                  IDAGetActualInitStep(memory, &first_step) == 0 &&
                    steps == 1 && error_fails == 0 && convergence_fails == 0 &&
                    std::abs(first_step / h - 1.) < 1e-12,
                  ExcMessage(
                    "Consistent pair did not accept the prescribed small first step."));
              }
          }
      };
      ida.solve_dae(state, derivative);
      errors.check();
      AssertThrow(initial_outputs == 1 && final_outputs == 1,
                  ExcMessage("Missing initial/final IDA observation."));
    }
}

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
            // Native CalcIC has an empty-rank defect in the installed SDK.
            // Keep this independent solver check serial; the pair itself is
            // still exercised above on all MPI partitions.
            if (Utilities::MPI::n_mpi_processes(comm) == 1 && refinement == 0)
              {
                check_calc_ic(flow,
                              state,
                              derivative,
                              t0,
                              flow.ida_parameters.relative_tolerance,
                              flow.jacobian_matrix);
                flow.initial_state_source = "rest";
                auto rest = flow.make_state(), rest_dot = flow.make_state();
                flow.initialize_consistent_state(rest, rest_dot, t0);
                check_calc_ic(flow,
                              rest,
                              rest_dot,
                              t0,
                              flow.ida_parameters.relative_tolerance,
                              flow.jacobian_matrix);
                flow.initial_state_source = "steady network";
              }
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
