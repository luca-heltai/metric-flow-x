/* -----------------------------------------------------------------------------
 *
 *  SPDX-License-Identifier: LGPL-2.1-or-later
 *  Copyright:
 *  2024-2025 by the deal.II authors
 *
 *  This file is part of the blood-flow example built on the deal.II library.
 *  It provides the top-level executable that steers the templated
 *  BloodFlowSystem<1,3> class.  The structure mirrors main_embedded.cc so that
 *  build rules, CMake targets, and user habits remain consistent across
 *  multiple applications in the same repository.
 *
 *  --------------------------------------------------------------------------
 */

#include <deal.II/base/logstream.h> // deallog control
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>

#include <metric_flow_x/blood_flow_system.h> // header exposing BloodFlowSystem
#include <metric_flow_x/ida_error_guard.h>
#include <metric_flow_x/serial_direct_solver.h>
#include <metric_flow_x/tolerance_ramp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept> // for std::invalid_argument

using namespace dealii;
using namespace MetricFlowX;

namespace MetricFlowX
{
  /** The standalone executable's stationary and IDA/time-stepping policy.
   *
   * BloodFlowSystem remains the mathematical Problem: it owns the mesh,
   * physics, residual, Jacobians, state, and observables.  This runner owns
   * IDA, linear-solver policy, time stepping, and output orchestration.
   */
  template <int dim, int spacedim>
  class BloodFlowIDARunner
  {
  public:
    using Problem    = BloodFlowSystem<dim, spacedim>;
    using VectorType = MetricFlowX::VectorType;
    using RampSource = ToleranceRampSource<VectorType>;

    explicit BloodFlowIDARunner(Problem &problem)
    {
      problem.add_parameter(
        "Run mode",
        run_mode_,
        "transient integrates the HDG model; steady network solves the rigid "
        "resistive 0D graph; steady HDG solves F(t0,y,0)=0 with IDACalcIC.",
        problem.prm,
        Patterns::Selection("transient|steady network|steady HDG"));
      problem.enter_subsection("IDA parameters");
      problem.enter_subsection("Running parameters");
      problem.add_parameter(
        "Maximum step size",
        maximum_step_size_,
        "Native IDA maximum step size; zero leaves it unlimited.");
      problem.leave_subsection();
      problem.leave_subsection();
      tolerance_ramp_.add_parameters(problem);
      problem.add_parameter("Maximum IDA internal steps per output",
                            maximum_ida_steps_per_output_,
                            "Positive work limit for each IDASolve call.");
    }


    void
    run(Problem &problem)
    {
      const bool steady_network_only = run_mode_ == "steady network";
      const bool steady_hdg          = run_mode_ == "steady HDG";
      AssertThrow(std::isfinite(problem.ida_parameters.initial_time),
                  ExcMessage("Initial time must be finite."));
      if (!steady_network_only && !steady_hdg)
        {
          AssertThrow(
            maximum_ida_steps_per_output_ > 0,
            ExcMessage(
              "Maximum IDA internal steps per output must be positive."));
          AssertThrow(
            std::isfinite(problem.ida_parameters.minimum_step_size) &&
              problem.ida_parameters.minimum_step_size >= 0. &&
              std::isfinite(maximum_step_size_) && maximum_step_size_ >= 0. &&
              (maximum_step_size_ == 0. ||
               problem.ida_parameters.minimum_step_size <= maximum_step_size_),
            ExcMessage(
              "IDA step limits require 0 <= minimum <= maximum (maximum=0 is unlimited)."));
          tolerance_ramp_.validate(problem.ida_parameters.initial_time,
                                   problem.ida_parameters.final_time,
                                   problem.ida_parameters.output_period);
        }

      problem.pcout << "=== Blood Flow HDG, polynomial degree p = "
                    << problem.fe_degree << ", running on "
                    << problem.n_mpi_processes << " MPI rank(s) ===\n";

      for (unsigned int cycle = 0; cycle < problem.n_refinement_cycles; ++cycle)
        {
          if (problem.verbosity > 0)
            problem.pcout << "\n--- Refinement cycle " << cycle << " ---\n";

          if (cycle == 0)
            problem.setup();
          else
            problem.triangulation_.refine_global(1);

          if (cycle > 0)
            {
              problem.setup_system();
              problem.initialize_terminal_capacitors();
              problem.build_per_cell_mass_inv();
            }

          if (steady_network_only)
            {
              problem.pcout
                << "Solving the stationary resistive 0D network at t = "
                << problem.ida_parameters.initial_time << '\n';
              problem.time = problem.ida_parameters.initial_time;
              // Preserve the graph reconstruction: an HDG trace correction
              // would change it without solving a stationary HDG problem.
              problem.compute_steady_network_initial_solution(problem.solution,
                                                              problem.time);
              output_stationary(problem, cycle);
              continue;
            }

          // After setup_system(), so the vessel probe cells and DoF indices
          // belong to the current mesh.  Opens HDG_IDA_Vessel_*.csv,
          // mass_conservation.csv and ida_step_history.csv, and resets the
          // IDA counters for this cycle.
          if (!steady_hdg)
            problem.open_output_files();
          const auto preparation_start = std::chrono::steady_clock::now();
          const bool native_pair_verified =
            !steady_hdg && (problem.initial_condition_policy == "native" ||
                            problem.ida_parameters.ic_type ==
                              SUNDIALS::IDA<VectorType>::AdditionalData::none);
          if (native_pair_verified)
            problem.initialize_consistent_state(
              problem.solution,
              problem.solution_dot,
              problem.ida_parameters.initial_time);
          else
            {
              problem.initialize_state(problem.solution,
                                       problem.ida_parameters.initial_time);
              problem.initialize_state_derivative(
                problem.solution_dot, problem.ida_parameters.initial_time);
            }
          const double preparation_wall =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          preparation_start)
              .count();
          bool          initial_output_seen = false;
          unsigned long ic_residual_calls = 0, ic_jacobian_setups = 0,
                        ic_linear_solves = 0;
          double ida_initialization_wall = 0., first_interval_wall = -1.;
          std::chrono::steady_clock::time_point solve_start, integration_start;
          problem.time = problem.ida_parameters.initial_time;

          auto check_jacobian_fd = [&problem](const VectorType &y0) {
            using Comp          = typename Problem::Component;
            const MPI_Comm comm = problem.mpi_communicator_;
            const double   t    = problem.time;
            const double   eps  = 1e-6; // relative perturbation

            VectorType ydot0(problem.locally_owned_dofs_, comm);
            ydot0 = 0.0;
            problem.assemble_jacobian(t, y0, ydot0, /*alpha=*/0.0);

            // Global max |y| over a block: used as a floor for unknowns that
            // are (close to) zero, e.g. the velocities at rest at t = 0.
            auto block_max = [&](const IndexSet &set) {
              double m = 0.0;
              for (const auto i : set)
                m = std::max(m, std::abs(y0(i)));
              return Utilities::MPI::max(m, comm);
            };

            struct ColBlock
            {
              std::string     name;
              const IndexSet *set;
              double          floor;
            };
            const IndexSet &sA  = problem.component_dofs(Comp::area);
            const IndexSet &sU  = problem.component_dofs(Comp::velocity);
            const IndexSet &sAh = problem.component_dofs(Comp::area_trace);
            const IndexSet &sUh = problem.component_dofs(Comp::velocity_trace);

            const std::vector<ColBlock> col_blocks = {
              {"A", &sA, 1e-3 * block_max(sA)},
              {"U", &sU, std::max(1e-3, 1e-3 * block_max(sU))},
              {"A_hat", &sAh, 1e-3 * block_max(sAh)},
              {"U_hat", &sUh, std::max(1e-3, 1e-3 * block_max(sUh))},
              {"Pc", &problem.rcr_dofs_owned, 1.0}};

            problem.pcout
              << "\n========== FD Jacobian check (scaled, eps = " << eps
              << ") ==========\n"
              << "  column   | rel.err CELL rows | TRACE rows | RCR rows\n";

            std::srand(problem.this_mpi_process + 1);

            for (const auto &cb : col_blocks)
              {
                // v perturbs only this column block, each entry scaled to the
                // size of the unknown it perturbs.
                VectorType v(problem.locally_owned_dofs_, comm);
                v = 0.0;
                for (const auto i : *cb.set)
                  {
                    const double r = 2.0 * std::rand() / double(RAND_MAX) - 1.0;
                    v(i)           = r * std::max(std::abs(y0(i)), cb.floor);
                  }
                v.compress(VectorOperation::insert);
                if (v.l2_norm() == 0.0)
                  continue; // e.g. no capacitor unknowns

                VectorType yp(y0), ym(y0);
                yp.add(eps, v);
                ym.add(-eps, v);

                VectorType Fp(problem.locally_owned_dofs_, comm);
                VectorType Fm(problem.locally_owned_dofs_, comm);
                problem.assemble_residual(t, yp, ydot0, Fp);
                problem.assemble_residual(t, ym, ydot0, Fm);

                VectorType fd(Fp);
                fd -= Fm;
                fd /= (2.0 * eps);

                VectorType Jv(problem.locally_owned_dofs_, comm);
                problem.jacobian_matrix.vmult(Jv, v);

                // Squared error and squared reference, per row type:
                // [0] cell, [1] trace, [2] rcr.
                double local[6] = {0, 0, 0, 0, 0, 0};
                for (const auto i : problem.locally_owned_dofs_)
                  {
                    const unsigned int k =
                      problem.cell_dofs_owned.is_element(i)  ? 0 :
                      problem.trace_dofs_owned.is_element(i) ? 1 :
                                                               2;
                    const double di = Jv(i) - fd(i);
                    local[k] += di * di;
                    local[3 + k] += fd(i) * fd(i);
                  }
                double global[6];
                MPI_Allreduce(local, global, 6, MPI_DOUBLE, MPI_SUM, comm);

                auto rel = [&](const unsigned int k) -> std::string {
                  const double       e = std::sqrt(global[k]);
                  const double       r = std::sqrt(global[3 + k]);
                  std::ostringstream s;
                  s << std::scientific << std::setprecision(2);
                  if (e == 0.0 && r == 0.0)
                    s << "      -     ";
                  else if (r == 0.0)
                    s << "abs " << e; // Jacobian entry where residual is flat
                  else
                    s << e / r;
                  return s.str();
                };

                problem.pcout << "  " << std::setw(8) << std::left << cb.name
                              << " | " << std::setw(17) << rel(0) << " | "
                              << std::setw(10) << rel(1) << " | " << rel(2)
                              << "\n";
              }
            problem.pcout << "==========================================="
                             "==========\n";
          };

          if (problem.check_initial_jacobian)
            check_jacobian_fd(problem.solution);

          auto ida_parameters = problem.ida_parameters;
          if (steady_hdg)
            {
              ida_parameters.ic_type =
                SUNDIALS::IDA<VectorType>::AdditionalData::use_y_dot;
              // Trigger the wrapper's differential-component setup hook so
              // the native error guard is installed before CalcIC. There is
              // no integration error test in this mode.
              ida_parameters.ignore_algebraic_terms_for_errors = true;
              if (ida_parameters.initial_step_size == 0.)
                ida_parameters.initial_step_size = 1.;
              AssertThrow(
                std::isfinite(ida_parameters.initial_step_size) &&
                  ida_parameters.initial_step_size > 0.,
                ExcMessage(
                  "Steady HDG requires a finite positive IC time scale."));
              ida_parameters.final_time =
                ida_parameters.initial_time + ida_parameters.initial_step_size;
              ida_parameters.output_period = ida_parameters.initial_step_size;
              AssertThrow(
                std::isfinite(ida_parameters.final_time) &&
                  ida_parameters.final_time > ida_parameters.initial_time,
                ExcMessage("Steady HDG IC time scale is not representable."));
            }
          else if (problem.initial_condition_policy == "native")
            ida_parameters.ic_type =
              SUNDIALS::IDA<VectorType>::AdditionalData::none;
          IDAErrorGuard::validate_setup(ida_parameters);
          IDAErrorGuard             ida_errors(problem.mpi_communicator_);
          SUNDIALS::IDA<VectorType> ida(ida_parameters,
                                        problem.mpi_communicator_);

          ida.reinit_vector = [&problem](VectorType &v) {
            v.reinit(problem.locally_owned_dofs_, problem.mpi_communicator_);
          };

          auto absolute_tolerances = problem.make_state();
          if (problem.use_componentwise_tolerances)
            {
              absolute_tolerances = problem.make_absolute_tolerances();
              ida.get_local_tolerances =
                [&absolute_tolerances]() -> VectorType & {
                return absolute_tolerances;
              };
            }

          if ((tolerance_ramp_.enabled || steady_hdg) &&
              !problem.use_componentwise_tolerances)
            absolute_tolerances = ida_parameters.absolute_tolerance;
          auto weight_policy = tolerance_ramp_;
          if (steady_hdg)
            {
              // Keep final componentwise weights constant. The custom-weight
              // path avoids the wrapper's SV minimum on empty MPI ranks.
              weight_policy.enabled                 = true;
              weight_policy.initial_relative_factor = 1.;
              weight_policy.initial_absolute_factor = 1.;
              weight_policy.hold_duration           = 0.;
              weight_policy.end_time                = ida_parameters.final_time;
            }
          RampSource::attach(ida,
                             weight_policy,
                             ida_parameters,
                             absolute_tolerances);

          const long int maximum_steps =
            steady_hdg ? 500 : maximum_ida_steps_per_output_;
          const double minimum_step =
            steady_hdg ? 0. : ida_parameters.minimum_step_size;
          const double maximum_step   = steady_hdg ? 0. : maximum_step_size_;
          ida.differential_components = [&problem,
                                         &ida,
                                         &ida_errors,
                                         maximum_steps,
                                         minimum_step,
                                         maximum_step]() -> IndexSet {
            ida_errors.install(ida, maximum_steps, minimum_step, maximum_step);
            RampSource::register_weights(internal::native_ida_memory(ida), ida);
            IndexSet is(problem.n_total_dofs);
            is.add_indices(problem.cell_dofs_owned);
            is.add_indices(problem.rcr_dofs_owned);
            is.compress();
            return is;
          };

          ida.residual = [&problem,
                          &ida,
                          &initial_output_seen,
                          &ic_residual_calls](const double      t,
                                              const VectorType &y,
                                              const VectorType &ydot,
                                              VectorType       &res) -> int {
            if (!initial_output_seen)
              ++ic_residual_calls;
            problem.poll_ida_internal_step(ida, false, true);
            problem.assemble_residual(t, y, ydot, res);
            return 0;
          };

          std::unique_ptr<SerialDirectSolver> serial_direct;
          ida.setup_jacobian = [&problem,
                                &initial_output_seen,
                                &ic_jacobian_setups,
                                &serial_direct](const double      t,
                                                const VectorType &y,
                                                const VectorType &ydot,
                                                const double alpha) -> int {
            if (!initial_output_seen)
              ++ic_jacobian_setups;
            ++problem.ida_jacobian_evaluations;
            TimerOutput::Scope ts(problem.computing_timer, "setup_jacobian");
            problem.assemble_jacobian(t, y, ydot, alpha);
            problem.linear_system_matrix.copy_from(problem.jacobian_matrix);

            if (problem.use_direct_solver)
              {
#ifdef METRIC_FLOW_X_WITH_PETSC
#  ifdef DEAL_II_PETSC_WITH_MUMPS
                problem.direct_solver =
                  std::make_unique<PETScWrappers::SparseDirectMUMPS>(
                    problem.direct_solver_control);
#  else
                AssertThrow(
                  problem.n_mpi_processes == 1,
                  ExcMessage(
                    "Parallel direct solves require PETSc with MUMPS. Set Use direct solver = false."));
#    ifdef DEAL_II_WITH_UMFPACK
                if (!serial_direct)
                  serial_direct = std::make_unique<SerialDirectSolver>();
                serial_direct->initialize(problem.linear_system_matrix);
#    else
                AssertThrow(
                  false,
                  ExcMessage(
                    "Serial direct solves require deal.II with UMFPACK or PETSc with MUMPS. Set Use direct solver = false."));
#    endif

#  endif
#else
                problem.direct_solver =
                  std::make_unique<TrilinosWrappers::SolverDirect>(
                    problem.direct_solver_control);
                problem.direct_solver->initialize(problem.linear_system_matrix);
#endif
              }
            else
              {
                problem.ilu_preconditioner =
                  std::make_unique<typename Problem::IterativePreconditioner>();
                problem.initialize_iterative_preconditioner(
                  *problem.ilu_preconditioner, problem.linear_system_matrix);
              }
            return 0;
          };

          ida.solve_with_jacobian =
            [&problem, &initial_output_seen, &ic_linear_solves, &serial_direct](
              const VectorType &r, VectorType &z, const double /*tol*/) -> int {
            if (!initial_output_seen)
              ++ic_linear_solves;
            ++problem.ida_linear_solves;
            TimerOutput::Scope ts(problem.computing_timer,
                                  "solve_with_jacobian");

            if (problem.use_direct_solver)
              {
#ifdef METRIC_FLOW_X_WITH_PETSC
#  ifdef DEAL_II_PETSC_WITH_MUMPS
                problem.direct_solver->solve(problem.linear_system_matrix,
                                             z,
                                             r);
#  else
#    ifdef DEAL_II_WITH_UMFPACK
                serial_direct->vmult(z, r);
#    else
                AssertThrow(false,
                            ExcMessage(
                              "Serial direct UMFPACK support is unavailable."));
#    endif
#  endif
#else
                problem.direct_solver->solve(z, r);
#endif
              }
            else
              {
                SolverControl solver_control(1000, 1e-10 * r.l2_norm());
                LA::SolverGMRES::AdditionalData solver_data;
                // SolverControl uses the unpreconditioned RHS norm.
                // Right preconditioning keeps the residual in those units.
                solver_data.right_preconditioning = true;
                LA::SolverGMRES solver(solver_control, solver_data);
                solver.solve(problem.linear_system_matrix,
                             z,
                             r,
                             *problem.ilu_preconditioner);
                problem.ida_linear_iterations += solver_control.last_step();
              }
            return 0;
          };

          const double initial_output_time =
            problem.ida_parameters.initial_time;
          const double final_output_time = problem.ida_parameters.final_time;
          double       next_output_time  = initial_output_time;
          if (steady_hdg)
            {
              problem.pcout
                << "Solving stationary HDG equations with IDACalcIC(use_y_dot) at t = "
                << problem.time << '\n';
              // reset() performs CalcIC and retrieves the corrected pair;
              // solve_dae() would additionally advance time.
              ida.reset(problem.time,
                        ida_parameters.initial_step_size,
                        problem.solution,
                        problem.solution_dot);
              ida_errors.check();
              long int steps        = -1;
              double   current_time = 0.;
              void    *memory       = internal::native_ida_memory(ida);
              AssertThrow(
                IDAGetNumSteps(memory, &steps) == 0 && steps == 0 &&
                  IDAGetCurrentTime(memory, &current_time) == 0 &&
                  current_time == problem.time &&
                  problem.solution_dot.linfty_norm() == 0.,
                ExcMessage(
                  "Stationary CalcIC changed time or the prescribed zero derivative."));
              auto residual   = problem.make_state(),
                   correction = problem.make_state();
              ida.residual(problem.time,
                           problem.solution,
                           problem.solution_dot,
                           residual);
              ida.setup_jacobian(problem.time,
                                 problem.solution,
                                 problem.solution_dot,
                                 0.);
              ida.solve_with_jacobian(residual, correction, 0.);
              double       defect  = 0.;
              unsigned int invalid = 0;
              for (const auto i : problem.locally_owned_dofs_)
                {
                  const double weight =
                    absolute_tolerances(i) + ida_parameters.relative_tolerance *
                                               std::abs(problem.solution(i));
                  const double scaled = std::abs(correction(i)) / weight;
                  if (!std::isfinite(problem.solution(i)) ||
                      !std::isfinite(residual(i)) || !std::isfinite(weight) ||
                      weight <= 0. || !std::isfinite(scaled) ||
                      ((problem.component_dofs_[0].is_element(i) ||
                        problem.component_dofs_[2].is_element(i)) &&
                       problem.solution(i) <= 0.))
                    invalid = 1;
                  else
                    defect = std::max(defect, scaled);
                }
              defect = Utilities::MPI::max(defect, problem.mpi_communicator_);
              AssertThrow(
                Utilities::MPI::max(invalid, problem.mpi_communicator_) == 0 &&
                  defect < 1.,
                ExcMessage(
                  "Stationary HDG residual needs a correction exceeding final tolerance weights: " +
                  std::to_string(defect)));
              output_stationary(problem, cycle);
              if (problem.this_mpi_process == 0)
                {
                  const std::string folder = problem.output_directory.empty() ?
                                               "." :
                                               problem.output_directory;
                  std::ofstream     statistics(folder +
                                           "/stationary_statistics.json");
                  statistics
                    << std::setprecision(17)
                    << "{\n  \"model\": \"steady HDG\",\n  \"time\": "
                    << current_time << ",\n  \"ida_steps\": " << steps
                    << ",\n  \"derivative_norm\": 0,\n  \"weighted_newton_defect\": "
                    << defect << "\n}\n";
                  AssertThrow(statistics.good(),
                              ExcMessage(
                                "Cannot write stationary statistics."));
                }
              continue;
            }

          unsigned int output_number        = 0;
          bool         final_output_written = false;

          ida.output_step = [&problem,
                             &ida,
                             &ida_errors,
                             initial_output_time,
                             final_output_time,
                             &next_output_time,
                             &output_number,
                             &final_output_written,
                             &initial_output_seen,
                             &ida_initialization_wall,
                             &solve_start,
                             &integration_start,
                             &first_interval_wall](
                              const double      callback_time,
                              const VectorType &sol,
                              const VectorType & /*ydot*/,
                              const unsigned int step_number) {
            ida_errors.check();
            RampSource::after_solve(ida);
            // deal.II IDA reports zero for its initial output callback,
            // including when integration starts at a nonzero time.
            const double t =
              step_number == 0 ? initial_output_time : callback_time;
            if (step_number == 0)
              {
                ida_initialization_wall =
                  std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - solve_start)
                    .count();
                initial_output_seen = true;
              }
            else if (first_interval_wall < 0.)
              first_interval_wall =
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              integration_start)
                  .count();
            problem.time = t;
            // IDA accumulates its callback times by repeated addition.
            // Use the callback grid only to select outputs; retain the
            // physical callback time for every recorded value and timestamp.
            const double schedule_time =
              initial_output_time +
              step_number * problem.ida_parameters.output_period;
            const double tolerance =
              16.0 * std::numeric_limits<double>::epsilon() *
              std::max(std::abs(schedule_time), std::abs(next_output_time));
            const double final_time_tolerance =
              16.0 * std::numeric_limits<double>::epsilon() *
              std::max(std::abs(t), std::abs(final_output_time));
            problem.poll_ida_internal_step(ida,
                                           /*resync_time=*/true,
                                           /*require_consecutive_steps=*/true);
            problem.print_current_statistics(ida, t);
            const bool at_final_time =
              t >= final_output_time ||
              final_output_time - t <= final_time_tolerance;
            const bool nominal_final_time =
              schedule_time >= final_output_time ||
              final_output_time - schedule_time <= final_time_tolerance;
            if (problem.time_interval > 0.0 &&
                (final_output_written ||
                 (nominal_final_time && !at_final_time)))
              return;
            const bool output_due =
              problem.time_interval == 0.0 ||
              schedule_time >= next_output_time ||
              next_output_time - schedule_time <= tolerance || at_final_time;
            if (!output_due)
              return;

            // IDA internal steps / nonlinear iterations taken since the
            // previous *written* output -> ida_step_history.csv.  Recorded
            // after the output_due check so each row spans one output
            // interval set by "Output time interval".
            problem.record_ida_step(ida, t);

            problem.compute_pressure(sol, problem.pressure_);
            problem.output_results(sol, problem.pressure_, output_number++);
            // HDG_IDA_Vessel_*.csv + mass_conservation.csv
            problem.write_output_step(t, sol);
            problem.report_flow_regime(sol, t);
            final_output_written = at_final_time;

            if (problem.time_interval > 0.0)
              {
                const long double elapsed =
                  (static_cast<long double>(schedule_time + tolerance) -
                   initial_output_time) /
                  problem.time_interval;
                const long double intervals = std::floor(elapsed) + 1.0L;
                next_output_time            = static_cast<double>(
                  static_cast<long double>(initial_output_time) +
                  intervals * problem.time_interval);
                if (next_output_time <= schedule_time)
                  next_output_time =
                    std::nextafter(schedule_time,
                                   std::numeric_limits<double>::infinity());
              }
            if (step_number == 0)
              integration_start = std::chrono::steady_clock::now();
          };

          problem.time = problem.ida_parameters.initial_time;
          solve_start  = std::chrono::steady_clock::now();
          ida.solve_dae(problem.solution, problem.solution_dot);
          ida_errors.check();
          RampSource::after_solve(ida);
          const double solve_wall_time =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          solve_start)
              .count();
          if (problem.this_mpi_process == 0)
            {
              const std::string  folder = problem.output_directory.empty() ?
                                            "." :
                                            problem.output_directory;
              std::ofstream      statistics(folder +
                                       "/initialization_statistics.json");
              const auto        &d = problem.initial_state_diagnostics();
              std::ostringstream differential_defect;
              if (native_pair_verified)
                differential_defect << std::setprecision(17)
                                    << d.differential_defect;
              else
                differential_defect << "null";

              statistics
                << std::setprecision(17) << "{\n  \"initial_state_source\": \""
                << problem.initial_state_source << "\",\n"
                << "  \"initial_condition_policy\": \""
                << problem.initial_condition_policy << "\",\n"
                << "  \"preparation_wall_s\": " << preparation_wall << ",\n"
                << "  \"ida_setup_and_ic_wall_s\": " << ida_initialization_wall
                << ",\n"
                << "  \"first_ida_output_interval_wall_s\": "
                << first_interval_wall << ",\n"
                << "  \"ic_residual_calls\": " << ic_residual_calls << ",\n"
                << "  \"ic_jacobian_setups\": " << ic_jacobian_setups << ",\n"
                << "  \"ic_linear_solves\": " << ic_linear_solves << ",\n"
                << "  \"trace_iterations\": " << d.trace_iterations << ",\n"
                << "  \"largest_trace_block\": " << d.largest_trace_block
                << ",\n"
                << "  \"algebraic_defect\": " << d.algebraic_defect << ",\n"
                << "  \"differential_defect\": " << differential_defect.str()
                << ",\n  \"native_pair_verified\": "
                << (native_pair_verified ? "true" : "false")
                << ",\n  \"tolerance_ramp_enabled\": "
                << (tolerance_ramp_.enabled ? "true" : "false");
              if (tolerance_ramp_.enabled)
                {
                  const auto &r = RampSource::diagnostics(ida);
                  statistics
                    << ",\n  \"ramp_registrations\": " << r.registrations
                    << ",\n  \"ramp_weight_evaluations\": "
                    << r.weight_evaluations
                    << ",\n  \"ramp_stop_arms\": " << r.stop_arms
                    << ",\n  \"ramp_stop_releases\": " << r.stop_releases
                    << ",\n  \"ramp_last_relative_factor\": "
                    << r.last_scales.relative
                    << ",\n  \"ramp_last_absolute_factor\": "
                    << r.last_scales.absolute;
                }
              statistics << "\n}\n";
              AssertThrow(statistics.good(),
                          ExcMessage(
                            "Cannot write initialization statistics."));
            }
          problem.finalize_output(ida, solve_wall_time);
          problem.compute_pressure(problem.solution, problem.pressure_);
          problem.compute_errors(cycle);
        }
      problem.write_scaling_timings();
    }

  private:
    static void
    output_stationary(Problem &problem, const unsigned int cycle)
    {
      problem.compute_pressure(problem.solution, problem.pressure_);
      problem.open_csv_files();
      problem.output_results(problem.solution, problem.pressure_, cycle);
      problem.write_csv_row(problem.time, problem.solution);
      problem.report_flow_regime(problem.solution, problem.time);
      problem.close_csv_files();
    }

    std::string   run_mode_ = "transient";
    ToleranceRamp tolerance_ramp_;
    long int      maximum_ida_steps_per_output_ = 500;
    double        maximum_step_size_            = 0.;
  };
} // namespace MetricFlowX

int
main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      /* --------------------------- 1. Locate parameter file -----------------
       */
      std::string par_name;
      if (argc > 1)
        par_name = argv[1]; // first CLI argument
      else
        par_name = "parameters.prm"; // fallback default

      /* ---------------------- 2. Initialise deal.II logging -----------------
       */
      dealii::deallog.depth_console(1);

      /* ------------------------- 3. Set up the problem ----------------------
       */
      BloodFlowSystem<1, 3> problem; // 1-dim geometry embedded in \mathbb{R}^3
      BloodFlowIDARunner<1, 3> runner(problem);
      problem.initialize_params(par_name);
      runner.run(problem);

      /* ------------------------ 4. Normal program exit ----------------------
       */
      return 0;
    }

  /* --------------------- 5. Dedicated exception catchers ------------------ */
  catch (const std::invalid_argument &theta_range) // parameter-specific errors
    {
      std::cerr << '\n'
                << "----------------------------------------------------\n"
                << "Invalid parameter: \n"
                << theta_range.what() << '\n'
                << "Aborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }
  catch (const std::exception &exc) // general deal.II/runtime errors
    {
      std::cerr << '\n'
                << "----------------------------------------------------\n"
                << "Exception on processing: \n"
                << exc.what() << '\n'
                << "Aborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }
  catch (...) // fallback
    {
      std::cerr << '\n'
                << "----------------------------------------------------\n"
                << "Unknown exception!\n"
                << "Aborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }
}
