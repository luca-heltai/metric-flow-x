// ---------------------------------------------------------------------
//
// Test trace-junction residual vs trace-junction Jacobian block.
//
// Assembles only assemble_trace_junction_equations and
// assemble_jacobian_trace_junction_block, then checks every entry
// (any row, any column) with central finite differences.
//
//
// Output:
//   L2_error       — ||J_an - J_fd||_F over the full n_total × n_total matrix
//   worst_row      — global row index with the largest per-row L2 error
//   row_L2_error   — ||J_an[worst_row,:] - J_fd[worst_row,:]||_2
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <cmath>
#include <iomanip>
#include <map>

#include "metric_flow_system.h"
#include "tests.h"
#include "vtk_utils.h"

using namespace dealii;

void
test()
{
  MetricFlowSystem<1, 3> problem;
  problem.initialize_params(PRM_DIR "aortic.prm");

  // initialize_params() resets deallog depth according to the parameter file.
  deallog.depth_console(10);
  deallog.depth_file(10);

  problem.create_triangulation();
  problem.setup_system();

  problem.initialize_terminal_capacitors();
  problem.build_per_cell_mass_inv();

  problem.compute_initial_solution(problem.solution,
                                   problem.ida_parameters.initial_time);

  problem.initialize_trace_unknowns(problem.solution,
                                    problem.ida_parameters.initial_time);

  problem.time = problem.ida_parameters.initial_time;

  const double t = problem.time;

  // ---- analytic Jacobian, assembled at the current (unperturbed) state ----
  // Both blocks read trace/Pc values through the ghosted vector, exactly as
  // assemble_jacobian() does, so the ghosted vectors must be refreshed from
  // problem.solution first.
  problem.update_ghosted_vectors(problem.solution);

  problem.jacobian_matrix = 0.0;
  problem.assemble_jacobian_trace_junction_block(problem.y_relevant);
  problem.jacobian_matrix.compress(VectorOperation::add);

  const double eps = 1e-8;

  VectorType yp(problem.locally_owned_dofs, problem.mpi_communicator);
  VectorType ym(problem.locally_owned_dofs, problem.mpi_communicator);

  VectorType Fp(problem.locally_owned_dofs, problem.mpi_communicator);
  VectorType Fm(problem.locally_owned_dofs, problem.mpi_communicator);

  VectorType ej(problem.locally_owned_dofs, problem.mpi_communicator);
  VectorType Jcol(problem.locally_owned_dofs, problem.mpi_communicator);

  double       l2_sq_local = 0.0;
  const double h           = eps;

  // Per-row running sum of squared errors over the columns swept so far --
  // only for rows this rank owns, so it can be accumulated locally with no
  // extra communication and reduced to the single global worst row at the
  // very end.
  std::map<types::global_dof_index, double> row_sq_local;
  for (const auto i : problem.locally_owned_dofs)
    row_sq_local[i] = 0.0;

  // Collective sweep over every global column: all ranks must call
  // update_ghosted_vectors / assemble_trace_boundary_equations / vmult the
  // same number of times, in the same order, regardless of who owns j.
  for (types::global_dof_index j = 0; j < problem.n_total_dofs; ++j)
    {
      // ---- +h: perturb, ghost, assemble, all before touching -h ----------
      yp = problem.solution;
      if (problem.locally_owned_dofs.is_element(j))
        yp(j) = yp(j) + h;
      yp.compress(VectorOperation::insert);
      problem.update_ghosted_vectors(yp);

      Fp = 0.0;
      problem.assemble_trace_junction_equations(problem.y_relevant, Fp);
      Fp.compress(VectorOperation::add);

      // ---- -h: perturb, ghost, assemble ------------------------------------
      ym = problem.solution;
      if (problem.locally_owned_dofs.is_element(j))
        ym(j) = ym(j) - h;
      ym.compress(VectorOperation::insert);
      problem.update_ghosted_vectors(ym);

      Fm = 0.0;
      problem.assemble_trace_junction_equations(problem.y_relevant, Fm);
      Fm.compress(VectorOperation::add);

      // ---- j-th unit vector, then the analytic column via vmult -----------
      ej = 0.0;
      if (problem.locally_owned_dofs.is_element(j))
        ej(j) = 1.0;
      ej.compress(VectorOperation::insert);

      problem.jacobian_matrix.vmult(Jcol, ej);

      // Unlike the cell-block test, this one is documented to check the
      // FULL n_total × n_total matrix, not just the trace rows --
      // assemble_trace_junction_equations only ever writes into trace rows
      // in practice, so restricting to trace_dofs_owned (as this loop did)
      // would happen to give the same L2_error whenever the code is
      // correct, but it would also silently hide a bug where
      // assemble_jacobian_trace_junction_block leaked nonzero entries into
      // cell or Pc rows -- exactly the kind of thing this test exists to
      // catch. So every locally owned row is checked, not just trace ones.
      for (const auto i : problem.locally_owned_dofs)
        {
          const double fd  = (Fp(i) - Fm(i)) / (2.0 * h);
          const double err = Jcol(i) - fd;

          l2_sq_local += err * err;
          row_sq_local[i] += err * err;
        }
    }

  const double l2_sq =
    Utilities::MPI::sum(l2_sq_local, problem.mpi_communicator);

  // Find the globally worst row via MPI_MAXLOC: each rank first finds its
  // own worst among the rows it owns, then one small reduction picks the
  // global winner and which rank (hence which global index) it belongs to.
  // (MPI_DOUBLE_INT stores the index as a plain int, which is fine here --
  // dof counts are nowhere near INT_MAX -- rather than risking a deal.II
  // convenience wrapper that may not be instantiated for this pair type in
  // every build.)
  struct
  {
    double value;
    int    index;
  } local_worst{-1.0, -1}, global_worst{};

  for (const auto &[row, sq] : row_sq_local)
    if (sq > local_worst.value)
      local_worst = {sq, static_cast<int>(row)};

  MPI_Allreduce(&local_worst,
                &global_worst,
                1,
                MPI_DOUBLE_INT,
                MPI_MAXLOC,
                problem.mpi_communicator);

  const types::global_dof_index worst_row =
    static_cast<types::global_dof_index>(global_worst.index);
  const double row_l2_error = std::sqrt(global_worst.value);

  if (Utilities::MPI::this_mpi_process(problem.mpi_communicator) == 0)
    {
      deallog << "L2_error = " << std::scientific << std::setprecision(6)
              << std::sqrt(l2_sq) << std::endl;
      deallog << "worst_row = " << worst_row << std::endl;
      deallog << "row_L2_error = " << std::scientific << std::setprecision(6)
              << row_l2_error << std::endl;
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}