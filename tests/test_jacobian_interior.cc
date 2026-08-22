
// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the bare-dealii-app application, based on the
// deal.II library.
//
// The bare-dealii-app application is free software; you can use it,
// redistribute it, and/or modify it under the terms of the Apache-2.0 License
// WITH LLVM-exception as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md
// at the top level of the bare-dealii-app distribution.
//
// ---------------------------------------------------------------------
//
// Test trace-interior residual vs trace-interior Jacobian block.
//
// Assembles only assemble_trace_interior_equations and
// assemble_jacobian_trace_interior_block, then checks every entry
// (any row, any column) with central finite differences.
//
//
// Output:
//   L2_error       — ||J_an - J_fd||_F over the full n_total × n_total matrix
//   worst_row      — global row index with the largest per-row L2 error
//   row_L2_error   — ||J_an[worst_row,:] - J_fd[worst_row,:]||_2
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <metric_flow_x/blood_flow_system.h>
#include <metric_flow_x/io/vtk_utils.h>

#include <cmath>
#include <iomanip>

#include "tests.h"

using namespace dealii;

void
test()
{
  BloodFlowSystem<1, 3> problem;
  problem.initialize_params(PRM_DIR "constant.prm");

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
  problem.assemble_jacobian_trace_interior_block(problem.y_relevant);
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
      problem.assemble_trace_interior_equations(problem.y_relevant, Fp);
      Fp.compress(VectorOperation::add);

      // ---- -h: perturb, ghost, assemble ------------------------------------
      ym = problem.solution;
      if (problem.locally_owned_dofs.is_element(j))
        ym(j) = ym(j) - h;
      ym.compress(VectorOperation::insert);
      problem.update_ghosted_vectors(ym);

      Fm = 0.0;
      problem.assemble_trace_interior_equations(problem.y_relevant, Fm);
      Fm.compress(VectorOperation::add);

      // ---- j-th unit vector, then the analytic column via vmult -----------
      ej = 0.0;
      if (problem.locally_owned_dofs.is_element(j))
        ej(j) = 1.0;
      ej.compress(VectorOperation::insert);

      problem.jacobian_matrix.vmult(Jcol, ej);

      for (const auto i : problem.locally_owned_dofs)
        {
          if (!problem.trace_dofs_owned.is_element(i))
            continue;

          const double fd  = (Fp(i) - Fm(i)) / (2.0 * h);
          const double err = Jcol(i) - fd;

          l2_sq_local += err * err;
        }
    }

  const double l2_sq =
    Utilities::MPI::sum(l2_sq_local, problem.mpi_communicator);

  if (Utilities::MPI::this_mpi_process(problem.mpi_communicator) == 0)
    deallog << "L2_error = " << std::scientific << std::setprecision(6)
            << std::sqrt(l2_sq) << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}