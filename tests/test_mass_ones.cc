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

// Test length of domain using mass matrix.
//
// solution = 1.0 sets every FE component to 1, not just area: per_cell_mass
// carries mass entries for BOTH cell components (0 = area, 1 = velocity --
// see build_per_cell_mass_inv(), which explicitly skips components >= 2
// since the trace DOFs carry no mass term). Area and velocity share the
// same FE_DGQ basis/degree, so each block independently integrates to the
// domain length L. With both components set to 1, sum_K v_K^T M_K v_K
// therefore evaluates to 2L, not L -- that's what "0.241370 + 0.241370" in
// the output below refers to: the area contribution and the velocity
// contribution, each equal to L, summed. This is a deliberately different
// check from the domain-length test that isolates component 0 only (which
// verifies the area block alone integrates to L); this one exercises both
// mass-carrying components at once via the cheaper solution = 1.0 shortcut.
//


#include <deal.II/base/mpi.h>

#include <deal.II/lac/vector.h>

#include "metric_flow_system.h"
#include "tests.h"
#include "vtk_utils.h"

using namespace dealii;

void
test()
{
  MetricFlowSystem<1, 3> problem;
  problem.initialize_params(PRM_DIR "constant.prm");
  // initialize_params() resets deallog depth according to the parameter file.
  deallog.depth_console(10);
  deallog.depth_file(10);

  problem.create_triangulation();

  problem.setup_system();
  problem.initialize_terminal_capacitors();

  problem
    .build_per_cell_mass_inv(); // fills per_cell_mass and per_cell_mass_inv

  problem.solution = 1.0;

  // --- Compute L = sum_K v_K^T M_K v_K using per-cell mass blocks ----------
  // Locally owned cells only -- per_cell_mass has no entries for ghost
  // cells -- then reduced across ranks.

  double             L_local = 0.0;
  const unsigned int n_dofs  = problem.fe->n_dofs_per_cell();
  Vector<double>     local_v(n_dofs), local_Mv(n_dofs);

  for (const auto &cell : problem.dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      std::vector<types::global_dof_index> ldofs(n_dofs);
      cell->get_dof_indices(ldofs);

      for (unsigned int i = 0; i < n_dofs; ++i)
        local_v(i) = problem.solution(ldofs[i]); // = 1.0 for all cell DOFs

      problem.per_cell_mass[cell->active_cell_index()].vmult(local_Mv, local_v);

      for (unsigned int i = 0; i < n_dofs; ++i)
        L_local += local_v(i) * local_Mv(i);
    }

  const double L = Utilities::MPI::sum(L_local, problem.mpi_communicator);

  if (Utilities::MPI::this_mpi_process(problem.mpi_communicator) == 0)
    deallog << "0.241370 + 0.241370 = " << L << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, numbers::invalid_unsigned_int);

  initlog();
  test();
}