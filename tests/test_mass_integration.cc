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
// Test length of domain using per-cell mass matrix blocks.
//
// Parallel: problem.triangulation_ is a parallel::fullydistributed
// triangulation_, so mesh loading, partitioning, material/boundary IDs and
// rcr_map all go through problem.create_triangulation() -- attaching
// GridIn directly to a fully-distributed triangulation_ (as a serial test
// might) does not work. per_cell_mass only holds entries for locally
// owned cells, indexed by cell->active_cell_index(), so both assembly
// loops are restricted to is_locally_owned() cells and the scalar results
// are reduced across ranks with Utilities::MPI::sum before comparing.
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <deal.II/lac/vector.h>

#include <metric_flow_x/blood_flow_system.h>
#include <metric_flow_x/io/vtk_utils.h>

#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

void
test()
{
  BloodFlowSystem<1, 3> problem;
  problem.initialize_params(PRM_DIR "constant.prm");
  // initialize_params() resets deallog depth according to the parameter file.
  deallog.depth_console(10);
  deallog.depth_file(10);
  // Handles VTK mesh read, cell/vertex data, material/boundary IDs,
  // rcr_map, partitioning across ranks, and n_global_refinements --
  // all internally, in a way that's safe for a fully-distributed
  // triangulation_. Do not reimplement this by hand.
  problem.create_triangulation();

  problem.setup_system();
  problem.initialize_terminal_capacitors();
  problem
    .build_per_cell_mass_inv(); // fills per_cell_mass and per_cell_mass_inv

  // Build v = 1 on area (component 0) cell DOFs, 0 elsewhere, over locally
  // owned cells only. With this choice:
  //   sum_K v_K^T M_K v_K = integral_Omega 1 dx = total length.
  //
  // v is a distributed, non-ghosted VectorType (matches every other
  // write-only vector in the class): every DOF of a locally owned cell is
  // itself locally owned, so direct local writes need no ghost exchange.
  VectorType v(problem.locally_owned_dofs_, problem.mpi_communicator_);
  v = 0.0;

  const unsigned int n_dofs = problem.fe_->n_dofs_per_cell();
  for (const auto &cell : problem.dof_handler_.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      std::vector<types::global_dof_index> ldofs(n_dofs);
      cell->get_dof_indices(ldofs);
      for (unsigned int i = 0; i < n_dofs; ++i)
        if (problem.fe_->system_to_component_index(i).first ==
            0) // area component
          v(ldofs[i]) = 1.0;
    }
  v.compress(VectorOperation::insert);

  // Compute L = sum_K  v_K^T M_K v_K  (cell DOFs only; trace block
  // untouched), locally owned cells only, then reduce across ranks.
  double         L_local = 0.0;
  Vector<double> local_v(n_dofs), local_Mv(n_dofs);

  for (const auto &cell : problem.dof_handler_.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      std::vector<types::global_dof_index> ldofs(n_dofs);
      cell->get_dof_indices(ldofs);

      for (unsigned int i = 0; i < n_dofs; ++i)
        local_v(i) = v(ldofs[i]);

      problem.per_cell_mass[cell->active_cell_index()].vmult(local_Mv, local_v);

      for (unsigned int i = 0; i < n_dofs; ++i)
        L_local += local_v(i) * local_Mv(i);
    }

  const double L = Utilities::MPI::sum(L_local, problem.mpi_communicator_);

  // Independent check: sum physical cell measures over locally owned cells,
  // then reduce across ranks.
  double L_geom_local = 0.0;
  for (const auto &cell : problem.triangulation_.active_cell_iterators())
    if (cell->is_locally_owned())
      L_geom_local += cell->measure();

  const double L_geom =
    Utilities::MPI::sum(L_geom_local, problem.mpi_communicator_);

  if (Utilities::MPI::this_mpi_process(problem.mpi_communicator_) == 0)
    {
      deallog << "L (mass matrix):   " << L << std::endl;
      deallog << "L (cell measures): " << L_geom << std::endl;
    }

  AssertThrow(std::abs(L - L_geom) / L_geom < 1e-10,
              ExcMessage("Mass matrix length integration FAILED."));

  if (Utilities::MPI::this_mpi_process(problem.mpi_communicator_) == 0)
    deallog << "Mass matrix length integration PASSED." << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, numbers::invalid_unsigned_int);

  initlog();
  test();
}