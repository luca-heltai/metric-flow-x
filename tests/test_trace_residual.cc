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
// Test: after initialize_trace_unknowns(), the trace residual
// F_trace(y_cell^0, y_hat) must be zero to machine precision.
// This verifies that the Newton solve converged correctly and that
// the initial condition is consistent before IDA even starts.
//
// Parallel: problem.triangulation_ is a parallel::fullydistributed
// triangulation_, so mesh loading goes through problem.create_triangulation()
// rather than attaching GridIn by hand. assemble_trace_*_equations() read
// from the ghosted y_relevant member (populated via update_ghosted_vectors),
// not directly from solution -- interior/junction equations touch DOFs on
// neighboring cells that may be owned by another rank. F_before/F_after are
// non-ghosted VectorType, matching every other write-only vector in the
// class, and the residual norm is accumulated over trace_dofs_owned (the
// locally owned trace rows) then reduced across ranks.
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
  problem.initialize_params(PRM_DIR "multi_vessel.prm");
  deallog.depth_console(10);
  deallog.depth_file(10);

  problem.create_triangulation();

  problem.setup_system();
  problem.initialize_terminal_capacitors();
  problem.build_per_cell_mass_inv();

  // Step 1: set cell DOFs to equilibrium (A=a_d, U=0); trace DOFs to a_d/0
  problem.compute_initial_solution(problem.solution, problem.time);

  // Step 2: check trace residual BEFORE initialize_trace_unknowns
  {
    problem.update_ghosted_vectors(problem.solution);

    VectorType F_before(problem.locally_owned_dofs_, problem.mpi_communicator_);
    F_before = 0.0;
    problem.assemble_trace_interior_equations(problem.y_relevant, F_before);
    problem.assemble_trace_boundary_equations(problem.time,
                                              problem.y_relevant,
                                              F_before);
    problem.assemble_trace_junction_equations(problem.y_relevant, F_before);
    F_before.compress(VectorOperation::add);

    double norm_before_local = 0.0;
    for (const auto i : problem.trace_dofs_owned)
      norm_before_local += F_before(i) * F_before(i);

    const double norm_before = std::sqrt(
      Utilities::MPI::sum(norm_before_local, problem.mpi_communicator_));

    if (Utilities::MPI::this_mpi_process(problem.mpi_communicator_) == 0)
      deallog << "Trace residual BEFORE initialize_trace_unknowns: "
              << norm_before << std::endl;
  }

  // Step 3: run Newton to find consistent trace unknowns
  problem.initialize_trace_unknowns(problem.solution, problem.time);

  // Step 4: check trace residual AFTER initialize_trace_unknowns
  {
    problem.update_ghosted_vectors(problem.solution);

    VectorType F_after(problem.locally_owned_dofs_, problem.mpi_communicator_);
    F_after = 0.0;
    problem.assemble_trace_interior_equations(problem.y_relevant, F_after);
    problem.assemble_trace_boundary_equations(problem.time,
                                              problem.y_relevant,
                                              F_after);
    problem.assemble_trace_junction_equations(problem.y_relevant, F_after);
    F_after.compress(VectorOperation::add);

    double norm_after_local = 0.0;
    for (const auto i : problem.trace_dofs_owned)
      norm_after_local += F_after(i) * F_after(i);

    const double norm_after = std::sqrt(
      Utilities::MPI::sum(norm_after_local, problem.mpi_communicator_));

    if (Utilities::MPI::this_mpi_process(problem.mpi_communicator_) == 0)
      deallog << "Trace residual AFTER initialize_trace_unknowns:  "
              << norm_after << std::endl;

    AssertThrow(norm_after < 1e-10,
                ExcMessage("initialize_trace_unknowns did not converge: "
                           "trace residual is not zero."));
  }

  if (Utilities::MPI::this_mpi_process(problem.mpi_communicator_) == 0)
    deallog << "initialize_trace_unknowns PASSED." << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, numbers::invalid_unsigned_int);

  initlog();
  test();
}