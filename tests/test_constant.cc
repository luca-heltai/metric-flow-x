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

// Test residual assembly for a simple constant solution,
// with a single vessel and non zero terminal pressure


#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <metric_flow_x/blood_flow_system.h>
#include <metric_flow_x/io/vtk_utils.h>

#include <cmath>
#include <iostream>

#include "tests.h"

using namespace dealii;

void
test()
{
  BloodFlowSystem<1, 3> problem;
  problem.initialize_params(PRM_DIR "constant.prm");

  // initialize_params() resets deallog depth according to the parameter file.
  // Re-enable logging for the regression test.
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

  VectorType ydot(problem.locally_owned_dofs, problem.mpi_communicator);
  ydot = 0.0;

  VectorType residual(problem.locally_owned_dofs, problem.mpi_communicator);


  problem.assemble_residual(problem.time, problem.solution, ydot, residual);


  double cell_sq_local  = 0.0;
  double trace_sq_local = 0.0;
  double rcr_sq_local   = 0.0;

  for (const auto i : problem.cell_dofs_owned)
    cell_sq_local += residual(i) * residual(i);

  for (const auto i : problem.trace_dofs_owned)
    trace_sq_local += residual(i) * residual(i);

  for (const auto i : problem.rcr_dofs_owned)
    rcr_sq_local += residual(i) * residual(i);

  const double cell_sq =
    Utilities::MPI::sum(cell_sq_local, problem.mpi_communicator);

  const double trace_sq =
    Utilities::MPI::sum(trace_sq_local, problem.mpi_communicator);

  // Note: Pc is initialized to p_d rather than the steady-state RCR value.
  // Consequently, the initial RCR residual is expected to be nonzero.
  const double rcr_sq =
    Utilities::MPI::sum(rcr_sq_local, problem.mpi_communicator);

  if (Utilities::MPI::this_mpi_process(problem.mpi_communicator) == 0)
    {
      deallog << "Cell residual  = " << std::sqrt(cell_sq) << std::endl;

      deallog << "Trace residual = " << std::sqrt(trace_sq) << std::endl;

      deallog << "RCR residual   = " << std::sqrt(rcr_sq) << std::endl;

      deallog << "Total residual = " << std::sqrt(cell_sq + trace_sq + rcr_sq)
              << std::endl;
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}