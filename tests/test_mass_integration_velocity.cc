// ---------------------------------------------------------------------
// Test domain length via mass matrix integration using per-cell mass blocks.
// For a constant solution (A=1, U=1 on component 0 i.e. area DOFs only),
// sum_K  v^T M_K v  should equal the total length of the domain.
//
// Parallel: each rank only holds per_cell_mass entries for the cells it
// owns (indexed by cell->active_cell_index(), not CellId), so v is built
// only over locally owned cells and the partial sums are reduced with
// Utilities::MPI::sum across ranks.
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <deal.II/lac/petsc_vector.h>

#include <metric_flow_x/blood_flow_system.h>
#include <metric_flow_x/io/vtk_utils.h>

#include <cmath>
#include <iomanip>
#include <map>

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

  problem.create_triangulation();
  problem.setup_system();

  problem.initialize_terminal_capacitors();
  problem.build_per_cell_mass_inv();

  problem.compute_initial_solution(problem.solution,
                                   problem.ida_parameters.initial_time);

  problem.initialize_trace_unknowns(problem.solution,
                                    problem.ida_parameters.initial_time);

  problem.time = problem.ida_parameters.initial_time;

  // --- Build a unit vector on the area (component 0) cell DOFs only ---
  // With constant area = 1 and shape functions that integrate to 1 per cell,
  // sum_K v^T M_K v = integral_Omega 1 dx = total length.
  //
  // v is a distributed, non-ghosted VectorType (matches every other
  // write-only vector in the class, e.g. solution/residual_F): only owned
  // cells are touched, and every DOF of a locally owned cell is itself
  // locally owned (no cross-rank continuity constraints_ on the cell block),
  // so direct local writes are safe without ghost communication.
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
        {
          // Only set the area component (component 0) DOFs to 1
          if (problem.fe_->system_to_component_index(i).first == 0)
            v(ldofs[i]) = 1.0;
        }
    }
  v.compress(VectorOperation::insert);

  // --- Compute L = sum_K  v_K^T M_K v_K  (cell block only, locally owned
  //     cells only -- per_cell_mass has no entries for ghost cells), then
  //     reduce across ranks.
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

  if (Utilities::MPI::this_mpi_process(problem.mpi_communicator_) == 0)
    {
      deallog << "n_active_cells = "
              << problem.triangulation_.n_global_active_cells() << std::endl;
      deallog << "L: total domain length = " << L << std::endl;
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, numbers::invalid_unsigned_int);

  initlog();
  test();
}