// Rebuilding the mesh/DoFs must replace face traversals, including on
// MPI ranks with no owned cells. The fixture is a single vessel with N-1
// ordinary interior faces, independent of partition and trace numbering.
#include <deal.II/base/mpi.h>

#include <metric_flow_x/blood_flow_system.h>

#include <cmath>

#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

void
test()
{
  BloodFlowSystem<1, 3> flow;
  flow.initialize_params(PRM_DIR "constant.prm");
  for (const unsigned int refinement : {0u, 2u, 1u})
    {
      flow.dof_handler_.clear();
      flow.triangulation_.clear();
      flow.n_global_refinements = refinement;
      flow.setup();
      const auto faces =
        Utilities::MPI::sum<unsigned int>(flow.interior_faces.size(),
                                          flow.mpi_communicator());
      AssertThrow(faces + 1 == flow.triangulation().n_global_active_cells(),
                  ExcInternalError());
      if (refinement == 0 &&
          Utilities::MPI::n_mpi_processes(flow.mpi_communicator()) > 1)
        AssertThrow(Utilities::MPI::sum<unsigned int>(
                      flow.triangulation().n_locally_owned_active_cells() == 0,
                      flow.mpi_communicator()) > 0,
                    ExcMessage(
                      "The unrefined fixture must exercise empty ranks."));
      for (const auto &face : flow.interior_faces)
        {
          AssertThrow(face.cell->is_locally_owned(), ExcInternalError());
          AssertThrow(flow.locally_owned_dofs().is_element(
                        face.trace.a_hat_dof),
                      ExcInternalError());
          AssertThrow(flow.locally_owned_dofs().is_element(
                        face.trace.u_hat_dof),
                      ExcInternalError());
          AssertThrow(!face.cell->face(face.face_no)->at_boundary(),
                      ExcInternalError());
        }
      AssertThrow(
        Utilities::MPI::sum<unsigned int>(flow.boundary_faces.size(),
                                          flow.mpi_communicator()) == 2,
        ExcMessage("The single vessel must retain its two endpoints."));
      for (const auto &face : flow.boundary_faces)
        {
          AssertThrow(face.cell->is_locally_owned(), ExcInternalError());
          AssertThrow(face.cell->face(face.face_no)->at_boundary(),
                      ExcInternalError());
          std::vector<types::global_dof_index> dofs(
            flow.fe_->n_dofs_per_cell());
          face.cell->get_dof_indices(dofs);
          const auto [area, velocity] =
            flow.face_trace_dofs(dofs, face.face_no);
          AssertThrow(face.a_hat_dof == area && face.u_hat_dof == velocity,
                      ExcMessage("Boundary trace indices must be refreshed."));
        }
      auto y = flow.make_state();
      flow.compute_initial_solution(y, 0.0);
      auto ydot     = flow.make_state();
      ydot          = 0.02;
      auto residual = flow.make_state();
      flow.assemble_residual(0.1, y, ydot, residual);
      flow.assemble_state_jacobian(0.1, y, ydot);
      AssertThrow(std::isfinite(residual.l2_norm()), ExcInternalError());
    }
  deallog.depth_file(10);
  deallog << "Interior face traversal replaced after mesh rebuild: PASSED"
          << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  mpi_initlog();
  test();
}
