#include <deal.II/base/mpi.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <metric_flow_x/blood_flow_system.h>

#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  initlog();
  BloodFlowSystem<1, 3> flow;
  flow.initialize_params(PRM_DIR "constant.prm");
  unsigned int calls          = 0;
  unsigned int expected_cells = 0;
  flow.setup([&](const auto &coarse_mesh, const unsigned int refinement) {
    ++calls;
    AssertThrow(flow.triangulation().n_global_active_cells() == 0,
                ExcInternalError());
    expected_cells = coarse_mesh.n_active_cells() * (1u << refinement);
    Triangulation<1, 3> prepared;
    prepared.copy_triangulation(coarse_mesh);
    prepared.refine_global(refinement);
    GridTools::partition_triangulation_zorder(
      Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), prepared, false);
    const auto description = TriangulationDescription::Utilities::
      create_description_from_triangulation(prepared, MPI_COMM_WORLD);
    flow.triangulation().create_triangulation(description);
  });
  AssertThrow(calls == 1, ExcInternalError());
  AssertThrow(flow.triangulation().n_global_active_cells() == expected_cells,
              ExcInternalError());
  AssertThrow(flow.dof_handler().n_dofs() > 0, ExcInternalError());
  AssertThrow(flow.vessel_properties(0).a0 > 0., ExcInternalError());
  unsigned int inlet = 0, outlet = 0;
  for (const auto &cell : flow.triangulation().active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto face : cell->face_indices())
        if (cell->face(face)->at_boundary())
          {
            inlet += cell->face(face)->boundary_id() == 0;
            outlet += cell->face(face)->boundary_id() != 0;
          }
  AssertThrow(Utilities::MPI::sum(inlet, MPI_COMM_WORLD) == 1,
              ExcInternalError());
  AssertThrow(Utilities::MPI::sum(outlet, MPI_COMM_WORLD) == 1,
              ExcInternalError());
  auto state      = flow.make_state();
  auto derivative = flow.make_state();
  auto residual   = flow.make_state();
  flow.initialize_state(state, 0.);
  flow.initialize_state_derivative(derivative, 0.);
  flow.assemble_residual(0., state, derivative, residual);
  flow.assemble_state_jacobian(0., state, derivative);
  AssertThrow(std::isfinite(residual.l2_norm()), ExcInternalError());
  deallog.depth_file(10);
  deallog << "MetricFlowX mesh initializer: PASSED" << std::endl;
}
