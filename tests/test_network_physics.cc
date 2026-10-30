// SPDX-License-Identifier: MIT
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <metric_flow_x/blood_flow_system.h>

#include "tests.h"

using namespace MetricFlowX;
using Flow = BloodFlowSystem<1, 3>;

Flow::NetworkPhysics
physics()
{
  Flow::NetworkPhysics result;
  for (unsigned int id = 0; id < 2; ++id)
    {
      auto &v = result.vessels[id];
      v.r_in  = .003 + .001 * id;
      v.r_out = v.r_in - .0005;
      v.r_d   = v.r_in;
      v.a0 = v.a_d        = numbers::PI * v.r_d * v.r_d;
      v.E                 = 5.e5;
      v.h_wall            = .0003;
      v.L                 = .1;
      v.dynamic_viscosity = .003 + .001 * id;
    }
  result.terminals = {{1, {0., 1.e8, 0., 0., "arterial"}},
                      {2, {0., 1.e8, 0., 0., "venous"}},
                      {3, {0., 1.e8, 0., 0., "reservoir"}}};
  result.terminal_connections.push_back({1, 2, 1.e9});
  return result;
}

void
mesh(Flow &flow, unsigned int refinements)
{
  Triangulation<1, 3>         prepared;
  const std::vector<Point<3>> points{{0., 0., 0.},
                                     {.1, 0., 0.},
                                     {0., 1., 0.},
                                     {.1, 1., 0.}};
  std::vector<CellData<1>>    cells(2);
  for (unsigned int id = 0; id < 2; ++id)
    {
      cells[id].vertices[0] = 2 * id;
      cells[id].vertices[1] = 2 * id + 1;
      cells[id].material_id = id;
    }
  prepared.create_triangulation(points, cells, SubCellData());
  for (const auto &cell : prepared.active_cell_iterators())
    for (const auto face : cell->face_indices())
      cell->face(face)->set_boundary_id(2 * cell->material_id() + face);
  prepared.refine_global(refinements);
  GridTools::partition_triangulation_zorder(
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), prepared, false);
  flow.triangulation().create_triangulation(
    TriangulationDescription::Utilities::create_description_from_triangulation(
      prepared, MPI_COMM_WORLD));
}

void
test()
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  const auto size = Utilities::MPI::n_mpi_processes(comm);
  if (rank == 0)
    {
      std::ofstream prm("network_physics.prm");
      prm << "subsection Functions\n set Inflow function = 2e-9*(1+t)\nend\n"
          << "subsection MetricFlowSystem<1, 3>\n"
          << " set Vtk file path for mesh input = unused_in_memory_input.vtk\n"
          << " set Wall model = Square root\n"
          << " set Outlet boundary condition type = RCR\n"
          << " set Number of global refinement = 0\n"
          << " set Finite element degree = 1\n"
          << " set Initial consistency tolerance = 1e-5\nend\n";
    }
  MPI_Barrier(comm);
  for (const unsigned int refinement : {0u, 2u})
    {
      ParameterAcceptor::clear();
      Flow flow(comm);
      flow.initialize_params("network_physics.prm");
      mesh(flow, refinement);
      const auto input    = physics();
      const auto packed   = Utilities::pack(input);
      const auto received = Utilities::unpack<Flow::NetworkPhysics>(packed);
      flow.setup(received);
      AssertThrow(received.terminals.at(2).label == "venous" &&
                    received.terminal_connections[0].boundary0 == 1 &&
                    received.terminal_connections[0].boundary1 == 2 &&
                    received.terminal_connections[0].resistance == 1.e9,
                  ExcMessage("Coefficient serialization lost terminal data."));
      AssertThrow(flow.triangulation().n_global_active_cells() ==
                    2u * (1u << refinement),
                  ExcInternalError());
      AssertThrow(flow.dof_handler().n_dofs() > 0, ExcInternalError());
      for (const auto &[id, expected] : input.vessels)
        {
          const auto &actual = flow.vessel_properties(id);
          AssertThrow(actual.r_in == expected.r_in &&
                        actual.r_out == expected.r_out &&
                        actual.a_d == expected.a_d &&
                        actual.a0 == expected.a0 &&
                        actual.r_d == expected.r_d && actual.E == expected.E &&
                        actual.h_wall == expected.h_wall &&
                        actual.p_d == expected.p_d &&
                        actual.p0 == expected.p0 && actual.L == expected.L &&
                        actual.dynamic_viscosity == expected.dynamic_viscosity,
                      ExcMessage("In-memory coefficients changed."));
        }
      auto y = flow.make_state(), rate = flow.make_state(),
           residual = flow.make_state();
      flow.initialize_consistent_state(y, rate, .17);
      flow.assemble_residual(.17, y, rate, residual);
      flow.assemble_state_jacobian(.17, y, rate);
      flow.assemble_derivative_jacobian(.17, y, rate);
      AssertThrow(std::isfinite(residual.l2_norm()) && rate.l2_norm() > 0.,
                  ExcMessage(
                    "Nonzero forcing did not reach the native equations."));
      AssertThrow(!std::ifstream("unused_in_memory_input.vtk"),
                  ExcMessage(
                    "The in-memory test unexpectedly has a mesh input file."));
    }

  for (unsigned int failure = 0; failure < 4; ++failure)
    {
      ParameterAcceptor::clear();
      Flow flow(comm);
      flow.initialize_params("network_physics.prm");
      mesh(flow, 0);
      auto input = physics();
      if (failure == 0 && rank == size - 1)
        input.vessels[0].a_d = size > 1 ? 2 * input.vessels[0].a_d : 0.;
      else if (failure == 1)
        input.terminals.erase(2);
      else if (failure == 2)
        input.terminal_connections[0].resistance = -1.;
      else if (failure == 3)
        input.vessels.erase(1);
      bool caught = false;
      try
        {
          flow.setup(input);
        }
      catch (const ExceptionBase &)
        {
          caught = true;
        }
      AssertThrow(Utilities::MPI::sum(caught ? 1u : 0u, comm) == size,
                  ExcMessage("Invalid input was not rejected collectively."));
    }
  deallog.depth_file(10);
  deallog << "MetricFlowX in-memory physics: PASSED" << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  test();
}
