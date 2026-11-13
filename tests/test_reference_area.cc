// SPDX-License-Identifier: MIT
// The public wall reference area is the tapered a_d(x) of the tube law, both
// pointwise and as a native vector, also after a checkpoint restore. a0 does
// not enter the flow equations.
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <metric_flow_x/blood_flow_system.h>

#include <filesystem>
#include <fstream>

#include "tests.h"

using namespace MetricFlowX;
using Flow = BloodFlowSystem<1, 3>;

constexpr double length = .1, r_in = .004, r_out = .002;

void
make_mesh(Flow &flow)
{
  Triangulation<1, 3>      mesh;
  std::vector<Point<3>>    vertices{{0., 0., 0.}, {length, 0., 0.}};
  std::vector<CellData<1>> cells(1);
  cells[0].vertices[0] = 0;
  cells[0].vertices[1] = 1;
  mesh.create_triangulation(vertices, cells, SubCellData());
  for (const auto &cell : mesh.active_cell_iterators())
    for (const auto face : cell->face_indices())
      cell->face(face)->set_boundary_id(face);
  mesh.refine_global(3);
  GridTools::partition_triangulation_zorder(
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), mesh, false);
  flow.triangulation().create_triangulation(
    TriangulationDescription::Utilities::create_description_from_triangulation(
      mesh, MPI_COMM_WORLD));
}

double
expected_area(const Point<3> &x)
{
  const double r = r_in + x[0] / length * (r_out - r_in);
  return numbers::PI * r * r;
}

// Collects reference-area entries on locally owned DoFs.
std::vector<double>
owned_values(const Flow &flow, const VectorType &vector)
{
  std::vector<double> result;
  for (const auto i : flow.locally_owned_dofs())
    result.push_back(vector[i]);
  return result;
}

void
test()
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  if (rank == 0)
    {
      std::filesystem::remove_all("reference-area-checkpoint");
      std::ofstream prm("reference-area.prm");
      prm << "subsection MetricFlowSystem<1, 3>\n"
          << " set Verbosity (console depth) = 0\n"
          << " set Wall model = Square root\n"
          << " set Outlet boundary condition type = RCR\n"
          << " set Finite element degree = 2\n"
          << " set Number of global refinement = 0\nend\n";
    }
  MPI_Barrier(comm);

  std::vector<double> reference;
  {
    ParameterAcceptor::clear();
    Flow flow(comm);
    flow.initialize_params("reference-area.prm", "");
    make_mesh(flow);
    Flow::NetworkPhysics physics;
    auto                &v = physics.vessels[0];
    v.r_d                  = .003;
    v.r_in                 = r_in;
    v.r_out                = r_out;
    v.a_d                  = numbers::PI * v.r_d * v.r_d;
    v.a0                   = 2. * v.a_d; // deliberately unrelated to a_d(x)
    v.E                    = 5.e5;
    v.h_wall               = .0003;
    v.L                    = length;
    v.p_d                  = 1000.;
    v.dynamic_viscosity    = .0035;
    physics.terminals[1]   = {1.e7, 1.e8, 1.e-9, 100., "capacitive outlet"};
    flow.setup(physics);

    // Pointwise: the tapered area, at which the tube law gives p0 + p_d.
    for (const auto &cell : flow.dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        for (const auto vertex : cell->vertex_indices())
          {
            const auto   x    = cell->vertex(vertex);
            const double area = flow.reference_area(cell, x);
            AssertThrow(std::abs(area - expected_area(x)) <=
                          1.e-12 * expected_area(x),
                        ExcMessage("Reference area is not the tapered a_d."));
            AssertThrow(std::abs(flow.compute_pressure_value(
                                   area, cell->material_id(), area) -
                                 v.p0 - v.p_d) < 1.e-9,
                        ExcMessage("Tube law is not at p0 + p_d."));
          }

    // Native vector: area and area-trace components only, matching the rest
    // state that seeds the initial solution.
    auto area = flow.make_state(), rest = flow.make_state();
    flow.interpolate_reference_area(area);
    flow.compute_initial_solution(rest, 0.);
    for (const auto c : {1u, 3u, 4u})
      for (const auto i : flow.component_dofs_[c])
        AssertThrow(area[i] == 0.,
                    ExcMessage("Non-area component is not zero."));
    for (const auto c : {0u, 2u})
      for (const auto i : flow.component_dofs_[c])
        AssertThrow(area[i] > 0. && area[i] == rest[i],
                    ExcMessage("Reference area differs from rest area."));
    for (const auto &cell : flow.dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          std::vector<types::global_dof_index> dofs(
            cell->get_fe().n_dofs_per_cell());
          cell->get_dof_indices(dofs);
          for (const auto face : cell->face_indices())
            {
              const auto x = cell->vertex(face);
              AssertThrow(std::abs(
                            area[flow.face_trace_dofs(dofs, face).first] -
                            expected_area(x)) <= 1.e-12 * expected_area(x),
                          ExcMessage("Area trace is not a_d at its face."));
            }
        }

    // a0 does not enter the residual.
    auto rate = flow.make_state(), residual = flow.make_state(),
         changed = flow.make_state();
    flow.assemble_residual(0., rest, rate, residual);
    flow.vessel_map.at(0).a0 = .5 * v.a_d;
    flow.assemble_residual(0., rest, rate, changed);
    changed -= residual;
    AssertThrow(changed.linfty_norm() == 0.,
                ExcMessage("a0 changed the flow residual."));
    flow.vessel_map.at(0).a0 = v.a0;

    reference = owned_values(flow, area);
    flow.save_checkpoint("reference-area-checkpoint", 0., rest, rate);
  }

  // A restored object reports the same reference area without input files.
  if (rank == 0)
    std::filesystem::remove("reference-area.prm");
  MPI_Barrier(comm);
  {
    ParameterAcceptor::clear();
    Flow restored(comm);
    restored.load_checkpoint("reference-area-checkpoint");
    auto area = restored.make_state();
    restored.interpolate_reference_area(area);
    AssertThrow(owned_values(restored, area) == reference,
                ExcMessage("Restored reference area changed."));
  }
  deallog.depth_file(10);
  deallog << "Wall reference area: PASSED" << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  test();
}
