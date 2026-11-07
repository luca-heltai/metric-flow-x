// SPDX-License-Identifier: MIT
// A rest tube has p_tube = p0 + p_d. Spatial external pressure must therefore
// appear at each area support point in the public nodal pressure output.
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <metric_flow_x/blood_flow_system.h>

#include <cmath>
#include <fstream>
#include <map>

#include "tests.h"

using namespace MetricFlowX;
using Flow = BloodFlowSystem<1, 3>;

namespace
{
  enum class PressureDatum
  {
    none,
    spatial,
    cellwise
  };
  void
  prepare_mesh(Flow &flow)
  {
    Triangulation<1, 3>         prepared;
    const std::vector<Point<3>> points{{0., 0., 0.},
                                       {.1, 0., 0.},
                                       {0., 1., .25},
                                       {.1, 1., .25}};
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
    GridTools::partition_triangulation_zorder(
      Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), prepared, false);
    flow.triangulation().create_triangulation(
      TriangulationDescription::Utilities::
        create_description_from_triangulation(prepared, MPI_COMM_WORLD));
  }

  double
  cellwise_datum(const unsigned int vessel, const double time)
  {
    return 200. + 19. * vessel + 2. * time;
  }

  double
  external_datum(const Point<3>    &point,
                 const double       time,
                 const unsigned int vessel)
  {
    return 17. + 100. * point[0] + 31. * point[1] - 9. * point[2] + 5. * time +
           7. * vessel;
  }
} // namespace

void
test()
{
  const auto       comm = MPI_COMM_WORLD;
  const auto       rank = Utilities::MPI::this_mpi_process(comm);
  constexpr double time = .37;
  for (const unsigned int degree : {1u, 2u})
    {
      const auto filename =
        "spatial_pressure_output_" + std::to_string(degree) + ".prm";
      if (rank == 0)
        {
          std::ofstream prm(filename);
          prm
            << "subsection Functions\n set Inflow function = 2e-9*(1+t)\nend\n"
            << "subsection MetricFlowSystem<1, 3>\n"
            << " set Vtk file path for mesh input = unused_in_memory.vtk\n"
            << " set Wall model = Square root\n"
            << " set Outlet boundary condition type = RCR\n"
            << " set Finite element degree = " << degree << "\nend\n";
        }
      MPI_Barrier(comm);
      ParameterAcceptor::clear();
      Flow flow(comm);
      flow.initialize_params(filename, "");
      prepare_mesh(flow);
      Flow::NetworkPhysics input;
      for (unsigned int id = 0; id < 2; ++id)
        {
          auto &v = input.vessels[id];
          v.r_in = v.r_out = v.r_d = .003 + .001 * id;
          v.a0 = v.a_d        = numbers::PI * v.r_d * v.r_d;
          v.E                 = 5.e5;
          v.h_wall            = .0003;
          v.p_d               = 700. + 100. * id;
          v.p0                = 10. + 3. * id;
          v.L                 = .1;
          v.dynamic_viscosity = .00371;
        }
      input.terminals = {{1, {0., 1.e8, 0., 0., "first outlet"}},
                         {2, {0., 1.e8, 0., 0., "second inlet"}},
                         {3, {0., 1.e8, 0., 0., "second outlet"}}};
      flow.setup(input);
      auto state = flow.make_state();
      // No trace solve changes this deliberately prescribed rest state.
      flow.compute_initial_solution(state, time);
      std::map<CellId, unsigned int> material_by_cell;
      for (const auto &cell : flow.dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          material_by_cell.emplace(cell->id(), cell->material_id());
      bool       valid_context = true;
      const auto check_context =
        [&](const Flow::PressureEvaluationPoint &point) {
          const auto entry = material_by_cell.find(point.cell_id);
          valid_context =
            valid_context && entry != material_by_cell.end() &&
            (entry == material_by_cell.end() ||
             entry->second == point.vessel_id) &&
            std::abs(point.time - time) < 1.e-14 &&
            std::abs(point.point[1] - point.vessel_id) < 1.e-14 &&
            std::abs(point.point[2] - .25 * point.vessel_id) < 1.e-14;
        };
      const Flow::ExternalPressureProvider provider =
        [&](const Flow::PressureEvaluationPoint &point) {
          check_context(point);
          return external_datum(point.point, point.time, point.vessel_id);
        };
      const Flow::ExternalPressureProvider constant_provider =
        [&](const Flow::PressureEvaluationPoint &point) {
          check_context(point);
          return cellwise_datum(point.vessel_id, point.time);
        };
      const auto check = [&](const VectorType   &pressure,
                             const PressureDatum datum) {
        double                local_error   = 0.;
        unsigned int          local_samples = 0;
        const MappingQ1<1, 3> mapping;
        for (const auto &cell : flow.dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              std::vector<types::global_dof_index> indices(
                cell->get_fe().n_dofs_per_cell());
              cell->get_dof_indices(indices);
              const auto &v = input.vessels.at(cell->material_id());
              for (unsigned int i = 0; i < indices.size(); ++i)
                {
                  double expected = 0.;
                  if (cell->get_fe().system_to_component_index(i).first == 0)
                    {
                      const auto point = mapping.transform_unit_to_real_cell(
                        cell, cell->get_fe().unit_support_point(i));
                      expected = v.p0 + v.p_d;
                      if (datum == PressureDatum::spatial)
                        expected +=
                          external_datum(point, time, cell->material_id());
                      else if (datum == PressureDatum::cellwise)
                        expected += cellwise_datum(cell->material_id(), time);
                      else
                        AssertThrow(datum == PressureDatum::none,
                                    ExcInternalError());
                      ++local_samples;
                    }
                  local_error =
                    std::max(local_error,
                             std::abs(pressure(indices[i]) - expected));
                }
            }
        AssertThrow(Utilities::MPI::sum(local_samples, comm) ==
                      2 * (degree + 1),
                    ExcMessage("Nodal pressure samples were not exercised."));
        const double error = Utilities::MPI::max(local_error, comm);
        AssertThrow(error < 1.e-9,
                    ExcMessage("Nodal pressure output error = " +
                               std::to_string(error)));
      };
      auto pressure = flow.make_state();
      flow.compute_pressure(state, pressure, time);
      check(pressure, PressureDatum::none);
      flow.compute_pressure(state, pressure, time, constant_provider);
      check(pressure, PressureDatum::cellwise);
      flow.set_external_pressure_provider(constant_provider);
      flow.compute_pressure(state, pressure, time);
      check(pressure, PressureDatum::cellwise);
      flow.clear_external_pressure_provider();
      flow.compute_pressure(state, pressure, time, provider);
      check(pressure, PressureDatum::spatial);
      flow.set_external_pressure_provider(provider);
      flow.compute_pressure(state, pressure, time);
      check(pressure, PressureDatum::spatial);
      auto rate = flow.make_state();
      rate      = 0.;
      flow.compute_pressure(state, rate, pressure, time);
      check(pressure, PressureDatum::spatial);
      AssertThrow(Utilities::MPI::min(valid_context ? 1 : 0, comm) == 1,
                  ExcMessage(
                    "Pressure output changed the cell/material/time context."));
      flow.clear_external_pressure_provider();
    }
  if (rank == 0)
    {
      deallog.depth_file(10);
      deallog << "Spatial external pressure output: PASSED" << std::endl;
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  test();
}
