// SPDX-License-Identifier: MIT
// Three terminal arms meet at an eliminated compartment. Check the exact
// pressure-driven steady flow, explicit/installed pressure parity, and the
// pressure derivative, including a capacitor linked to algebraic neighbours.
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <metric_flow_x/blood_flow_system.h>

#include <algorithm>
#include <array>
#include <set>

#include "pressure_jacobian_checks.h"
#include "tests.h"

using namespace MetricFlowX;
using Flow = BloodFlowSystem<1, 3>;

void
make_mesh(Flow &flow)
{
  Triangulation<1, 3>      prepared;
  std::vector<Point<3>>    points;
  std::vector<CellData<1>> cells(3);
  for (unsigned int v = 0; v < 3; ++v)
    {
      points.emplace_back(0., v, 0.);
      points.emplace_back(.1, v, 0.);
      cells[v].vertices[0] = 2 * v;
      cells[v].vertices[1] = 2 * v + 1;
      cells[v].material_id = v;
    }
  prepared.create_triangulation(points, cells, SubCellData());
  for (const auto &cell : prepared.active_cell_iterators())
    for (const auto f : cell->face_indices())
      cell->face(f)->set_boundary_id(2 * cell->material_id() + f + 1);
  GridTools::partition_triangulation_zorder(
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), prepared, false);
  flow.triangulation().create_triangulation(
    TriangulationDescription::Utilities::create_description_from_triangulation(
      prepared, MPI_COMM_WORLD));
}

void
test()
{
  const auto                              comm = MPI_COMM_WORLD;
  const std::array<double, 3>             physical{{1200., 1000., 600.}};
  const std::array<double, 3>             external{{200., 20., -30.}};
  const std::array<double, 3>             arms{{1.e9, 1.e8, 1.e8}};
  const std::array<types::boundary_id, 3> leaves{{2, 4, 5}};
  constexpr double                        area = 1.e-4, root_resistance = 1.e7;
  double conductance = 0., weighted_pressure = 0.;
  for (unsigned int v = 0; v < 3; ++v)
    {
      conductance += 1. / arms[v];
      weighted_pressure += physical[v] / arms[v];
    }
  const double          compartment_pressure = weighted_pressure / conductance;
  std::array<double, 3> outward;
  for (unsigned int v = 0; v < 3; ++v)
    outward[v] = (physical[v] - compartment_pressure) / arms[v];
  AssertThrow(outward[0] > 0. && outward[1] > 0. && outward[2] < 0. &&
                std::abs(outward[0] + outward[1] + outward[2]) < 1.e-20,
              ExcMessage(
                "Independent three-arm Kirchhoff solution is invalid"));

  for (const bool capacitor : {false, true})
    {
      ParameterAcceptor::clear();
      Flow flow(comm);
      flow.initialize_params("linked_external_pressure.prm", "");
      make_mesh(flow);
      Flow::NetworkPhysics input;
      for (unsigned int id = 0; id < 3; ++id)
        {
          auto &v = input.vessels[id];
          v.a0 = v.a_d                = area;
          v.r_d                       = std::sqrt(area / numbers::PI);
          v.E                         = 5.e5;
          v.h_wall                    = .0005;
          v.L                         = .1;
          v.p_d                       = physical[id] - external[id];
          const auto root             = id == 2 ? 6 : 2 * id + 1;
          input.terminals[root]       = {0.,
                                         root_resistance,
                                         0.,
                                         physical[id] + root_resistance * outward[id],
                                         "root"};
          input.terminals[leaves[id]] = {
            0., 1.e6, capacitor && id == 0 ? 1.e-9 : 0., -500., "connected"};
          for (unsigned int other = 0; other < id; ++other)
            input.terminal_connections.push_back(
              {leaves[id],
               leaves[other],
               conductance * arms[id] * arms[other]});
        }
      flow.setup(input);
      auto y = flow.make_state(), rate = flow.make_state();
      y    = 0.;
      rate = 0.;
      std::set<CellId> visible;
      for (const auto &cell : flow.dof_handler().active_cell_iterators())
        {
          if (!cell->is_artificial())
            visible.insert(cell->id());
          if (!cell->is_locally_owned())
            continue;
          const unsigned int                   id = cell->material_id();
          std::vector<types::global_dof_index> indices(
            cell->get_fe().n_dofs_per_cell());
          cell->get_dof_indices(indices);
          for (unsigned int i = 0; i < indices.size(); ++i)
            {
              const auto component =
                cell->get_fe().system_to_component_index(i).first;
              y[indices[i]] =
                component == 0 || component == 2 ?
                  area :
                  (component == 4 ? physical[id] :
                                    (id == 2 ? -1. : 1.) * outward[id] / area);
            }
        }
      y.compress(VectorOperation::insert);
      check_sparse_pressure_jacobian(flow, y);
      const auto provider = [&](const auto &point) {
        AssertThrow(
          visible.count(point.cell_id),
          ExcMessage(
            "A pressure provider was asked to evaluate a distant cell"));
        return external[point.vessel_id];
      };
      auto installed = flow.make_state(), composed = flow.make_state();
      flow.set_external_pressure_provider(provider);
      flow.assemble_residual(7.2, y, rate, installed);
      AssertThrow(
        installed.linfty_norm() < 1.e-9,
        ExcMessage(
          "The exact three-arm flow violates native physical pressure balance"));
      flow.clear_external_pressure_provider();
      flow.assemble_residual(7.2, y, rate, composed);
      flow.add_external_pressure_residual(7.2, y, provider, composed);
      composed -= installed;
      AssertThrow(
        composed.linfty_norm() < 1.e-9,
        ExcMessage(
          "Explicit linked pressure differs from installed pressure: " +
          Utilities::to_string(composed.linfty_norm())));
      double net = 0.;
      for (const auto &face : flow.boundary_faces)
        {
          const auto bid = face.cell->face(face.face_no)->boundary_id();
          if (std::find(leaves.begin(), leaves.end(), bid) != leaves.end())
            net += (face.face_no == 0 ? -1. : 1.) * y[face.a_hat_dof] *
                   y[face.u_hat_dof];
        }
      AssertThrow(
        std::abs(Utilities::MPI::sum(net, comm)) < 1.e-20,
        ExcMessage(
          "The independent steady state must conserve compartment flow"));

      const std::array<double, 3> delta{{7., -11., 5.}};
      const auto                  direction = [&](const auto &point) {
        AssertThrow(visible.count(point.cell_id),
                    ExcMessage("Pressure derivative evaluated a distant cell"));
        return delta[point.vessel_id];
      };
      constexpr double epsilon = 1.e-4;
      auto             plus = flow.make_state(), minus = flow.make_state(),
           action = flow.make_state();
      flow.set_external_pressure_provider([&](const auto &point) {
        return provider(point) + epsilon * direction(point);
      });
      flow.assemble_residual(7.2, y, rate, plus);
      flow.set_external_pressure_provider([&](const auto &point) {
        return provider(point) - epsilon * direction(point);
      });
      flow.assemble_residual(7.2, y, rate, minus);
      flow.clear_external_pressure_provider();
      action = 0.;
      flow.add_external_pressure_residual(7.2, y, direction, action);
      plus -= minus;
      plus *= .5 / epsilon;
      plus -= action;
      for (const auto component :
           {Flow::Component::area_trace, Flow::Component::capacitor_pressure})
        {
          double error = 0., scale = 0.;
          for (const auto i : flow.component_dofs(component))
            if (flow.locally_owned_dofs().is_element(i))
              {
                error += plus[i] * plus[i];
                scale += action[i] * action[i];
              }
          error = std::sqrt(Utilities::MPI::sum(error, comm));
          scale = std::sqrt(Utilities::MPI::sum(scale, comm));
          AssertThrow(
            error < 1.e-6 * scale + 1.e-13,
            ExcMessage(
              "Linked pressure derivative disagrees with physical-pressure finite differences"));
        }
    }
  deallog.depth_file(10);
  deallog
    << "Linked physical pressure, Kirchhoff balance and pressure derivative: PASSED"
    << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::ofstream prm("linked_external_pressure.prm");
      prm
        << "subsection Metric Flow Parameters\n set Density (rho) = 1060\n set Viscosity coefficient (mu) = 0\n set Profile constant for friction term (xi) = 0\nend\n"
        << "subsection MetricFlowSystem<1, 3>\n set Wall model = Square root\n set Numerical flux type = HLL\n set Outlet boundary condition type = RCR\n set Finite element degree = 1\n set Verbosity (console depth) = 0\nend\n";
    }
  MPI_Barrier(MPI_COMM_WORLD);
  test();
}
