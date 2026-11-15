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

  // A small native archive analogue: three role trees, an eliminated star bed,
  // and a sealed side branch at a genuine venous junction. Keep this separate
  // from the analytical steady-state oracle above: all cell/trace states here
  // are deliberately nonuniform and need not satisfy the nonlinear equations.
  for (const bool capacitor : {false, true})
    {
      ParameterAcceptor::clear();
      Flow flow(comm);
      flow.initialize_params("linked_external_pressure.prm", "");
      flow.fe_degree = 2;
      Triangulation<1, 3>                              prepared;
      const std::vector<Point<3>>                      points{{0., 0., 0.},
                                                              {.1, 0., 0.},
                                                              {0., 1., 0.},
                                                              {.1, 1., 0.},
                                                              {0., 2., 0.},
                                                              {.1, 2., 0.},
                                                              {.05, 2., 0.},
                                                              {.05, 2.04, 0.}};
      const std::array<std::array<unsigned int, 2>, 5> endpoints{
        {{{0, 1}}, {{2, 3}}, {{4, 6}}, {{6, 5}}, {{6, 7}}}};
      const std::array<unsigned int, 5> materials{{0, 1, 2, 2, 3}};
      std::vector<CellData<1>>          cells(endpoints.size());
      for (unsigned int i = 0; i < cells.size(); ++i)
        {
          cells[i].vertices[0] = endpoints[i][0];
          cells[i].vertices[1] = endpoints[i][1];
          cells[i].material_id = materials[i];
        }
      prepared.create_triangulation(points, cells, SubCellData());
      const std::array<types::boundary_id, 8> boundaries{
        {1, 2, 3, 4, 5, 6, 255, 7}};
      for (const auto &cell : prepared.active_cell_iterators())
        for (const auto face : cell->face_indices())
          if (cell->face(face)->at_boundary())
            cell->face(face)->set_boundary_id(
              boundaries[cell->vertex_index(face)]);
      prepared.refine_global(1);
      GridTools::partition_triangulation_zorder(
        Utilities::MPI::n_mpi_processes(comm), prepared, false);
      flow.triangulation().create_triangulation(
        TriangulationDescription::Utilities::
          create_description_from_triangulation(prepared, comm));

      Flow::NetworkPhysics        input;
      const std::array<double, 4> reference{{1200., 1000., 600., 600.}};
      const std::array<double, 3> arms{{1.e9, 1.e8, 1.e8}};
      const std::array<types::boundary_id, 3> leaves{{2, 4, 5}};
      for (unsigned int id = 0; id < 4; ++id)
        {
          auto &v = input.vessels[id];
          v.a0 = v.a_d        = 1.e-4;
          v.r_d               = std::sqrt(v.a_d / numbers::PI);
          v.E                 = 5.e5;
          v.h_wall            = .0005;
          v.L                 = id == 3 ? .04 : .1;
          v.p_d               = reference[id];
          v.dynamic_viscosity = .00371;
        }
      const double conductance = 1. / arms[0] + 1. / arms[1] + 1. / arms[2];
      for (unsigned int id = 0; id < 3; ++id)
        {
          input.terminals[id == 2 ? 6 : 2 * id + 1] = {
            0., 1.e7, 0., reference[id], "root"};
          input.terminals[leaves[id]] = {
            0., 1.e6, capacitor && id == 0 ? 1.e-9 : 0., 0., "bed"};
          for (unsigned int other = 0; other < id; ++other)
            input.terminal_connections.push_back(
              {leaves[id],
               leaves[other],
               conductance * arms[id] * arms[other]});
        }
      input.sealed_terminals.insert(7);
      flow.setup(input);
      AssertThrow(flow.terminal_links.size() == 3 &&
                    flow.sealed_terminal_ids.count(7) &&
                    Utilities::MPI::sum(flow.junctions.size(), comm) > 0,
                  ExcMessage("The combined native fixture lost its topology."));
      auto state = flow.make_state(), rate = flow.make_state();
      auto column_scales = flow.make_state(), row_scales = flow.make_state();
      const std::array<double, 5> scales{{1.e-4, .1, 1.e-4, .1, 1000.}};
      // Cell/trace area equations and capacitor balances can be in flow units;
      // RCR pressure rows are larger, so normalize each matrix entry by its
      // actual action as well. This does not hide small bed conductances behind
      // an unscaled unit absolute tolerance.
      const std::array<double, 5> residual_scales{
        {1.e-5, .01, 1.e-5, .1, 1.e-5}};
      const std::array<Flow::Component, 5> components{
        {Flow::Component::area,
         Flow::Component::velocity,
         Flow::Component::area_trace,
         Flow::Component::velocity_trace,
         Flow::Component::capacitor_pressure}};
      for (unsigned int c = 0; c < components.size(); ++c)
        for (const auto i : flow.component_dofs(components[c]))
          if (flow.locally_owned_dofs().is_element(i))
            {
              column_scales[i] = scales[c];
              row_scales[i]    = residual_scales[c];
              state[i]         = c == 0 || c == 2 ?
                                   1.e-4 * (1.04 + .025 * std::sin(.37 * (i + 1))) :
                                 c == 4 ? 1100. + 70. * std::cos(.31 * (i + 1)) :
                                          .04 + .017 * std::sin(.29 * (i + 1));
              rate[i] = c == 0 || c == 2 ? 2.e-7 * (1. + .1 * std::cos(i + 1)) :
                        c == 4           ? 20. :
                                           .03;
            }
      for (auto *v : {&state, &rate, &column_scales, &row_scales})
        v->compress(VectorOperation::insert);
      const Flow::ExternalPressureProvider pressure = [](const auto &point) {
        return 43. + 7. * point.vessel_id + 19. * point.point[0] +
               11. * point.point[1];
      };
      flow.set_external_pressure_provider(pressure);
      flow.assemble_state_jacobian(.375, state, rate);
      flow.assemble_derivative_jacobian(.375, state, rate);
      for (const bool derivative : {false, true})
        for (const double epsilon : {2.e-5, 1.e-5})
          {
            const auto &matrix = derivative ?
                                   flow.derivative_jacobian_matrix() :
                                   flow.state_jacobian_matrix();
            double      worst  = 0.;
            for (types::global_dof_index column = 0; column < state.size();
                 ++column)
              {
                const bool owned = flow.locally_owned_dofs().is_element(column);
                const double scale =
                  Utilities::MPI::sum(owned ? column_scales[column] : 0., comm);
                auto plus = derivative ? rate : state, minus = plus;
                auto unit = flow.make_state(), action = flow.make_state();
                if (owned)
                  {
                    plus[column] += epsilon * scale;
                    minus[column] -= epsilon * scale;
                    unit[column] = scale;
                  }
                for (auto *v : {&plus, &minus, &unit})
                  v->compress(VectorOperation::insert);
                matrix.vmult(action, unit);
                auto fp = flow.make_state(), fm = flow.make_state();
                flow.assemble_residual(.375,
                                       derivative ? state : plus,
                                       derivative ? plus : rate,
                                       fp);
                flow.assemble_residual(.375,
                                       derivative ? state : minus,
                                       derivative ? minus : rate,
                                       fm);
                for (const auto row : flow.locally_owned_dofs())
                  {
                    const double fd =
                      (fp[row] - fm[row]) / (2. * epsilon) / row_scales[row];
                    const double exact = action[row] / row_scales[row];
                    worst              = std::max(worst,
                                     std::abs(fd - exact) /
                                       (1. + std::abs(fd) + std::abs(exact)));
                  }
              }
            worst = Utilities::MPI::max(worst, comm);
            if (Utilities::MPI::this_mpi_process(comm) == 0)
              std::cout << "Combined native FD " << (derivative ? "ydot" : "y")
                        << " C=" << capacitor << " epsilon=" << epsilon
                        << " error=" << worst << std::endl;
            AssertThrow(worst < 2.e-6,
                        ExcMessage(
                          "Combined native state/mass Jacobian mismatch: " +
                          Utilities::to_string(worst)));
          }

      // Lambda is an independent piecewise-linear field on every cell.
      // Build its sparse Jacobian through the public visitor; the pressure
      // provider uses -lambda, matching the coupled sign convention.
      using Index = types::global_dof_index;
      std::map<CellId, std::tuple<Index, Point<3>, Point<3>>> visible;
      for (const auto &cell : flow.dof_handler().active_cell_iterators())
        if (!cell->is_artificial())
          visible.emplace(cell->id(),
                          std::make_tuple(2 * cell->global_active_cell_index(),
                                          cell->vertex(0),
                                          cell->vertex(1)));
      const Flow::ExternalPressureBasisProvider basis = [&](const auto &point) {
        const auto &[first, p0, p1] = visible.at(point.cell_id);
        const auto   d              = p1 - p0;
        const double x              = (point.point - p0) * d / d.norm_square();
        return Flow::ExternalPressureBasis{{first, 1. - x}, {first + 1, x}};
      };
      const auto coefficient = [](Index i) {
        return 23. * std::sin(.43 * (i + 1));
      };
      const auto lambda = [&](const auto &point) {
        double value = 0.;
        for (const auto &[column, weight] : basis(point))
          value += coefficient(column) * weight;
        return value;
      };
      auto action = flow.make_state();
      flow.visit_external_pressure_jacobian(
        .375, state, basis, [&](Index row, Index column, double value) {
          action[row] -= value * coefficient(column);
        });
      action.compress(VectorOperation::add);
      for (const double epsilon : {2.e-4, 1.e-4})
        {
          auto fp = flow.make_state(), fm = flow.make_state();
          flow.set_external_pressure_provider([&](const auto &point) {
            return pressure(point) - epsilon * lambda(point);
          });
          flow.assemble_residual(.375, state, rate, fp);
          flow.set_external_pressure_provider([&](const auto &point) {
            return pressure(point) + epsilon * lambda(point);
          });
          flow.assemble_residual(.375, state, rate, fm);
          double worst = 0.;
          for (const auto row : flow.locally_owned_dofs())
            {
              const double fd =
                (fp[row] - fm[row]) / (2. * epsilon) / row_scales[row];
              const double exact = action[row] / row_scales[row];
              worst              = std::max(worst,
                               std::abs(fd - exact) /
                                 (1. + std::abs(fd) + std::abs(exact)));
            }
          worst = Utilities::MPI::max(worst, comm);
          if (Utilities::MPI::this_mpi_process(comm) == 0)
            std::cout << "Combined native FD lambda C=" << capacitor
                      << " epsilon=" << epsilon << " error=" << worst
                      << std::endl;
          AssertThrow(worst < 2.e-6,
                      ExcMessage(
                        "Combined native pressure Jacobian mismatch: " +
                        Utilities::to_string(worst)));
        }
      flow.clear_external_pressure_provider();
    }
  deallog
    << "Native three-arm microbed, sealed branch and full Jacobians: PASSED"
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
