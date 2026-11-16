// SPDX-License-Identifier: MIT
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <metric_flow_x/blood_flow_system.h>

#include <filesystem>
#include <fstream>

#include "tests.h"
using namespace MetricFlowX;
using Flow = BloodFlowSystem<1, 3>;

Flow::InitialGraph
graph()
{
  Flow::InitialGraph result;
  // Coincident points belong to disconnected tubes; node identity is not x.
  result.points     = {{0., 0., 0.}, {.1, 0., 0.}, {.1, 0., 0.}, {0., 0., 0.}};
  result.cells      = {{0, {0, 1, 0}}, {1, {2, 3, 1}}};
  result.boundaries = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
  return result;
}
void
mesh(Flow &flow, const Flow::InitialGraph &g, unsigned int refinements = 0)
{
  Triangulation<1, 3>      prepared;
  std::vector<CellData<1>> cells;
  for (const auto &[id, edge] : g.cells)
    {
      (void)id;
      CellData<1> cell;
      cell.vertices[0] = edge.node0;
      cell.vertices[1] = edge.node1;
      cell.material_id = edge.vessel_id;
      cells.push_back(cell);
    }
  prepared.create_triangulation(g.points, cells, SubCellData());
  for (const auto &cell : prepared.active_cell_iterators())
    for (const auto face : cell->face_indices())
      {
        const auto node = g.cells.at(cell->id().get_coarse_cell_id()).node0;
        const auto key =
          face == 0 ? node : g.cells.at(cell->id().get_coarse_cell_id()).node1;
        const auto boundary = g.boundaries.find(key);
        if (cell->face(face)->at_boundary())
          cell->face(face)->set_boundary_id(
            boundary == g.boundaries.end() ? 255 : boundary->second);
      }
  prepared.refine_global(refinements);
  GridTools::partition_triangulation_zorder(
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), prepared, false);
  flow.triangulation().create_triangulation(
    TriangulationDescription::Utilities::create_description_from_triangulation(
      prepared, MPI_COMM_WORLD));
}
Flow::NetworkPhysics
physics()
{
  Flow::NetworkPhysics result;
  for (unsigned int id = 0; id < 2; ++id)
    {
      auto &v = result.vessels[id];
      v.r_in = v.r_out = v.r_d = .003 + .001 * id;
      v.a0 = v.a_d        = numbers::PI * v.r_d * v.r_d;
      v.E                 = 5e5;
      v.h_wall            = .0003;
      v.p_d               = 1000.;
      v.L                 = .1;
      v.dynamic_viscosity = .0035;
    }
  result.sealed_terminals = {1, 3};
  result.terminals        = {{2, {1e7, 1e8, 1e-9, 800., "first"}},
                             {4, {2e7, 2e8, 0., 1200., "reversed"}}};
  result.terminals.at(2).outlet_pressure_series.read(
    "native-graph-pressure.csv", "pressure_Pa");
  return result;
}
void
test()
{
  const auto comm     = MPI_COMM_WORLD;
  const auto rejected = [&](const auto &action) {
    bool caught = false;
    try
      {
        action();
      }
    catch (const std::exception &)
      {
        caught = true;
      }
    AssertThrow(Utilities::MPI::min(caught ? 1u : 0u, comm) == 1,
                ExcMessage("Expected collective rejection."));
  };
  VectorType saved;
  for (const auto refinement : {0u, 1u})
    {
      ParameterAcceptor::clear();
      Flow flow(comm);
      flow.initialize_params("native-graph.prm", "");
      AssertThrow(flow.requests_steady_network_initialization(),
                  ExcInternalError());
      const auto g =
        Utilities::unpack<Flow::InitialGraph>(Utilities::pack(graph()));
      mesh(flow, g, refinement);
      const auto input = physics();
      flow.setup(input, g);
      AssertThrow(flow.triangulation().n_global_active_cells() ==
                    2u * (1u << refinement),
                  ExcInternalError());
      for (const double t : {.25, .75})
        {
          auto y = flow.make_state();
          flow.compute_steady_network_initial_solution(y, t);
          for (const auto &cell : flow.dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                std::vector<types::global_dof_index> dofs(
                  cell->get_fe().n_dofs_per_cell());
                cell->get_dof_indices(dofs);
                const auto  &v = input.vessels.at(cell->material_id());
                const double pressure =
                  cell->material_id() == 0 ? 800. + 100. * t : 1200.;
                for (unsigned int i = 0; i < dofs.size(); ++i)
                  {
                    const auto c =
                      cell->get_fe().system_to_component_index(i).first;
                    if (c == 4)
                      continue;
                    const double ad =
                      flow.reference_area(cell, flow.support_point(cell, i));
                    const double beta =
                      4. * std::sqrt(numbers::PI) / 3. * v.E * v.h_wall;
                    const double root =
                      std::sqrt(ad) + ad / beta * (pressure - v.p_d - v.p0);
                    const double expected = c == 0 || c == 2 ? root * root : 0.;
                    AssertThrow(
                      std::abs(y[dofs[i]] - expected) <
                        (c == 0 || c == 2 ? 1e-14 : 1e-10),
                      ExcMessage(
                        "Graph tube-law inverse or zero sealed flow differs from independent oracle."));
                  }
              }
          auto rate = flow.make_state();
          flow.initialize_state(y, t);
          flow.initialize_state_derivative(rate, t);
          AssertThrow(rate.l2_norm() == 0. &&
                        flow.has_admissible_wall_states(y),
                      ExcInternalError());
          if (refinement == 1 && t == .25)
            {
              saved = y;
              flow.save_checkpoint("native-graph-checkpoint", t, y, rate);
            }
        }
    }
  // Coordinates alone cannot establish incidence. Reject a split real
  // junction (fake extra inlet0s) and a merge of coincident disconnected ones.
  for (const bool coincident : {false, true})
    for (const bool corrupt : {false, true})
      {
        ParameterAcceptor::clear();
        Flow flow(comm);
        flow.initialize_params("native-graph.prm", "");
        Flow::InitialGraph good, bad;
        good.points     = {{0., 0., 0.}, {.1, 0., 0.}, {.2, 0., 0.}};
        good.cells      = {{0, {0, 1, 0}}, {1, {1, 2, 1}}};
        good.boundaries = {{0, 0}, {2, 1}};
        auto input      = physics();
        input.sealed_terminals.clear();
        input.terminals.clear();
        input.terminals[1] = {0., 1e8, 0., 1000., "reservoir"};
        if (coincident)
          {
            good.points.insert(good.points.end(),
                               {Point<3>(0., 0., 0.),
                                Point<3>(.1, 0., 0.),
                                Point<3>(.2, 0., 0.)});
            good.cells.emplace(2, Flow::InitialGraphCell{3, 4, 0});
            good.cells.emplace(3, Flow::InitialGraphCell{4, 5, 1});
            good.boundaries.emplace(3, 2);
            good.boundaries.emplace(5, 3);
            input.terminals[2] = {0., 1e8, 0., 1000., "second-root"};
            input.terminals[3] = {0., 1e8, 0., 1000., "second-outlet"};
            bad.points         = {{0., 0., 0.},
                                  {.1, 0., 0.},
                                  {.2, 0., 0.},
                                  {0., 0., 0.},
                                  {.2, 0., 0.}};
            bad.cells          = {{0, {0, 1, 0}},
                                  {1, {1, 2, 1}},
                                  {2, {3, 1, 0}},
                                  {3, {1, 4, 1}}};
            bad.boundaries     = {{0, 0}, {2, 1}, {3, 2}, {4, 3}};
          }
        else
          {
            bad.points     = {{0., 0., 0.},
                              {.1, 0., 0.},
                              {.1, 0., 0.},
                              {.2, 0., 0.}};
            bad.cells      = {{0, {0, 1, 0}}, {1, {2, 3, 1}}};
            bad.boundaries = {{0, 0}, {1, 0}, {2, 0}, {3, 1}};
          }
        mesh(flow, good, 1);
        if (corrupt)
          rejected([&] { flow.setup(input, bad); });
        else
          {
            flow.setup(input, good);
            auto y = flow.make_state();
            flow.compute_steady_network_initial_solution(y, .25);
            AssertThrow(std::isfinite(y.l2_norm()), ExcInternalError());
          }
      }
  // Malformed or inconsistent metadata must fail before native DoF setup.
  for (unsigned int kind = 0; kind < 5; ++kind)
    {
      ParameterAcceptor::clear();
      Flow flow(comm);
      flow.initialize_params("native-graph.prm", "");
      mesh(flow, graph());
      auto g = graph();
      if (kind == 0)
        g.cells.at(0).vessel_id = 99;
      if (kind == 1)
        {
          g.cells.emplace(7, g.cells.at(0));
          g.cells.erase(0);
        }
      if (kind == 2)
        g.points[0][1] = .1;
      if (kind == 3)
        g.boundaries.at(0) = 2;
      if (kind == 4 && Utilities::MPI::this_mpi_process(comm) == 0)
        g.points[0][2] = .2;
      rejected([&] { flow.setup(physics(), g); });
    }
  {
    ParameterAcceptor::clear();
    Flow flow(comm);
    flow.initialize_params("native-graph.prm", "");
    mesh(flow, graph());
    rejected([&] { flow.setup(physics()); });
  }
  {
    ParameterAcceptor::clear();
    Flow flow(comm);
    flow.initialize_params("native-graph.prm", "");
    mesh(flow, graph());
    auto input = physics();
    input.terminals.clear();
    input.sealed_terminals = {1, 2, 3, 4};
    flow.setup(input, graph());
    auto y = flow.make_state();
    rejected([&] { flow.compute_steady_network_initial_solution(y, .25); });
  }
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    {
      std::filesystem::remove("native-graph-pressure.csv");
      std::filesystem::remove("native-graph.prm");
    }
  MPI_Barrier(comm);
  {
    ParameterAcceptor::clear();
    Flow restored(comm);
    restored.load_checkpoint("native-graph-checkpoint");
    AssertThrow(restored.requests_steady_network_initialization() &&
                  restored.initial_graph_cells_.empty(),
                ExcInternalError());
    auto difference = restored.state();
    difference -= saved;
    AssertThrow(
      difference.l2_norm() == 0. && restored.state_derivative().l2_norm() == 0.,
      ExcMessage("Native checkpoint lost the accepted guess state/rates."));
    AssertThrow(restored.rcr_map.at(2).outlet_pressure(.9) == 890.,
                ExcMessage("Checkpoint lost future reservoir samples."));
    auto p = restored.make_state();
    restored.compute_pressure(restored.state(),
                              restored.state_derivative(),
                              p,
                              .25);
    AssertThrow(std::isfinite(p.l2_norm()), ExcInternalError());
    auto residual = restored.make_state();
    restored.assemble_residual(.3,
                               restored.state(),
                               restored.state_derivative(),
                               residual);
    AssertThrow(
      std::isfinite(residual.l2_norm()),
      ExcMessage(
        "Restored continuation residual requires no initialization graph."));
    auto unused = restored.make_state();
    rejected([&] { restored.initialize_state(unused, .25); });
  }
}
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::filesystem::remove_all("native-graph-checkpoint");
      std::ofstream csv("native-graph-pressure.csv");
      csv << "time_s,pressure_Pa\n0,800\n1,900\n";
      std::ofstream prm("native-graph.prm");
      prm
        << "subsection Functions\n set Inflow function = 0\nend\nsubsection MetricFlowSystem<1, 3>\n set Output directory = native-graph-output\n set Wall model = Square root\n set Numerical flux type = HLL\n set Outlet boundary condition type = RCR\n set Initial state source = steady network\n set Finite element degree = 2\n set Verbosity (console depth) = 0\n set Initial consistency tolerance = 1e-8\nend\n";
    }
  MPI_Barrier(MPI_COMM_WORLD);
  test();
  deallog.depth_file(10);
  deallog
    << "Prepared native graph, seals, identities and graph-free checkpoint restore: PASSED"
    << std::endl;
}
