// SPDX-License-Identifier: MIT
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <metric_flow_x/blood_flow_system.h>

#include <fstream>

#include "tests.h"
using namespace MetricFlowX;
using Flow = BloodFlowSystem<1, 3>;

// Independent integral of Poiseuille resistance for linear radius.
double
resistance(double length, double a, double b, double mu)
{
  return a == b ? 8. * mu * length / (numbers::PI * std::pow(a, 4)) :
                  8. * mu * length / (3. * numbers::PI * (b - a)) *
                    (1. / std::pow(a, 3) - 1. / std::pow(b, 3));
}
void
test()
{
  for (const bool capacitive : {false, true})
    for (const unsigned int levels : {0u, 1u})
      {
        ParameterAcceptor::clear();
        Flow flow(MPI_COMM_WORLD);
        flow.initialize_params("native-graph-links.prm", "");
        Flow::InitialGraph graph;
        graph.points     = {{0., 0., 0.},
                            {.1, 0., 0.},
                            {0., 1., 0.},
                            {.1, 1., 0.},
                            {.1, 2., 0.},
                            {0., 2., 0.}};
        graph.cells      = {{0, {0, 1, 0}}, {1, {2, 3, 1}}, {2, {4, 5, 2}}};
        graph.boundaries = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}};
        Flow::NetworkPhysics input;
        for (unsigned int i = 0; i < 3; ++i)
          {
            auto &v = input.vessels[i];
            v.r_d = v.r_in = .003;
            v.r_out        = i >= 1 ? .0025 : .003;
            v.a_d = v.a0        = numbers::PI * v.r_d * v.r_d;
            v.E                 = 5e5;
            v.h_wall            = .0003;
            v.p_d               = 1000.;
            v.L                 = .1;
            v.dynamic_viscosity = .0035;
          }
        const double proximal = capacitive ? 1e7 : 0.;
        // Distal star arms are eliminated analytically, as in a three-way bed.
        const std::array<double, 3>       arm{{1e8, 2e8, 3e8}};
        const std::array<unsigned int, 3> leaves{{1, 3, 4}};
        for (unsigned int i = 0; i < 3; ++i)
          {
            input.terminals[leaves[i]] = {proximal,
                                          1e9,
                                          capacitive ? 1e-9 : 0.,
                                          0.,
                                          "linked-unused-reservoir"};
            input.terminals.at(leaves[i]).outlet_pressure_series.read(
              "unused-linked-reservoir.csv", "pressure_Pa");
            for (unsigned int j = i + 1; j < 3; ++j)
              {
                const auto k = 3 - i - j;
                input.terminal_connections.push_back(
                  {leaves[i],
                   leaves[j],
                   arm[i] + arm[j] + arm[i] * arm[j] / arm[k]});
              }
          }
        input.terminals[2] = {0., 1e7, 0., 1100., "portal-root"};
        input.terminals[5] = {0., 2e7, 0., 900., "venous-root"};
        input.terminals.at(5).outlet_pressure_series.read(
          "native-graph-ivc.csv", "pressure_Pa");
        Triangulation<1, 3>      prepared;
        std::vector<CellData<1>> cells(3);
        for (unsigned int i = 0; i < 3; ++i)
          {
            cells[i].vertices[0] = 2 * i;
            cells[i].vertices[1] = 2 * i + 1;
            cells[i].material_id = i;
          }
        prepared.create_triangulation(graph.points, cells, SubCellData());
        for (const auto &cell : prepared.active_cell_iterators())
          for (const auto f : cell->face_indices())
            cell->face(f)->set_boundary_id(
              graph.boundaries.at(2 * cell->material_id() + f));
        prepared.refine_global(levels);
        GridTools::partition_triangulation_zorder(
          Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), prepared, false);
        flow.triangulation().create_triangulation(
          TriangulationDescription::Utilities::
            create_description_from_triangulation(prepared, MPI_COMM_WORLD));
        flow.setup(input, graph);
        // The native reference area is unchanged on owned and ghost cells,
        // including the reversed tapered vessel; it is not the guessed area.
        for (const auto &cell : flow.dof_handler().active_cell_iterators())
          if (!cell->is_artificial())
            {
              const auto  id   = cell->material_id();
              const auto &edge = graph.cells.at(id);
              const auto &v    = input.vessels.at(id);
              const auto  tangent =
                (graph.points[edge.node1] - graph.points[edge.node0]) / .1;
              const double x = std::clamp(
                ((cell->center() - graph.points[edge.node0]) * tangent) / .1,
                0.,
                1.);
              const double r = v.r_in + x * (v.r_out - v.r_in);
              AssertThrow(
                std::abs(flow.reference_area(cell, cell->center()) -
                         numbers::PI * r * r) < 1e-17,
                ExcMessage(
                  "Prepared native reference taper/orientation changed."));
            }
        for (const double t : {.37, .73})
          {
            const double qa = 2e-9 * (1. + t), pp = 1100., pv = 900. + 100. * t;
            std::array<double, 3> tube;
            for (unsigned int i = 0; i < 3; ++i)
              {
                const auto &v = input.vessels.at(i);
                tube[i] = resistance(.1, v.r_in, v.r_out, v.dynamic_viscosity);
              }
            const double rp = arm[1] + proximal + tube[1] + 1e7,
                         rv = arm[2] + proximal + tube[2] + 2e7;
            const double pc = (qa + pp / rp + pv / rv) / (1. / rp + 1. / rv);
            const double qp = (pp - pc) / rp, qv = (pc - pv) / rv;
            const std::array<double, 3>                q{{qa, qp, qv}};
            const std::array<std::array<double, 2>, 3> pressure{
              {{{pc + (arm[0] + proximal + tube[0]) * qa,
                 pc + (arm[0] + proximal) * qa}},
               {{pp - 1e7 * qp, pc + (arm[1] + proximal) * qp}},
               {{pc - (arm[2] + proximal) * qv, pv + 2e7 * qv}}}};
            auto y = flow.make_state();
            flow.compute_steady_network_initial_solution(y, t);
            for (const auto &cell : flow.dof_handler().active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  const auto  id   = cell->material_id();
                  const auto &edge = graph.cells.at(id);
                  const auto &v    = input.vessels.at(id);
                  const auto  tangent =
                    (graph.points[edge.node1] - graph.points[edge.node0]) / .1;
                  const double orientation =
                    flow.compute_directional_vector(cell) * tangent >= 0. ? 1. :
                                                                            -1.;
                  std::vector<types::global_dof_index> dofs(
                    cell->get_fe().n_dofs_per_cell());
                  cell->get_dof_indices(dofs);
                  for (unsigned int i = 0; i < dofs.size(); ++i)
                    {
                      const auto c =
                        cell->get_fe().system_to_component_index(i).first;
                      if (c == 4)
                        continue;
                      const auto   point = flow.support_point(cell, i);
                      const double x     = std::clamp(
                        ((point - graph.points[edge.node0]) * tangent) / .1,
                        0.,
                        1.);
                      const double r = v.r_in + x * (v.r_out - v.r_in);
                      const double fraction =
                        x == 0. ?
                          0. :
                          resistance(.1 * x, v.r_in, r, v.dynamic_viscosity) /
                            tube[id];
                      const double p =
                        pressure[id][0] +
                        fraction * (pressure[id][1] - pressure[id][0]);
                      const double ad   = flow.reference_area(cell, point),
                                   beta = 4. * std::sqrt(numbers::PI) / 3. *
                                          v.E * v.h_wall;
                      const double a =
                        std::pow(std::sqrt(ad) + ad / beta * (p - v.p_d - v.p0),
                                 2);
                      const double expected =
                        c == 0 || c == 2 ? a : orientation * q[id] / a;
                      AssertThrow(
                        std::abs(y[dofs[i]] - expected) <
                          (c == 0 || c == 2 ? 1e-13 : 1e-9),
                        ExcMessage(
                          "Linked native graph differs from independent three-arm circuit/taper oracle."));
                    }
                }
            for (const auto &[bid, index] : flow.rcr_pc_dof)
              if (flow.locally_owned_dofs().is_element(index))
                {
                  const double expected = bid == 1 ? pc + arm[0] * qa :
                                          bid == 3 ? pc + arm[1] * qp :
                                                     pc - arm[2] * qv;
                  AssertThrow(
                    std::abs(y[index] - expected) < 1e-8,
                    ExcMessage(
                      "Linked compartment pressure ignored R1/link precedence."));
                }
          }
      }
}
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::ofstream ignored("unused-linked-reservoir.csv");
      ignored << "time_s,pressure_Pa\n0,1\n0.01,2\n";
      std::ofstream ivc("native-graph-ivc.csv");
      ivc << "time_s,pressure_Pa\n0,900\n1,1000\n";
      std::ofstream prm("native-graph-links.prm");
      prm
        << "subsection Functions\n set Inflow function = 2e-9*(1+t)\nend\nsubsection Metric Flow Parameters\n set Profile constant for friction term (xi) = 2\nend\nsubsection MetricFlowSystem<1, 3>\n set Output directory = native-graph-links-output\n set Wall model = Square root\n set Numerical flux type = HLL\n set Outlet boundary condition type = RCR\n set Initial state source = steady network\n set Finite element degree = 2\n set Verbosity (console depth) = 0\nend\n";
    }
  MPI_Barrier(MPI_COMM_WORLD);
  test();
  deallog.depth_file(10);
  deallog
    << "Prepared linked three-arm graph, taper and time-dependent reservoirs: PASSED"
    << std::endl;
}
