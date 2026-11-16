// SPDX-License-Identifier: MIT
#include <deal.II/grid/tria_description.h>

#include <metric_flow_x/blood_flow_system.h>

#include "tests.h"
using namespace MetricFlowX;
using Flow = BloodFlowSystem<1, 3>;

std::string
fingerprint(const Flow &flow)
{
  std::ostringstream out;
  out << std::hexfloat;
  for (const auto &cell : flow.triangulation().active_cell_iterators())
    if (!cell->is_artificial())
      out << cell->id().to_string() << ':' << cell->material_id() << ':'
          << cell->subdomain_id() << ':' << cell->vertex(0) << ':'
          << cell->vertex(1) << ':' << cell->face(0)->boundary_id() << ':'
          << cell->face(1)->boundary_id() << '\n';
  return out.str();
}

std::string
dof_fingerprint(const Flow &flow)
{
  std::ostringstream out;
  out << flow.dof_handler().n_dofs() << '\n';
  for (const auto &cell : flow.dof_handler().active_cell_iterators())
    if (cell->is_locally_owned())
      {
        out << cell->id().to_string() << ':';
        std::vector<types::global_dof_index> indices(
          cell->get_fe().n_dofs_per_cell());
        cell->get_dof_indices(indices);
        for (const auto index : indices)
          out << index << ',';
        out << '\n';
      }
  return out.str();
}

void
test()
{
  const auto ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  for (const bool disconnected : {false, true})
    for (const unsigned int refinement : {0u, 1u})
      for (unsigned int rotation = 0; rotation < ranks; ++rotation)
        for (const bool split : {false, true})
          {
            ParameterAcceptor::clear();
            Flow::InitialGraph graph;
            const unsigned int edges = disconnected ? 6 : 8;
            if (disconnected)
              for (unsigned int k = 0; k < 3; ++k)
                for (unsigned int j = 0; j < 3; ++j)
                  graph.points.emplace_back(.1 * j, k, 0.);
            else
              {
                graph.points.emplace_back(0., 0., 0.);
                for (unsigned int i = 0; i < edges; ++i)
                  graph.points.emplace_back(
                    .1 * std::cos(2. * numbers::PI * i / edges),
                    .1 * std::sin(2. * numbers::PI * i / edges),
                    0.);
              }
            std::vector<CellData<1>> cells(edges);
            for (unsigned int i = 0; i < edges; ++i)
              {
                const unsigned int a =
                  disconnected ? 3 * (i / 2) + i % 2 : i + 1;
                const unsigned int z = disconnected ? a + 1 : 0;
                cells[i].vertices[0] = a;
                cells[i].vertices[1] = z;
                cells[i].material_id = 0;
                graph.cells.emplace(7 + 3 * i, Flow::InitialGraphCell{a, z, 0});
              }
            if (disconnected)
              for (unsigned int k = 0; k < 3; ++k)
                {
                  graph.boundaries[3 * k]     = 2 * k;
                  graph.boundaries[3 * k + 2] = 2 * k + 1;
                }
            else
              for (unsigned int i = 0; i < edges; ++i)
                graph.boundaries[i + 1] = i;
            Flow::NetworkPhysics input;
            auto                &v = input.vessels[0];
            v.r_in = v.r_out = v.r_d = .003;
            v.a0 = v.a_d        = numbers::PI * .003 * .003;
            v.E                 = 5e5;
            v.h_wall            = .0003;
            v.p_d               = 1000.;
            v.L                 = .1;
            v.dynamic_viscosity = .0035;
            for (const auto &[node, id] : graph.boundaries)
              {
                (void)node;
                if (id != 0)
                  input.terminals[id] = {0., 1e8, 0., 1000., "outlet"};
              }
            Triangulation<1, 3> serial;
            serial.create_triangulation(graph.points, cells, SubCellData());
            for (const auto &cell : serial.active_cell_iterators())
              for (const auto face : cell->face_indices())
                if (cell->face(face)->at_boundary())
                  {
                    const auto node =
                      cells[cell->active_cell_index()].vertices[face];
                    const auto found = graph.boundaries.find(node);
                    cell->face(face)->set_boundary_id(
                      found == graph.boundaries.end() ? 255 : found->second);
                  }
            serial.refine_global(refinement);
            for (const auto &cell : serial.active_cell_iterators())
              {
                const auto         coarse = cell->id().get_coarse_cell_id();
                const unsigned int owner =
                  disconnected && ranks == 4 ?
                    (coarse % 2 == 0 ? coarse / 2 : 3) :
                    (rotation + coarse) % ranks;
                cell->set_subdomain_id(owner);
              }
            auto description = TriangulationDescription::Utilities::
              create_description_from_triangulation(serial, MPI_COMM_WORLD);
            for (auto &id : description.coarse_cell_index_to_coarse_cell_id)
              id = 7 + 3 * id;
            for (auto &level : description.cell_infos)
              for (auto &cell : level)
                {
                  const CellId id(cell.id);
                  cell.id = CellId(7 + 3 * id.get_coarse_cell_id(),
                                   id.get_child_indices().size(),
                                   id.get_child_indices().data())
                              .to_binary<1>();
                }
            std::string reference_dofs;
            {
              Flow reference(MPI_COMM_WORLD);
              reference.initialize_params("distributed-rest.prm", "");
              reference.triangulation().create_triangulation(description);
              reference.setup(input);
              reference_dofs = dof_fingerprint(reference);
            }
            ParameterAcceptor::clear();
            Flow flow(MPI_COMM_WORLD);
            flow.initialize_params("distributed-graph.prm", "");
            flow.triangulation().create_triangulation(description);
            const auto before = fingerprint(flow);
            if (split)
              {
                // Independent malformed metadata: separate a true shared
                // junction without moving either point. Extra inlet0s make
                // all degree-one entries otherwise legitimate.
                auto      &edge = graph.cells.at(disconnected ? 10 : 7);
                auto      &node = disconnected ? edge.node0 : edge.node1;
                const auto old  = node;
                graph.points.push_back(graph.points[old]);
                node                   = graph.points.size() - 1;
                graph.boundaries[node] = 0;
                if (disconnected)
                  graph.boundaries[old] = 0;
                bool rejected = false;
                try
                  {
                    flow.setup(input, graph);
                    AssertThrow(
                      reference_dofs == dof_fingerprint(flow),
                      ExcMessage(
                        "Incidence metadata changed native flow DoFs."));
                  }
                catch (const std::exception &)
                  {
                    rejected = true;
                  }
                AssertThrow(Utilities::MPI::min(rejected ? 1u : 0u,
                                                MPI_COMM_WORLD) == 1,
                            ExcMessage("Split native junction was accepted."));
              }
            else
              {
                flow.setup(input, graph);
                AssertThrow(reference_dofs == dof_fingerprint(flow),
                            ExcMessage(
                              "Incidence metadata changed native flow DoFs."));
                auto state = flow.make_state();
                flow.initialize_state(state, 0.);
                AssertThrow(flow.has_admissible_wall_states(state) &&
                              std::isfinite(state.l2_norm()),
                            ExcInternalError());
                AssertThrow(flow.triangulation().n_global_active_cells() ==
                              edges * (1u << refinement),
                            ExcInternalError());
              }
            AssertThrow(before == fingerprint(flow),
                        ExcMessage("Incidence validation changed the mesh."));
          }
}
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      const std::string parameters =
        "subsection Functions\n set Inflow function = 0\nend\nsubsection MetricFlowSystem<1, 3>\n set Output directory = distributed-output\n set Wall model = Square root\n set Numerical flux type = HLL\n set Outlet boundary condition type = RCR\n set Initial state source = SOURCE\n set Finite element degree = 2\n set Verbosity (console depth) = 0\nend\n";
      for (const auto &source :
           {std::string("rest"), std::string("steady network")})
        {
          auto text = parameters;
          text.replace(text.find("SOURCE"), 6, source);
          std::ofstream prm(source == "rest" ? "distributed-rest.prm" :
                                               "distributed-graph.prm");
          prm << text;
        }
    }
  MPI_Barrier(MPI_COMM_WORLD);
  test();
  deallog.depth_file(10);
  deallog << "Distributed native graph endpoint identities: PASSED"
          << std::endl;
}
