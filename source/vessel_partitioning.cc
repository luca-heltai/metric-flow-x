// SPDX-License-Identifier: MIT
#include <deal.II/base/mpi.h>

#include <metric_flow_x/blood_flow_system.h>

#include <algorithm>
#include <cstdint>
#include <numeric>

#ifdef DEAL_II_WITH_METIS
extern "C"
{
#  include <metis.h>
}
#endif

#include "rapidcsv.h"

namespace MetricFlowX
{
  using namespace dealii;
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::partition_vessels(
    Triangulation<dim, spacedim> &mesh) const
  {
    static_assert(dim == 1, "Vessel graph partitioning requires a line mesh.");
    AssertThrow(mesh.n_active_cells() > 0,
                ExcMessage("Cannot partition an empty network."));
    if (n_mpi_processes == 1)
      {
        for (const auto &cell : mesh.active_cell_iterators())
          cell->set_subdomain_id(0);
        return;
      }
#ifndef DEAL_II_WITH_METIS
    AssertThrow(false,
                ExcMessage("Graph partitioning requires deal.II with METIS. "
                           "Select Mesh partitioning = zorder."));
#else
    // Keep each input vessel's descendants together when it fits in a rank.
    // Oversized vessels are split into contiguous chunks of their 1-D tree.
    const std::uint64_t fe_cost = 2ull * (fe_degree + 1) + 4;
    const auto          target_cells =
      std::max<std::uint64_t>(1, mesh.n_active_cells() / n_mpi_processes);
    std::map<
      types::coarse_cell_id,
      std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>>
      vessels;
    for (const auto &cell : mesh.active_cell_iterators())
      vessels[cell->id().get_coarse_cell_id()].push_back(cell);
    std::vector<unsigned int>                  unit(mesh.n_active_cells());
    std::vector<idx_t>                         weights;
    std::vector<std::vector<unsigned int>>     incident(mesh.n_vertices());
    std::map<types::boundary_id, unsigned int> terminals;
    for (const auto &[id, cells] : vessels)
      for (std::size_t begin = 0; begin < cells.size(); begin += target_cells)
        {
          const unsigned int index = weights.size();
          std::uint64_t      cost  = 0;
          for (std::size_t j = begin;
               j < std::min<std::size_t>(cells.size(), begin + target_cells);
               ++j)
            {
              const auto &cell                = cells[j];
              unit[cell->active_cell_index()] = index;
              cost += fe_cost;
              for (const auto f : cell->face_indices())
                {
                  incident[cell->vertex_index(f)].push_back(index);
                  if (cell->face(f)->at_boundary())
                    {
                      const auto bid = cell->face(f)->boundary_id();
                      if (rcr_map.count(bid))
                        {
                          AssertThrow(terminals.emplace(bid, index).second,
                                      ExcMessage(
                                        "Repeated terminal boundary ID."));
                          if (rcr_map.at(bid).C > 0. && outlet_type == "RCR")
                            ++cost;
                        }
                    }
                }
            }
          AssertThrow(cost <= static_cast<std::uint64_t>(
                                std::numeric_limits<idx_t>::max()),
                      ExcMessage("Vessel weight exceeds METIS integer range."));
          weights.push_back(static_cast<idx_t>(cost));
        }
    AssertThrow(weights.size() <=
                  static_cast<std::uint64_t>(std::numeric_limits<idx_t>::max()),
                ExcMessage("Vessel graph exceeds METIS integer range."));
    std::vector<std::map<unsigned int, idx_t>> graph(weights.size());
    const auto add_edge = [&](const unsigned int a,
                              const unsigned int b,
                              const unsigned int weight) {
      if (a != b)
        {
          graph[a][b] += weight;
          graph[b][a] += weight;
        }
    };
    for (auto &neighbors : incident)
      {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                        neighbors.end());
        for (unsigned int i = 0; i < neighbors.size(); ++i)
          for (unsigned int j = i + 1; j < neighbors.size(); ++j)
            add_edge(neighbors[i], neighbors[j], partition_junction_weight);
      }
    if (!terminal_connections_csv_file.empty())
      {
        const rapidcsv::Document csv(terminal_connections_csv_file);
        for (std::size_t row = 0; row < csv.GetRowCount(); ++row)
          {
            const auto a = csv.GetCell<types::boundary_id>("boundary0", row);
            const auto b = csv.GetCell<types::boundary_id>("boundary1", row);
            AssertThrow(
              terminals.count(a) && terminals.count(b),
              ExcMessage(
                "Partition connection references an unknown terminal."));
            add_edge(terminals.at(a),
                     terminals.at(b),
                     partition_terminal_weight);
          }
      }

    // METIS's contiguous option requires a connected input graph. Geometrically
    // separate vascular compartments may become connected by peripheral links.
    std::vector<bool>         seen(graph.size(), false);
    std::vector<unsigned int> stack{0};
    seen[0] = true;
    while (!stack.empty())
      {
        const auto a = stack.back();
        stack.pop_back();
        for (const auto &[b, weight] : graph[a])
          if (!seen[b])
            {
              seen[b] = true;
              stack.push_back(b);
            }
      }
    std::vector<idx_t> offsets{0}, neighbors, edge_weights;
    for (const auto &row : graph)
      {
        for (const auto &[other, weight] : row)
          {
            neighbors.push_back(other);
            edge_weights.push_back(weight);
          }
        AssertThrow(neighbors.size() <= static_cast<std::uint64_t>(
                                          std::numeric_limits<idx_t>::max()),
                    ExcMessage(
                      "Vessel adjacency exceeds METIS integer range."));
        offsets.push_back(neighbors.size());
      }
    idx_t vertices = weights.size(), constraints = 1;
    idx_t partitions = std::min<std::uint64_t>(n_mpi_processes, weights.size());
    std::vector<idx_t> labels(vertices, 0);
    // Only root runs METIS; the broadcast makes the assignment identical even
    // with library builds whose random state is process dependent.
    if (partitions == vertices)
      std::iota(labels.begin(), labels.end(), 0);
    else if (partitions > 1)
      {
        int status = METIS_OK;
        if (this_mpi_process == 0)
          {
            idx_t options[METIS_NOPTIONS];
            METIS_SetDefaultOptions(options);
            options[METIS_OPTION_SEED]    = 42;
            options[METIS_OPTION_NCUTS]   = 4;
            options[METIS_OPTION_MINCONN] = 1;
            options[METIS_OPTION_CONTIG] =
              std::all_of(seen.begin(), seen.end(), [](bool v) { return v; });
            real_t imbalance = 1. + partition_imbalance;
            idx_t  edgecut   = 0;
            status           = METIS_PartGraphKway(&vertices,
                                         &constraints,
                                         offsets.data(),
                                         neighbors.data(),
                                         weights.data(),
                                         nullptr,
                                         edge_weights.data(),
                                         &partitions,
                                         nullptr,
                                         &imbalance,
                                         options,
                                         &edgecut,
                                         labels.data());
            if (status == METIS_OK)
              {
                // With indivisible weights and an infeasible balance target,
                // METIS can leave parts empty even when enough units exist.
                // Move a unit from the heaviest non-singleton part. This cannot
                // increase the maximum load. Prefer a removal that leaves as
                // few components as possible in the donor part.
                std::vector<unsigned int>  counts(partitions, 0);
                std::vector<std::uint64_t> loads(partitions, 0);
                for (unsigned int i = 0; i < labels.size(); ++i)
                  {
                    ++counts[labels[i]];
                    loads[labels[i]] += weights[i];
                  }
                for (idx_t empty = 0; empty < partitions; ++empty)
                  if (counts[empty] == 0)
                    {
                      idx_t donor = 0;
                      for (idx_t r = 0; r < partitions; ++r)
                        if (counts[r] > 1 &&
                            (counts[donor] <= 1 || loads[r] > loads[donor]))
                          donor = r;
                      unsigned int best            = labels.size();
                      unsigned int best_components = labels.size();
                      for (unsigned int candidate = 0;
                           candidate < labels.size();
                           ++candidate)
                        if (labels[candidate] == donor)
                          {
                            std::vector<bool> visited(labels.size(), false);
                            visited[candidate]      = true;
                            unsigned int components = 0;
                            for (unsigned int v = 0; v < labels.size(); ++v)
                              if (labels[v] == donor && !visited[v])
                                {
                                  ++components;
                                  std::vector<unsigned int> pending{v};
                                  visited[v] = true;
                                  while (!pending.empty())
                                    {
                                      const auto a = pending.back();
                                      pending.pop_back();
                                      for (const auto &[b, weight] : graph[a])
                                        if (labels[b] == donor && !visited[b])
                                          {
                                            visited[b] = true;
                                            pending.push_back(b);
                                          }
                                    }
                                }
                            if (best == labels.size() ||
                                components < best_components ||
                                (components == best_components &&
                                 weights[candidate] < weights[best]))
                              {
                                best            = candidate;
                                best_components = components;
                              }
                          }
                      labels[best] = empty;
                      --counts[donor];
                      ++counts[empty];
                      loads[donor] -= weights[best];
                      loads[empty] += weights[best];
                    }
              }
          }
        status = Utilities::MPI::broadcast(mpi_communicator_, status, 0);
        AssertThrow(status == METIS_OK,
                    ExcMessage("METIS vessel partitioning failed."));
        labels = Utilities::MPI::broadcast(mpi_communicator_, labels, 0);
      }
    for (const auto &cell : mesh.active_cell_iterators())
      cell->set_subdomain_id(labels[unit[cell->active_cell_index()]]);
#endif
  }

  template void
  BloodFlowSystem<1, 3>::partition_vessels(Triangulation<1, 3> &) const;
} // namespace MetricFlowX
