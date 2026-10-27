// SPDX-License-Identifier: MIT
// Keep small vessels intact, split a large refined vessel, avoid empty ranks
// when there are enough cells, and use the same labels on repeated calls.
#include <metric_flow_x/blood_flow_system.h>

#include "tests.h"
using namespace MetricFlowX;

void
test()
{
#ifdef DEAL_II_WITH_METIS
  BloodFlowSystem<1, 3> flow;
  const auto            ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  for (const unsigned int coarse : {1u, 3u, 8u, 37u})
    for (const unsigned int refinements : {0u, 1u, 3u})
      {
        Triangulation<1, 3> mesh;
        GridGenerator::subdivided_hyper_cube(mesh, coarse);
        mesh.refine_global(refinements);
        flow.partition_vessels(mesh);
        std::vector<unsigned int>                     counts(ranks, 0), labels;
        std::map<types::coarse_cell_id, unsigned int> coarse_owner;
        for (const auto &cell : mesh.active_cell_iterators())
          {
            AssertThrow(cell->subdomain_id() < ranks, ExcInternalError());
            ++counts[cell->subdomain_id()];
            labels.push_back(cell->subdomain_id());
            if (coarse >= ranks)
              {
                const auto id = cell->id().get_coarse_cell_id();
                const auto [it, inserted] =
                  coarse_owner.emplace(id, cell->subdomain_id());
                AssertThrow(inserted || it->second == cell->subdomain_id(),
                            ExcMessage("Small vessel split between ranks."));
              }
          }
        AssertThrow(std::count(counts.begin(), counts.end(), 0) ==
                      ranks -
                        std::min<unsigned int>(ranks, mesh.n_active_cells()),
                    ExcMessage("Unexpected empty vessel partition."));
        flow.partition_vessels(mesh);
        unsigned int i = 0;
        for (const auto &cell : mesh.active_cell_iterators())
          AssertThrow(labels[i++] == cell->subdomain_id(),
                      ExcMessage("Nondeterministic vessel partition."));
      }
#endif
  deallog << "OK" << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
