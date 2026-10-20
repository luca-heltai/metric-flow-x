#include <deal.II/base/mpi.h>

#include <metric_flow_x/blood_flow_system.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace dealii;
using namespace MetricFlowX;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  AssertThrow(argc == 3,
              ExcMessage(
                "Usage: benchmark_assembly parameters.prm repetitions"));
  const unsigned int repetitions = std::stoul(argv[2]);
  AssertThrow(repetitions > 0, ExcMessage("Repetitions must be positive."));
  const auto            start = std::chrono::steady_clock::now();
  BloodFlowSystem<1, 3> flow;
  flow.initialize_params(argv[1]);
  flow.setup();
  const double startup =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
  auto y        = flow.make_state();
  auto ydot     = flow.make_state();
  auto residual = flow.make_state();
  y             = 0.0;
  ydot          = 0.02;
  std::vector<types::global_dof_index> indices(
    flow.finite_element().n_dofs_per_cell());
  for (const auto &cell : flow.dof_handler().active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell->get_dof_indices(indices);
        for (unsigned int i = 0; i < indices.size(); ++i)
          {
            const auto component =
              flow.finite_element().system_to_component_index(i).first;
            y(indices[i]) =
              component % 2 == 0 ?
                1.01 * flow.vessel_properties(cell->material_id()).a_d :
                0.1;
          }
      }
  y.compress(VectorOperation::insert);
  const auto measure = [&](const auto &operation, const unsigned int count) {
    operation();
    std::vector<double> samples;
    for (unsigned int sample = 0; sample < 3; ++sample)
      {
        MPI_Barrier(flow.mpi_communicator());
        const auto begin = std::chrono::steady_clock::now();
        for (unsigned int iteration = 0; iteration < count; ++iteration)
          operation();
        const double seconds = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - begin)
                                 .count() /
                               count;
        samples.push_back(
          Utilities::MPI::max(seconds, flow.mpi_communicator()));
      }
    std::sort(samples.begin(), samples.end());
    return samples;
  };
  const auto res =
    measure([&]() { flow.assemble_residual(0.1, y, ydot, residual); },
            repetitions);
  const auto jac =
    measure([&]() { flow.assemble_state_jacobian(0.1, y, ydot); },
            std::max(1u, repetitions / 10));
  const double norm           = residual.l2_norm();
  double       checksum_local = 0.0;
  for (const auto i : flow.locally_owned_dofs())
    checksum_local += (i + 1) * residual(i);
  const double checksum =
    Utilities::MPI::sum(checksum_local, flow.mpi_communicator());
  if (Utilities::MPI::this_mpi_process(flow.mpi_communicator()) == 0)
    std::cout << std::setprecision(17)
              << "BENCHMARK {\"parameter_file\":" << std::quoted(argv[1])
              << ",\"deal_ii\":" << std::quoted(DEAL_II_PACKAGE_VERSION)
              << ",\"compiler\":" << std::quoted(__VERSION__)
              << ",\"mpi_ranks\":"
              << Utilities::MPI::n_mpi_processes(flow.mpi_communicator())
              << ",\"cells\":" << flow.triangulation().n_global_active_cells()
              << ",\"dofs\":" << y.size() << ",\"startup_s\":" << startup
              << ",\"residual_min_s\":" << res[0]
              << ",\"residual_median_s\":" << res[1]
              << ",\"residual_max_s\":" << res[2]
              << ",\"jacobian_min_s\":" << jac[0]
              << ",\"jacobian_median_s\":" << jac[1]
              << ",\"jacobian_max_s\":" << jac[2] << ",\"residual_l2\":" << norm
              << ",\"residual_checksum\":" << checksum << "}\n";
}
