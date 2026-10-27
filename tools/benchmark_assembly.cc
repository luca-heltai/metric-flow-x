#include <deal.II/base/mpi.h>

#include <metric_flow_x/blood_flow_system.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace dealii;
using namespace MetricFlowX;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  AssertThrow(
    argc == 3 || argc == 4,
    ExcMessage(
      "Usage: benchmark_assembly parameters.prm repetitions [snapshot_prefix]"));
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
        flow.get_system_dof_indices(cell, indices);
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
  const auto measure = [&](const auto          &operation,
                           const unsigned int   count,
                           std::vector<double> &rank_medians) {
    operation();
    std::vector<double> samples, local_samples;
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
        local_samples.push_back(seconds);
        samples.push_back(
          Utilities::MPI::max(seconds, flow.mpi_communicator()));
      }
    std::sort(samples.begin(), samples.end());
    std::sort(local_samples.begin(), local_samples.end());
    rank_medians =
      Utilities::MPI::gather(flow.mpi_communicator(), local_samples[1], 0);
    return samples;
  };
  std::vector<double> rank_residual, rank_jacobian, rank_derivative;
  const auto          res =
    measure([&]() { flow.assemble_residual(0.1, y, ydot, residual); },
            repetitions,
            rank_residual);
  const auto jac =
    measure([&]() { flow.assemble_state_jacobian(0.1, y, ydot); },
            std::max(1u, repetitions / 10),
            rank_jacobian);
  const auto mass =
    measure([&]() { flow.assemble_derivative_jacobian(0.1, y, ydot); },
            repetitions,
            rank_derivative);
  unsigned int owned_cells = 0, ghost_cells = 0;
  for (const auto &cell : flow.triangulation().active_cell_iterators())
    {
      owned_cells += cell->is_locally_owned();
      ghost_cells += cell->is_ghost();
    }
  const auto owned =
    Utilities::MPI::gather(flow.mpi_communicator(), owned_cells, 0);
  const auto ghosts =
    Utilities::MPI::gather(flow.mpi_communicator(), ghost_cells, 0);
  const auto owned_dofs =
    Utilities::MPI::gather(flow.mpi_communicator(),
                           flow.locally_owned_dofs().n_elements(),
                           0);
  const auto capacitor_dofs =
    Utilities::MPI::gather(flow.mpi_communicator(),
                           flow.locally_owned_dofs().n_elements() -
                             flow.dof_handler().n_locally_owned_dofs(),
                           0);
  const auto json_array = [](const auto &values) {
    std::ostringstream out;
    out << std::setprecision(17) << '[';
    for (size_t i = 0; i < values.size(); ++i)
      out << (i ? "," : "") << values[i];
    out << ']';
    return out.str();
  };
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
              << ",\"owned_dofs_per_rank\":" << json_array(owned_dofs)
              << ",\"capacitor_dofs_per_rank\":" << json_array(capacitor_dofs)
              << ",\"owned_cells_per_rank\":" << json_array(owned)
              << ",\"ghost_cells_per_rank\":" << json_array(ghosts)
              << ",\"residual_rank_median_s\":" << json_array(rank_residual)
              << ",\"jacobian_rank_median_s\":" << json_array(rank_jacobian)
              << ",\"derivative_rank_median_s\":" << json_array(rank_derivative)
              << ",\"residual_min_s\":" << res[0]
              << ",\"residual_median_s\":" << res[1]
              << ",\"residual_max_s\":" << res[2]
              << ",\"jacobian_min_s\":" << jac[0]
              << ",\"jacobian_median_s\":" << jac[1]
              << ",\"jacobian_max_s\":" << jac[2] << ",\"residual_l2\":" << norm
              << ",\"residual_checksum\":" << checksum
              << ",\"derivative_min_s\":" << mass[0]
              << ",\"derivative_median_s\":" << mass[1]
              << ",\"derivative_max_s\":" << mass[2] << "}\n";

  // Snapshots are outside the measured region. Compare the same MPI layout
  // between revisions: global numbering may differ with the number of ranks.
  if (argc == 4)
    {
      const auto rank =
        Utilities::MPI::this_mpi_process(flow.mpi_communicator());
      for (unsigned int state = 0; state < 3; ++state)
        {
          for (const auto &cell : flow.dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                flow.get_system_dof_indices(cell, indices);
                for (unsigned int i = 0; i < indices.size(); ++i)
                  {
                    const auto component =
                      flow.finite_element().system_to_component_index(i).first;
                    const double variation =
                      1.0 + 0.001 * ((indices[i] + state) % 13);
                    y(indices[i]) =
                      component % 2 == 0 ?
                        (1.01 + 0.02 * state) * variation *
                          flow.vessel_properties(cell->material_id()).a_d :
                        (state == 1 ? -0.1 : 0.1) * variation;
                  }
              }
          y.compress(VectorOperation::insert);
          for (const auto i : flow.locally_owned_dofs())
            ydot(i) = 0.02 * (1 + (i + state) % 7);
          ydot.compress(VectorOperation::insert);
          const double t = 0.1 + 0.03 * state;
          flow.assemble_residual(t, y, ydot, residual);
          flow.assemble_state_jacobian(t, y, ydot);
          flow.assemble_derivative_jacobian(t, y, ydot);
          const std::string prefix = std::string(argv[3]) + ".state" +
                                     std::to_string(state) + ".rank" +
                                     std::to_string(rank);
          std::ofstream vector(prefix + ".residual");
          AssertThrow(vector, ExcMessage("Cannot write residual snapshot."));
          vector << std::hexfloat;
          for (const auto i : flow.locally_owned_dofs())
            vector << i << ' ' << residual(i) << '\n';
          const auto dump = [&](const MatrixType  &matrix,
                                const std::string &name) {
            std::ofstream output(prefix + name);
            AssertThrow(output, ExcMessage("Cannot write matrix snapshot."));
            output << std::hexfloat;
            for (const auto row : flow.locally_owned_dofs())
              for (auto entry = matrix.begin(row); entry != matrix.end(row);
                   ++entry)
                output << row << ' ' << entry->column() << ' ' << entry->value()
                       << '\n';
          };
          dump(flow.state_jacobian_matrix(), ".state_jacobian");
          dump(flow.derivative_jacobian_matrix(), ".derivative_jacobian");
        }
    }
  flow.write_scaling_timings();
}
