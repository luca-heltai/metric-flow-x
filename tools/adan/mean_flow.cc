// SPDX-License-Identifier: MIT
// Full arterial/venous reference-geometry resistive calculation. The graph
// preparation and the distinction from the pulsatile ADAVN model are documented
// in tutorials/05_adavn_full/README.md.
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include <metric_flow_x/steady_network.h>
#include <rapidcsv.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

using namespace dealii;
using namespace MetricFlowX;

int
main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
      AssertThrow(argc == 2,
                  ExcMessage("Usage: adavn_mean_flow parameters.prm"));
      AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
                  ExcMessage("The native resistive graph solver is serial."));
      std::string  edges_file, outlets_file, nodes_output, flows_output;
      unsigned int inlet_node = 0, repetitions = 5;
      double inflow = 0., reservoir_pressure = 0., reservoir_resistance = 0.;
      ParameterHandler prm;
      prm.add_parameter("Edges CSV file", edges_file);
      prm.add_parameter("Outlets CSV file", outlets_file);
      prm.add_parameter("Node output CSV file", nodes_output);
      prm.add_parameter("Flow output CSV file", flows_output);
      prm.add_parameter("Inlet node", inlet_node);
      prm.add_parameter("Inflow (m3/s)", inflow);
      prm.add_parameter("Reservoir pressure (Pa)", reservoir_pressure);
      prm.add_parameter("Reservoir resistance (Pa s/m3)", reservoir_resistance);
      prm.add_parameter("Repetitions", repetitions);
      prm.parse_input(argv[1]);
      AssertThrow(repetitions > 0, ExcMessage("Repetitions must be positive."));

      rapidcsv::Document         table(edges_file);
      std::vector<HydraulicEdge> edges;
      unsigned int               n_nodes = 0;
      for (size_t i = 0; i < table.GetRowCount(); ++i)
        {
          HydraulicEdge edge{table.GetCell<unsigned int>("node0", i),
                             table.GetCell<unsigned int>("node1", i),
                             table.GetCell<double>("resistance_Pa_s_m3", i)};
          n_nodes = std::max(n_nodes, std::max(edge.node0, edge.node1) + 1);
          edges.push_back(edge);
        }
      AssertThrow(inlet_node < n_nodes, ExcMessage("Invalid inlet node."));
      std::vector<double> injections(n_nodes, 0.);
      injections[inlet_node] = inflow;
      rapidcsv::Document           outlet_table(outlets_file);
      std::vector<HydraulicOutlet> outlets;
      for (size_t i = 0; i < outlet_table.GetRowCount(); ++i)
        outlets.push_back({outlet_table.GetCell<unsigned int>("node", i),
                           reservoir_resistance,
                           reservoir_pressure});

      // One unmeasured warmup, then complete assemble/factorize/solve timings.
      auto state = solve_steady_network(edges, injections, outlets);
      std::vector<double> seconds;
      for (unsigned int i = 0; i < repetitions; ++i)
        {
          const auto start = std::chrono::steady_clock::now();
          state            = solve_steady_network(edges, injections, outlets);
          seconds.push_back(std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - start)
                              .count());
        }
      std::sort(seconds.begin(), seconds.end());
      std::ofstream nodes(nodes_output), flows(flows_output);
      AssertThrow(nodes && flows, ExcMessage("Cannot open result CSV files."));
      nodes << "node,pressure_Pa,mass_balance_m3_s\n" << std::setprecision(17);
      flows << "edge,flow_m3_s\n" << std::setprecision(17);
      double max_balance = 0.;
      for (unsigned int i = 0; i < n_nodes; ++i)
        {
          nodes << i << ',' << state.pressures[i] << ','
                << state.mass_balance_residual[i] << '\n';
          max_balance =
            std::max(max_balance, std::abs(state.mass_balance_residual[i]));
        }
      for (unsigned int i = 0; i < edges.size(); ++i)
        flows << i << ',' << state.edge_flows[i] << '\n';
      const double outflow = std::accumulate(state.outlet_flows.begin(),
                                             state.outlet_flows.end(),
                                             0.);
      std::cout << std::setprecision(17) << "MEAN_FLOW {\"nodes\":" << n_nodes
                << ",\"edges\":" << edges.size()
                << ",\"inflow_m3_s\":" << inflow
                << ",\"outflow_m3_s\":" << outflow
                << ",\"max_mass_balance_m3_s\":" << max_balance
                << ",\"solve_min_s\":" << seconds.front()
                << ",\"solve_median_s\":" << seconds[seconds.size() / 2]
                << ",\"solve_max_s\":" << seconds.back() << "}\n";
    }
  catch (const std::exception &error)
    {
      std::cerr << error.what() << '\n';
      return 1;
    }
}
