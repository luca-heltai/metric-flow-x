#include <deal.II/base/mpi.h>

#include <metric_flow_x/blood_flow_system.h>

#include <filesystem>

#include "rapidcsv.h"
#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

void
test()
{
  const auto        rank  = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const std::string dir   = "csv_output";
  const std::string label = "right, \"iliac\"\nterminal";
  if (rank == 0)
    {
      std::filesystem::remove_all(dir);
      std::filesystem::create_directories(dir);
      rapidcsv::Document csv;
      csv.SetColumnName(0, "boundary_id");
      csv.SetColumnName(1, "label");
      csv.SetCell(0, 0, 2);
      csv.SetCell(1, 0, label);
      csv.Save(dir + "/input.csv");
    }
  MPI_Barrier(MPI_COMM_WORLD);
  BloodFlowSystem<1, 3> flow;
  flow.initialize_params(PRM_DIR "constant.prm");
  flow.vtk_file_path = std::string(METRIC_FLOW_X_SOURCE_DIR) +
                       "/notebooks/bifurcation_network.vtk";
  flow.terminal_parameter_source    = "Parameters";
  flow.terminal_parameters_csv_file = dir + "/input.csv";
  flow.default_terminal_r1          = 2e7;
  flow.default_terminal_r2          = 3e7;
  flow.default_terminal_c           = 0;
  flow.default_terminal_p_out       = 321;
  flow.output_directory             = dir;
  flow.output_terminal_csv_file     = "terminals.csv";
  flow.n_global_refinements         = 2;
  flow.setup();
  auto y = flow.make_state();
  flow.initialize_state(y, 0.0);
  flow.open_output_files();
  flow.write_output_step(0.0, y);
  flow.write_output_step(0.125, y);
  auto pressure = flow.make_state();
  flow.compute_pressure(y, pressure);
  flow.output_results(y, pressure, 0);
  flow.close_csv_files();
  flow.mass_log.close();
  // The statistics writer must handle both an existing empty file and append.
  if (rank == 0)
    std::ofstream(dir + "/ida_statistics.csv");
  MPI_Barrier(MPI_COMM_WORLD);
  SUNDIALS::IDA<VectorType> ida(flow.ida_parameters, MPI_COMM_WORLD);
  flow.write_ida_statistics(ida, 1.2345678901234567);
  flow.write_ida_statistics(ida, 2.0);
  {
    TimerOutput::Scope timer(flow.computing_timer, "quoted, \"section\"\nname");
  }
  flow.write_scaling_timings();
  flow.write_scaling_timings();
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
    {
      rapidcsv::SeparatorParams separators;
      separators.mQuotedLinebreaks = true;
      const auto read              = [&](const std::string &name) {
        return rapidcsv::Document(dir + "/" + name,
                                  rapidcsv::LabelParams(),
                                  separators);
      };
      // Locate the parallel record without assuming deal.II's filename suffix.
      bool visualization_has_vessel_id = false;
      for (const auto &entry : std::filesystem::directory_iterator(dir))
        if (entry.path().extension() == ".pvtu")
          {
            std::ifstream     input(entry.path());
            const std::string text((std::istreambuf_iterator<char>(input)), {});
            visualization_has_vessel_id =
              text.find("Name=\"vessel_id\"") != std::string::npos;
          }
      AssertThrow(visualization_has_vessel_id, ExcInternalError());
      const auto traces = read("terminals.csv");
      AssertThrow(traces.GetRowCount() == 6, ExcInternalError());
      for (unsigned int row = 0; row < 6; ++row)
        AssertThrow(
          traces.GetCell<unsigned int>("boundary_id", row) == row % 3 &&
            traces.GetCell<double>("time_s", row) == (row < 3 ? 0. : .125),
          ExcMessage(
            "Bifurcation terminal snapshots must be ordered and complete."));
      const auto terminals = read("network_terminals.csv");
      AssertThrow(terminals.GetRowCount() == 2, ExcInternalError());
      for (std::size_t row = 0; row < terminals.GetRowCount(); ++row)
        {
          const auto bid = terminals.GetCell<unsigned int>("boundary_id", row);
          const auto vid = terminals.GetCell<unsigned int>("vessel_id", row);
          const std::string expected = bid == 2 ? label : "boundary_1";
          AssertThrow(terminals.GetCell<std::string>("label", row) ==
                          expected &&
                        terminals.GetCell<double>("R1", row) == 2e7 &&
                        terminals.GetCell<double>("R2", row) == 3e7 &&
                        terminals.GetCell<double>("C", row) == 0 &&
                        terminals.GetCell<double>("P_out", row) == 321,
                      ExcInternalError());
          for (unsigned int snapshot = 0; snapshot < 2; ++snapshot)
            AssertThrow(
              traces.GetCell<std::string>("label", snapshot * 3 + bid) ==
                  expected &&
                traces.GetCell<unsigned int>("vessel_id", snapshot * 3 + bid) ==
                  vid,
              ExcMessage(
                "Consolidated traces must retain outlet labels and vessel ids."));
          const auto probe =
            read("HDG_IDA_Vessel_" + std::to_string(vid) + ".csv");
          AssertThrow(probe.GetRowCount() == 2 &&
                        probe.GetCell<double>("time_s", 1) == 0.125,
                      ExcInternalError());
          for (std::size_t i = 0; i < 2; ++i)
            AssertThrow(probe.GetCell<std::string>(
                          "boundary_" + std::to_string(bid) + "_label", i) ==
                          expected,
                        ExcInternalError());
        }
      const auto mass = read("mass_conservation.csv");
      AssertThrow(mass.GetRowCount() == 2 && mass.GetColumnCount() == 6 &&
                    mass.GetCell<double>("t", 1) == 0.125,
                  ExcInternalError());
      const auto statistics = read("ida_statistics.csv");
      AssertThrow(statistics.GetRowCount() == 2 &&
                    statistics.GetColumnCount() == 15 &&
                    statistics.GetCell<double>("wall_time", 0) ==
                      1.2345678901234567 &&
                    statistics.GetCell<double>("wall_time", 1) == 2.0,
                  ExcInternalError());
      const auto   timings = read("scaling_timings.csv");
      unsigned int matches = 0;
      for (std::size_t row = 0; row < timings.GetRowCount(); ++row)
        if (timings.GetCell<std::string>("section", row) ==
            "quoted, \"section\"\nname")
          ++matches;
      AssertThrow(matches == 2, ExcInternalError());
      unsigned int probes = 0;
      for (const auto &entry : std::filesystem::directory_iterator(dir))
        if (entry.path().filename().string().find("HDG_IDA_Vessel_") == 0)
          ++probes;
      AssertThrow(probes == flow.vessel_map.size(), ExcInternalError());
    }
  deallog.depth_file(10);
  deallog << "CSV labels, escaping, append and MPI probe ownership: PASSED"
          << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
