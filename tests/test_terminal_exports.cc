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
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::filesystem::remove_all("terminal-output");
      std::filesystem::remove_all("terminal-disabled");
      std::filesystem::remove_all("terminal-blocked");
    }
  MPI_Barrier(MPI_COMM_WORLD);
  BloodFlowSystem<1, 3> flow;
  flow.initialize_params(PRM_DIR "constant.prm");
  flow.setup();
  AssertThrow(flow.rcr_map.at(1).label == "boundary_1", ExcInternalError());
  const std::string label  = "right, \"iliac\"\nterminal";
  flow.rcr_map.at(1).label = label;
  auto y                   = flow.make_state();
  flow.initialize_state(y, 0.0);
  const double     area     = 1.2 * flow.vessel_properties(0).a_d;
  constexpr double velocity = 0.13;
  flow.set_external_pressure_provider([](const auto &) { return 234.0; });
  for (const auto &cell : flow.dof_handler_.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto face : cell->face_indices())
        if (cell->face(face)->at_boundary())
          {
            const auto &trace =
              flow.face_dof_map.at(flow.canonical_face_key(cell, face));
            y(trace.a_hat_dof) = area;
            y(trace.u_hat_dof) = velocity;
          }
  y.compress(VectorOperation::insert);
  flow.output_terminal_csv_file = "boundary_traces.csv";
  flow.output_directory         = "terminal-output";
  flow.open_csv_files();
  AssertThrow(Utilities::MPI::sum(static_cast<unsigned int>(
                                    flow.csv_terminal.is_open()),
                                  flow.mpi_communicator()) == 1,
              ExcInternalError());
  flow.write_csv_row(0.005, y);
  flow.write_csv_row(0.125, y);
  flow.close_csv_files();
  MPI_Barrier(flow.mpi_communicator());
  if (Utilities::MPI::this_mpi_process(flow.mpi_communicator()) == 0)
    {
      rapidcsv::SeparatorParams separators;
      separators.mQuotedLinebreaks = true;
      const rapidcsv::Document input("terminal-output/boundary_traces.csv",
                                     rapidcsv::LabelParams(),
                                     separators);
      const std::vector<std::string> header{"time_s",
                                            "pressure_Pa",
                                            "flow_m3_s",
                                            "boundary_id",
                                            "vessel_id",
                                            "label"};
      AssertThrow(input.GetColumnNames() == header && input.GetRowCount() == 4,
                  ExcInternalError());
      for (unsigned int row = 0; row < 4; ++row)
        {
          const unsigned int boundary = row % 2;
          AssertThrow(input.GetCell<std::string>("label", row) ==
                        (boundary == 0 ? "boundary_0" : label),
                      ExcInternalError());
          AssertThrow(input.GetCell<double>("time_s", row) ==
                          (row < 2 ? .005 : .125) &&
                        input.GetCell<unsigned int>("boundary_id", row) ==
                          boundary &&
                        input.GetCell<unsigned int>("vessel_id", row) == 0,
                      ExcInternalError());
          AssertThrow(std::abs(input.GetCell<double>("pressure_Pa", row) -
                               flow.pressure(area, 0) - 234.) < 1e-9,
                      ExcInternalError());
          AssertThrow(std::abs(input.GetCell<double>("flow_m3_s", row) -
                               (boundary == 0 ? -1 : 1) * area * velocity) <
                        1e-16,
                      ExcInternalError());
        }
      for (const auto &entry :
           std::filesystem::directory_iterator("terminal-output"))
        AssertThrow(entry.path().filename().string().find("terminal_") != 0,
                    ExcMessage("Per-terminal CSV files must not be created."));
    }
  flow.output_terminal_csv_file.clear();
  flow.output_directory = "terminal-disabled";
  flow.open_csv_files();
  flow.write_csv_row(.25, y);
  flow.close_csv_files();
  MPI_Barrier(flow.mpi_communicator());
  AssertThrow(!std::filesystem::exists("terminal-disabled/boundary_traces.csv"),
              ExcMessage("Disabled terminal output created a file."));
  if (Utilities::MPI::this_mpi_process(flow.mpi_communicator()) == 0)
    std::filesystem::create_directories("terminal-blocked/boundary_traces.csv");
  MPI_Barrier(flow.mpi_communicator());
  flow.output_terminal_csv_file = "boundary_traces.csv";
  flow.output_directory         = "terminal-blocked";
  bool rejected                 = false;
  try
    {
      flow.open_csv_files();
    }
  catch (const std::exception &)
    {
      rejected = true;
    }
  AssertThrow(Utilities::MPI::sum(static_cast<unsigned int>(rejected),
                                  flow.mpi_communicator()) ==
                Utilities::MPI::n_mpi_processes(flow.mpi_communicator()),
              ExcMessage("Terminal file-open failure must reach every rank."));
  deallog.depth_file(10);
  deallog
    << "Consolidated terminal traces, ordering, labels and unique MPI writer: PASSED"
    << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
