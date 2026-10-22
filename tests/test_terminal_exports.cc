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
  flow.write_terminal_csv = true;
  flow.output_directory   = "terminal-output";
  flow.open_csv_files();
  AssertThrow(Utilities::MPI::sum(static_cast<unsigned int>(
                                    flow.csv_terminal.size()),
                                  flow.mpi_communicator()) == 2,
              ExcInternalError());
  flow.write_csv_row(0.005, y);
  flow.close_csv_files();
  MPI_Barrier(flow.mpi_communicator());
  if (Utilities::MPI::this_mpi_process(flow.mpi_communicator()) == 0)
    for (const unsigned int boundary : {0, 1})
      {
        rapidcsv::SeparatorParams separators;
        separators.mQuotedLinebreaks = true;
        const rapidcsv::Document       input("terminal-output/terminal_" +
                                         std::to_string(boundary) + ".csv",
                                       rapidcsv::LabelParams(),
                                       separators);
        const std::vector<std::string> header{"time_s",
                                              "pressure_Pa",
                                              "flow_m3_s",
                                              "boundary_id",
                                              "vessel_id",
                                              "label"};
        AssertThrow(input.GetColumnNames() == header &&
                      input.GetRowCount() == 1,
                    ExcInternalError());
        const auto time     = input.GetCell<double>("time_s", 0);
        const auto pressure = input.GetCell<double>("pressure_Pa", 0);
        const auto flux     = input.GetCell<double>("flow_m3_s", 0);
        const auto bid      = input.GetCell<unsigned int>("boundary_id", 0);
        const auto vid      = input.GetCell<unsigned int>("vessel_id", 0);
        AssertThrow(input.GetCell<std::string>("label", 0) ==
                      (boundary == 0 ? "boundary_0" : label),
                    ExcInternalError());
        AssertThrow(time == 0.005 && bid == boundary && vid == 0,
                    ExcInternalError());
        AssertThrow(std::abs(pressure - flow.pressure(area, 0) - 234.0) < 1e-9,
                    ExcInternalError());
        AssertThrow(std::abs(flux - (boundary == 0 ? -1 : 1) * area *
                                      velocity) < 1e-16,
                    ExcInternalError());
      }
  deallog.depth_file(10);
  deallog << "Terminal trace values and unique MPI file ownership: PASSED"
          << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
