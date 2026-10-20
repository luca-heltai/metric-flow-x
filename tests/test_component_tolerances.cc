#include <deal.II/base/mpi.h>

#include <metric_flow_x/blood_flow_system.h>

#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

void
test()
{
  BloodFlowSystem<1, 3> flow;
  flow.initialize_params(PRM_DIR "constant.prm");
  AssertThrow(!flow.use_componentwise_tolerances, ExcInternalError());
  flow.setup();
  flow.area_absolute_tolerance_fraction = 2e-3;
  flow.velocity_absolute_tolerance      = 3e-5;
  flow.capacitor_absolute_tolerance     = 2.0;
  const auto  tolerances                = flow.make_absolute_tolerances();
  const auto &vessel                    = flow.vessel_properties(0);
  for (const auto i : flow.locally_owned_dofs_)
    {
      const bool is_area = flow.component_dofs_[0].is_element(i) ||
                           flow.component_dofs_[2].is_element(i);
      const double expected = flow.rcr_dofs_owned.is_element(i) ?
                                2.0 :
                                (is_area ? 2e-3 * vessel.a_d : 3e-5);
      AssertThrow(tolerances(i) == expected, ExcInternalError());
    }

  auto state            = flow.make_state();
  state                 = 0.0;
  const double area     = 1.2 * vessel.a_d;
  const double velocity = -0.9 * flow.wave_speed(area, 0);
  for (unsigned int component = 0; component < 4; ++component)
    for (const auto i : flow.component_dofs_[component])
      if (flow.locally_owned_dofs_.is_element(i))
        state(i) = component % 2 == 0 ? area : velocity;
  state.compress(VectorOperation::insert);
  const auto maxima = flow.maximum_mach_numbers(state);
  AssertThrow(maxima.size() == 1, ExcInternalError());
  AssertThrow(std::abs(maxima.at(0) - 0.9) < 1e-12, ExcInternalError());
  flow.write_flow_regime_diagnostics = true;
  flow.output_directory              = "regime-output";
  flow.report_flow_regime(state, 0.0);
  flow.report_flow_regime(state, 0.1);
  flow.close_csv_files();
  if (Utilities::MPI::this_mpi_process(flow.mpi_communicator()) == 0)
    {
      std::ifstream input("regime-output/flow_regime.csv");
      std::string   line;
      unsigned int  lines = 0;
      while (std::getline(input, line))
        ++lines;
      AssertThrow(lines == 3, ExcInternalError());
    }
  deallog.depth_file(10);
  deallog << "Component tolerances and vessel Mach diagnostics: PASSED"
          << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
