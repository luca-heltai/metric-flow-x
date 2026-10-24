// The tube-law stiffness is linear in E and h. Rebuilding network data must
// refresh it, while the public vessel-property view retains the input values.
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
  flow.setup();
  const auto   original   = flow.vessel_properties(0);
  const double area       = 1.1 * original.a_d;
  const double derivative = flow.compute_pressure_derivative(area, 0);
  const double wave       = flow.wave_speed(area, 0);
  const double pressure =
    flow.compute_pressure_value(area, 0) - original.p0 - original.p_d;
  for (const double factor : {4.0, 0.25, 2.0})
    {
      flow.cell_E[0]      = factor * original.E;
      flow.cell_h_wall[0] = 2.0 * original.h_wall;
      flow.build_global_vessel_data();
      const double scale = 2.0 * factor;
      AssertThrow(flow.vessel_properties(0).E == factor * original.E,
                  ExcInternalError());
      AssertThrow(std::abs(flow.compute_pressure_derivative(area, 0) /
                             derivative -
                           scale) < 1e-13,
                  ExcInternalError());
      AssertThrow(std::abs(flow.wave_speed(area, 0) / wave - std::sqrt(scale)) <
                    1e-13,
                  ExcInternalError());
      const double actual =
        flow.compute_pressure_value(area, 0) - original.p0 - original.p_d;
      AssertThrow(std::abs(actual / pressure - scale) < 1e-12,
                  ExcInternalError());
    }
  deallog.depth_file(10);
  deallog << "Tube-law stiffness follows material rebuilds: PASSED"
          << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  mpi_initlog();
  test();
}
