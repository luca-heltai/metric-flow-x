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
  const double       area           = 1.1 * flow.vessel_properties(0).a_d;
  const double       before         = flow.wave_speed(area, 0);
  const double       density_before = flow.par["rho"];
  std::ifstream      original(PRM_DIR "constant.prm");
  std::ostringstream changed;
  std::string        line;
  while (std::getline(original, line))
    {
      if (line.find("Density (rho)") != std::string::npos)
        changed << "  set Density (rho) = " << std::setprecision(17)
                << 2 * density_before << '\n';
      else
        changed << line << '\n';
    }
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::ofstream output("changed_density.prm");
      output << changed.str();
    }
  MPI_Barrier(MPI_COMM_WORLD);
  flow.initialize_params("changed_density.prm");
  AssertThrow(std::abs(flow.wave_speed(area, 0) / before - 1 / std::sqrt(2.0)) <
                1e-14,
              ExcMessage("Physical constants must observe parameter reloads."));
  deallog.depth_file(10);
  deallog << "Physical constants follow parameter reloads: PASSED" << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
