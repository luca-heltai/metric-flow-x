// The wave-speed derivative must follow the implemented square-root tube law,
// including when the registered expression constant m is changed.
#include <deal.II/base/mpi.h>

#include <metric_flow_x/blood_flow_system.h>

#include <cmath>

#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

void
test()
{
  BloodFlowSystem<1, 3> flow;
  flow.initialize_params(PRM_DIR "constant.prm");
  flow.setup();
  const auto &vessel = flow.vessel_properties(0);
  for (const double exponent : {0.5, 1.0, 0.25})
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::ofstream parameters("wave_exponent.prm");
          std::ifstream original(PRM_DIR "constant.prm");
          std::string   line;
          while (std::getline(original, line))
            if (line.find("Tube law exponent (m)") != std::string::npos)
              parameters << "  set Tube law exponent (m) = " << exponent
                         << '\n';
            else
              parameters << line << '\n';
        }
      MPI_Barrier(MPI_COMM_WORLD);
      flow.initialize_params("wave_exponent.prm");
      AssertThrow(flow.par["m"] == exponent, ExcInternalError());
      for (const double rest_area : {vessel.a_d, 2 * vessel.a_d})
        for (const double factor : {0.25, 1.0, 4.0})
          {
            // All samples and both finite-difference points exceed the floor.
            const double area = factor * rest_area;
            const double step = 1e-5 * area;
            const double finite_difference =
              (flow.compute_wave_speed(area + step, 0, rest_area) -
               flow.compute_wave_speed(area - step, 0, rest_area)) /
              (2 * step);
            const double derivative =
              flow.compute_wave_speed_derivative(area, 0, rest_area);
            AssertThrow(std::abs(derivative / finite_difference - 1) < 1e-8,
                        ExcMessage(
                          "Wave-speed derivative disagrees with central "
                          "differences: m=" +
                          std::to_string(exponent) + ", derivative/FD=" +
                          std::to_string(derivative / finite_difference)));
            const double beta =
              (4 * std::sqrt(numbers::PI) / 3) * vessel.E * vessel.h_wall;
            const double analytical =
              std::sqrt(beta / (2 * flow.par["rho"] * rest_area)) *
              std::pow(area, -0.75) / 4;
            AssertThrow(std::abs(derivative / analytical - 1) < 1e-13,
                        ExcMessage("Incorrect analytical wave derivative."));
            if (rest_area == vessel.a_d)
              AssertThrow(flow.compute_wave_speed_derivative(area, 0) ==
                            derivative,
                          ExcInternalError());
          }
    }
  deallog.depth_file(10);
  deallog << "Square-root tube-law wave derivative for m=0.5,1,0.25: PASSED"
          << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  mpi_initlog();
  test();
}
