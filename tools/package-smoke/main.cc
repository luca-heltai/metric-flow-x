#include <metric_flow_x/blood_flow_system.h>

#include <cmath>
#include <fstream>

int
main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  MetricFlowX::BloodFlowSystem<1, 3>       flow(MPI_COMM_WORLD);
  {
    std::ofstream file("package_inflow.csv");
    file << "flow_m3_s,time_s\n0,0\n2e-8,1\n";
  }
  MetricFlowX::InflowSeries inflow;
  inflow.read("package_inflow.csv");
  return flow.mpi_communicator() == MPI_COMM_WORLD &&
             std::abs(inflow.value(0.5) - 1e-8) < 1e-20 ?
           0 :
           1;
}
