#include <metric_flow_x/blood_flow_system.h>

int
main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  MetricFlowX::BloodFlowSystem<1, 3>       flow(MPI_COMM_WORLD);
  return flow.mpi_communicator() == MPI_COMM_WORLD ? 0 : 1;
}
