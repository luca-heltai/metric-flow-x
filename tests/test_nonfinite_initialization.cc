// A provider valid at mesh endpoints but invalid at volume quadrature points
// must never produce an accepted initial pair. One cell also leaves empty
// ranks in the MPI runs.
#include <deal.II/base/mpi.h>

#include <metric_flow_x/blood_flow_system.h>

#include <fstream>
#include <limits>
#include <string>

#include "tests.h"
using namespace dealii;
using namespace MetricFlowX;
void
test()
{
  const MPI_Comm comm = MPI_COMM_WORLD;
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    {
      std::ofstream file("nonfinite_initialization.prm");
      file << "subsection MetricFlowSystem<1, 3>\n"
           << " set Vtk file path for mesh input = " << METRIC_FLOW_X_SOURCE_DIR
           << "/parameters/single_vessel.vtk\n"
           << " set Finite element degree = 1\n"
           << " set Number of global refinement = 0\n"
           << " set Outlet boundary condition type = RCR\n"
           << " set Terminal parameter source = Parameters\n"
           << " set Default terminal R1 = 1e7\n"
           << " set Default terminal R2 = 1e8\n"
           << " set Default terminal C = 1e-8\nend\n";
    }
  MPI_Barrier(comm);
  BloodFlowSystem<1, 3> flow(comm);
  flow.initialize_params("nonfinite_initialization.prm");
  flow.setup();
  unsigned int owned_cells = 0;
  for (const auto &cell : flow.dof_handler_.active_cell_iterators())
    owned_cells += cell->is_locally_owned();
  if (Utilities::MPI::n_mpi_processes(comm) > 1)
    AssertThrow(Utilities::MPI::min(owned_cells, comm) == 0,
                ExcMessage("The MPI regression must include an empty rank."));
  flow.set_external_pressure_provider([](const auto &evaluation) {
    return evaluation.point[0] > 0. && evaluation.point[0] < .24137 ?
             std::numeric_limits<double>::quiet_NaN() :
             0.;
  });
  auto         state = flow.make_state(), derivative = flow.make_state();
  unsigned int rejected = 0;
  try
    {
      flow.initialize_consistent_state(state, derivative, 0.);
    }
  catch (const std::exception &error)
    {
      rejected =
        std::string(error.what()).find("Non-finite") != std::string::npos;
    }
  AssertThrow(Utilities::MPI::min(rejected, comm) == 1,
              ExcMessage(
                "Every rank must reject non-finite initial volume data."));
  deallog.depth_file(10);
  deallog << "Non-finite initial volume data rejected collectively: PASSED"
          << std::endl;
}
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  mpi_initlog();
  test();
}
