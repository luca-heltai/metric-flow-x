#include <deal.II/base/mpi.h>

#include <metric_flow_x/blood_flow_system.h>
#include <metric_flow_x/ida_error_guard.h>
#include <metric_flow_x/tolerance_ramp.h>

#include <string>

#include "tests.h"
using namespace dealii;
using namespace MetricFlowX;
using Integrator = SUNDIALS::IDA<VectorType>;
void
run(const bool fail_ic, const long int limit, const int expected_code)
{
  const MPI_Comm comm = MPI_COMM_WORLD;
  const auto     owned =
    Utilities::MPI::create_evenly_distributed_partitioning(comm, 2);
  VectorType y(owned, comm), dot(owned, comm);
  y   = 1.;
  dot = -1.;
  Integrator::AdditionalData data;
  data.initial_time       = 0.;
  data.final_time         = .1;
  data.output_period      = .1;
  data.initial_step_size  = 1e-5;
  data.absolute_tolerance = 1e-10;
  data.relative_tolerance = 1e-7;
  data.maximum_order      = 2;
  data.ic_type            = fail_ic ? Integrator::AdditionalData::use_y_diff :
                                      Integrator::AdditionalData::none;
  IDAErrorGuard::validate_setup(data);
  IDAErrorGuard guard(comm);
  Integrator    ida(data, comm);
  ida.reinit_vector           = [&](VectorType &v) { v.reinit(owned, comm); };
  ida.differential_components = [&] {
    void *memory = MetricFlowX::internal::native_ida_memory(ida);
    guard.install(ida, limit);
    void     *userdata = nullptr;
    const int status   = IDAGetUserData(memory, &userdata);
    AssertThrow(status >= 0 && userdata == &ida,
                ExcMessage("The error guard must preserve wrapper userdata."));
    return owned;
  };
  ida.residual = [&](double,
                     const VectorType &state,
                     const VectorType &rate,
                     VectorType       &res) {
    for (const auto i : owned)
      res(i) = fail_ic ? 1. : rate(i) + state(i);
    res.compress(VectorOperation::insert);
  };
  double alpha = 0.;
  ida.setup_jacobian =
    [&](double, const VectorType &, const VectorType &, double a) {
      alpha = a;
    };
  ida.solve_with_jacobian =
    [&](const VectorType &rhs, VectorType &dst, double) {
      dst = rhs;
      dst /= 1. + alpha;
    };
  unsigned int outputs = 0;
  ida.output_step =
    [&](double, const VectorType &, const VectorType &, unsigned int) {
      guard.check();
      ++outputs;
    };
  bool        rejected = false;
  bool        caught   = false;
  std::string failure;
  try
    {
      ida.solve_dae(y, dot);
      guard.check();
    }
  catch (const std::exception &error)
    {
      caught              = true;
      std::string message = error.what();
      // Debug SDK builds throw before output_step; retain the same native
      // diagnostic that the Release compatibility guard supplies there.
      try
        {
          guard.check();
        }
      catch (const std::exception &reported)
        {
          message = reported.what();
        }
      failure = message;
      rejected =
        message.find("SUNDIALS IDA reported error -") != std::string::npos &&
        message.find(fail_ic ? "IDACalcIC" : "IDASolve") != std::string::npos;
    }
  AssertThrow(caught == (expected_code < 0) && rejected == (expected_code < 0),
              ExcMessage("Expected IDA code " + std::to_string(expected_code) +
                         ", outputs=" + std::to_string(outputs) + ": " +
                         failure));
  AssertThrow(
    outputs == (fail_ic           ? 0u :
                expected_code < 0 ? 1u :
                                    2u),
    ExcMessage(
      "A failed IC or integration interval must not emit a solution."));
}
void
test()
{
  run(false, 10000, 0);
  run(false, 1, IDA_TOO_MUCH_WORK);
  run(true, 500, IDA_LINESEARCH_FAIL);
  deallog.depth_file(10);
  deallog
    << "Native IDA IC and integration failures rejected before invalid output: PASSED"
    << std::endl;
}
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  mpi_initlog();
  test();
}
