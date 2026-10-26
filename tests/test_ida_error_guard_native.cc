// Exercise native error capture independently of deal.II's IDA status checks.
#include <deal.II/base/mpi.h>

#include <metric_flow_x/ida_error_guard.h>
#if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
#  include <sundials/sundials_context.h>
#endif

#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  mpi_initlog();
  IDAErrorGuard guard(MPI_COMM_WORLD);
#if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
  SUNContext context = nullptr;
#  if DEAL_II_SUNDIALS_VERSION_GTE(7, 0, 0)
  AssertThrow(SUNContext_Create(SUN_COMM_NULL, &context) == 0,
              ExcInternalError());
#  else
  AssertThrow(SUNContext_Create(nullptr, &context) == 0, ExcInternalError());
#  endif
  void *memory = IDACreate(context);
#else
  void *memory = IDACreate();
#endif
  AssertThrow(memory != nullptr, ExcInternalError());
  for (unsigned int i = 0; i < 2; ++i)
    guard.install(memory,
                  500
#if DEAL_II_SUNDIALS_VERSION_GTE(7, 0, 0)
                  ,
                  context
#endif
    );
  // A failure on one rank must be reported on every rank.
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    AssertThrow(IDASetMaxStep(memory, -1.) == IDA_ILL_INPUT,
                ExcInternalError());
  bool caught = false;
  try
    {
      guard.check();
    }
  catch (const std::exception &error)
    {
      const std::string message = error.what();
      caught =
        message.find("SUNDIALS IDA reported error -") != std::string::npos &&
        message.find("IDASetMaxStep") != std::string::npos;
    }
  IDAFree(&memory);
#if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
  AssertThrow(SUNContext_Free(&context) == 0, ExcInternalError());
#endif
  AssertThrow(caught,
              ExcMessage(
                "The native error handler did not capture the failure."));
  deallog << "Native IDA error handler captures invalid setup: PASSED"
          << std::endl;
}
