#ifndef metric_flow_x_ida_error_guard_h
#define metric_flow_x_ida_error_guard_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <idas/idas.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <string>

namespace MetricFlowX
{
  // deal.II's Release IDA wrapper does not check negative return codes.
  // Capture them at the native C boundary and throw only back in C++.
  class IDAErrorGuard
  {
  public:
    explicit IDAErrorGuard(const MPI_Comm communicator)
      : comm(communicator)
    {}

    template <class Data>
    static void
    validate_setup(const Data &data)
    {
      AssertThrow(data.ic_type == Data::use_y_diff ||
                    data.reset_type == Data::use_y_diff ||
                    data.ignore_algebraic_terms_for_errors,
                  dealii::ExcMessage(
                    "This deal.II IDA compatibility guard requires the "
                    "differential-component setup hook before CalcIC."));
    }

    void
    install(void *memory, const long int maximum_steps)
    {
      AssertThrow(memory != nullptr && maximum_steps > 0,
                  dealii::ExcMessage("Invalid IDA error guard setup."));
      const int handler_status = IDASetErrHandlerFn(memory, &capture, this);
      AssertThrow(handler_status >= 0,
                  dealii::ExcMessage("IDASetErrHandlerFn failed with status " +
                                     std::to_string(handler_status)));
      const int limit_status = IDASetMaxNumSteps(memory, maximum_steps);
      if (limit_status < 0 && code == 0)
        capture(limit_status, "IDAS", "IDASetMaxNumSteps", nullptr, this);
      check();
    }

    void
    install(void          *memory,
            const long int maximum_steps,
            const double   minimum_step,
            const double   maximum_step)
    {
      AssertThrow(std::isfinite(minimum_step) && minimum_step >= 0. &&
                    std::isfinite(maximum_step) && maximum_step >= 0. &&
                    (maximum_step == 0. || minimum_step <= maximum_step),
                  dealii::ExcMessage("Invalid native IDA step bounds."));
      install(memory, maximum_steps);
      // deal.II's AdditionalData minimum is otherwise not passed to IDA.
      const int minimum_status = IDASetMinStep(memory, minimum_step);
      AssertThrow(minimum_status >= 0,
                  dealii::ExcMessage("IDASetMinStep failed with status " +
                                     std::to_string(minimum_status)));
      const int maximum_status = IDASetMaxStep(memory, maximum_step);
      AssertThrow(maximum_status >= 0,
                  dealii::ExcMessage("IDASetMaxStep failed with status " +
                                     std::to_string(maximum_status)));
      check();
    }

    void
    check() const
    {
      if (dealii::Utilities::MPI::min(code, comm) >= 0)
        return;
      const std::string local = code < 0 ? "SUNDIALS IDA reported error " +
                                             std::to_string(code) + " in " +
                                             details.data() :
                                           "";
      for (const auto &message :
           dealii::Utilities::MPI::all_gather(comm, local))
        AssertThrow(message.empty(), dealii::ExcMessage(message));
    }

  private:
    static void
    capture(const int   error_code,
            const char *module,
            const char *function,
            char       *message,
            void       *context) noexcept
    {
      auto &guard = *static_cast<IDAErrorGuard *>(context);
      if (error_code >= 0 || guard.code < 0)
        return;
      guard.code = error_code;
      std::snprintf(guard.details.data(),
                    guard.details.size(),
                    "%s::%s: %s",
                    module ? module : "IDAS",
                    function ? function : "unknown",
                    message ? message : "native setup failed");
      // Debug deal.II may abort on the native status before C++ regains
      // control. Preserve the native diagnostic as well as the saved Release
      // guard state.
      std::fprintf(stderr,
                   "SUNDIALS IDA reported error %d in %s\n",
                   guard.code,
                   guard.details.data());
    }

    MPI_Comm               comm;
    int                    code = 0;
    std::array<char, 2048> details{};
  };
} // namespace MetricFlowX
#endif
