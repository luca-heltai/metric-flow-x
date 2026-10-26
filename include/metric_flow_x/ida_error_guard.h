#ifndef metric_flow_x_ida_error_guard_h
#define metric_flow_x_ida_error_guard_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <idas/idas.h>
#include <metric_flow_x/ida_access.h>
#if DEAL_II_SUNDIALS_VERSION_GTE(7, 0, 0)
#  include <sundials/sundials_context.h>
#endif

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

    template <typename VectorType>
    void
    install(const dealii::SUNDIALS::IDA<VectorType> &ida,
            const long int                           maximum_steps)
    {
      install(internal::native_ida_memory(ida),
              maximum_steps
#if DEAL_II_SUNDIALS_VERSION_GTE(7, 0, 0)
              ,
              internal::native_ida_context(ida)
#endif
      );
    }

    void
    install(void          *memory,
            const long int maximum_steps
#if DEAL_II_SUNDIALS_VERSION_GTE(7, 0, 0)
            ,
            const SUNContext context
#endif
    )
    {
      AssertThrow(memory != nullptr && maximum_steps > 0,
                  dealii::ExcMessage("Invalid IDA error guard setup."));
#if DEAL_II_SUNDIALS_VERSION_GTE(7, 0, 0)
      AssertThrow(context != nullptr,
                  dealii::ExcMessage("Invalid IDA error guard context."));
      // The wrapper retains its context across resets. Install only once per
      // context, and keep this guard alive until the wrapper is destroyed.
      if (context != handler_context)
        {
          const int status =
            SUNContext_PushErrHandler(context, &capture_sun, this);
          AssertThrow(status == 0,
                      dealii::ExcMessage(
                        "SUNContext_PushErrHandler failed with status " +
                        std::to_string(status)));
          handler_context = context;
        }
#else
      const int handler_status = IDASetErrHandlerFn(memory, &capture, this);
      AssertThrow(handler_status >= 0,
                  dealii::ExcMessage("IDASetErrHandlerFn failed with status " +
                                     std::to_string(handler_status)));
#endif
      const int limit_status = IDASetMaxNumSteps(memory, maximum_steps);
      if (limit_status < 0 && code == 0)
        capture(limit_status, "IDAS", "IDASetMaxNumSteps", nullptr, this);
      check();
    }

    template <typename VectorType>
    void
    install(const dealii::SUNDIALS::IDA<VectorType> &ida,
            const long int                           maximum_steps,
            const double                             minimum_step,
            const double                             maximum_step)
    {
      AssertThrow(std::isfinite(minimum_step) && minimum_step >= 0. &&
                    std::isfinite(maximum_step) && maximum_step >= 0. &&
                    (maximum_step == 0. || minimum_step <= maximum_step),
                  dealii::ExcMessage("Invalid native IDA step bounds."));
      install(ida, maximum_steps);
      void *memory = internal::native_ida_memory(ida);
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
#if DEAL_II_SUNDIALS_VERSION_GTE(7, 0, 0)
    static void
    capture_sun(int,
                const char *function,
                const char *,
                const char *message,
                SUNErrCode  error_code,
                void       *context,
                SUNContext) noexcept
    {
      record(error_code, "IDAS", function, message, context);
    }

    SUNContext handler_context = nullptr;
#endif
    static void
    capture(const int   error_code,
            const char *module,
            const char *function,
            char       *message,
            void       *context) noexcept
    {
      record(error_code, module, function, message, context);
    }

    static void
    record(const int   error_code,
           const char *module,
           const char *function,
           const char *message,
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
