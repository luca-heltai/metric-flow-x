#ifndef METRIC_FLOW_X_TOLERANCE_RAMP_H
#define METRIC_FLOW_X_TOLERANCE_RAMP_H

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/sundials/ida.h>
#include <deal.II/sundials/n_vector.h>

#include <metric_flow_x/config.h>
#ifdef METRIC_FLOW_X_WITH_PETSC
#  include <deal.II/lac/petsc_vector.h>
#else
#  include <deal.II/lac/trilinos_vector.h>
#endif
#include <idas/idas.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace MetricFlowX
{
  /** Optional geometric interpolation from loose to configured IDA weights.
   * Hold duration is measured from the configured initial time; end_time is an
   * absolute physical time on IDA's output grid. Disabled policies are inert.
   */
  struct ToleranceRamp
  {
    bool   enabled                 = false;
    double initial_relative_factor = 100.0;
    double initial_absolute_factor = 100.0;
    double hold_duration           = 0.0;
    double end_time                = 0.0;

    struct Scales
    {
      double relative;
      double absolute;
    };

    void
    add_parameters(dealii::ParameterAcceptor &owner)
    {
      owner.add_parameter("Tolerance ramp enabled", enabled);
      owner.add_parameter(
        "Tolerance ramp initial relative factor",
        initial_relative_factor,
        "Initial relative tolerance multiplier, at least one.");
      owner.add_parameter(
        "Tolerance ramp initial absolute factor",
        initial_absolute_factor,
        "Initial absolute tolerance multiplier, at least one.");
      owner.add_parameter("Tolerance ramp hold duration",
                          hold_duration,
                          "Seconds held at the initial multipliers after t0.");
      owner.add_parameter(
        "Tolerance ramp end time",
        end_time,
        "Absolute time when configured tolerances are reached; "
        "must lie on the IDA output grid.");
    }

    static double
    time_tolerance(const double initial_time, const double final_time)
    {
      return 32.0 * std::numeric_limits<double>::epsilon() *
             std::max({std::abs(initial_time),
                       std::abs(final_time),
                       std::abs(final_time - initial_time)});
    }

    void
    validate(const double initial_time,
             const double final_time,
             const double output_period) const
    {
      if (!enabled)
        return;
      AssertThrow(std::isfinite(initial_time) && std::isfinite(final_time) &&
                    final_time > initial_time && std::isfinite(output_period) &&
                    output_period > 0.0,
                  dealii::ExcMessage(
                    "Invalid IDA interval for tolerance ramp."));
      AssertThrow(std::isfinite(initial_relative_factor) &&
                    initial_relative_factor >= 1.0 &&
                    std::isfinite(initial_absolute_factor) &&
                    initial_absolute_factor >= 1.0,
                  dealii::ExcMessage(
                    "Tolerance ramp factors must be finite and at least one."));
      const double hold_until = initial_time + hold_duration;
      AssertThrow(
        std::isfinite(hold_duration) && hold_duration >= 0.0 &&
          std::isfinite(hold_until) && std::isfinite(end_time) &&
          hold_until < end_time && end_time <= final_time,
        dealii::ExcMessage(
          "Tolerance ramp requires t0 + hold duration < end time <= final time."));
      const double periods = (end_time - initial_time) / output_period;
      const double nearest = std::round(periods);
      AssertThrow(
        std::isfinite(periods) && nearest >= 1.0 &&
          std::abs(end_time - (initial_time + nearest * output_period)) <=
            time_tolerance(initial_time, end_time),
        dealii::ExcMessage(
          "Tolerance ramp end time must lie on the IDA output grid."));
    }

    Scales
    scales(const double initial_time, const double time) const
    {
      if (!enabled || time >= end_time - time_tolerance(initial_time, end_time))
        return {1.0, 1.0};
      const double hold_until = initial_time + hold_duration;
      if (time <= hold_until)
        return {initial_relative_factor, initial_absolute_factor};
      const double fraction =
        std::clamp((end_time - time) / (end_time - hold_until), 0.0, 1.0);
      const auto interpolate = [fraction](const double initial) {
        return std::min(initial, std::exp(fraction * std::log(initial)));
      };
      return {interpolate(initial_relative_factor),
              interpolate(initial_absolute_factor)};
    }
  };

  namespace internal
  {
    // Defined beside the existing IDA statistics compatibility shim. This is
    // not a public BloodFlowSystem interface or another private-member shim.
#ifdef METRIC_FLOW_X_WITH_PETSC
    void *
    native_ida_memory(
      const dealii::SUNDIALS::IDA<dealii::PETScWrappers::MPI::Vector> &ida);
#else
    void *
    native_ida_memory(
      const dealii::SUNDIALS::IDA<dealii::TrilinosWrappers::MPI::Vector> &ida);
#endif
  } // namespace internal

  /** Carry ramp state in a forwarding public vector-initializer callback.
   * The enabled ramp uses scalar setup, then installs its componentwise weight
   * function before CalcIC. This avoids the installed wrapper's empty-rank
   * minimum bug in temporary IDASVtolerances setup, without global storage.
   * Register the weight function in differential_components(), before CalcIC.
   * Call after_solve() at every output callback, including the initial one.
   * Native SS/SV setters must not be called after registration: SUNDIALS6.7
   * invalidates their error-weight userdata when used during integration.
   */
  template <typename VectorType>
  class ToleranceRampSource
  {
  public:
    using IDA = dealii::SUNDIALS::IDA<VectorType>;

    struct Diagnostics
    {
      unsigned int          registrations      = 0;
      unsigned long         weight_evaluations = 0;
      unsigned int          stop_arms          = 0;
      unsigned int          stop_releases      = 0;
      double                last_time          = 0.0;
      ToleranceRamp::Scales last_scales        = {1.0, 1.0};
    };

    void
    operator()(VectorType &vector)
    {
      reinitialize(vector);
    }

    static void
    attach(IDA                                &ida,
           const ToleranceRamp                &policy,
           const typename IDA::AdditionalData &data,
           const VectorType                   &base_absolute)
    {
      if (!policy.enabled)
        return;
      policy.validate(data.initial_time, data.final_time, data.output_period);
      using Data = typename IDA::AdditionalData;
      AssertThrow(
        data.ic_type == Data::use_y_diff ||
          data.reset_type == Data::use_y_diff ||
          data.ignore_algebraic_terms_for_errors,
        dealii::ExcMessage(
          "Enabled tolerance ramp requires IDA's differential-component setup hook."));
      bool valid =
        std::isfinite(data.relative_tolerance) &&
        data.relative_tolerance >= 0.0 &&
        std::isfinite(data.absolute_tolerance) &&
        data.absolute_tolerance > 0.0 &&
        std::isfinite(data.relative_tolerance * policy.initial_relative_factor);
      double largest_absolute = 0.0;
      for (const auto index : base_absolute.locally_owned_elements())
        {
          const double value = base_absolute(index);
          valid              = valid && std::isfinite(value) && value > 0.0;
          largest_absolute   = std::max(largest_absolute, std::abs(value));
        }
      const auto comm  = base_absolute.get_mpi_communicator();
      valid            = dealii::Utilities::MPI::min(valid ? 1 : 0, comm) == 1;
      largest_absolute = dealii::Utilities::MPI::max(largest_absolute, comm);
      AssertThrow(
        valid && base_absolute.size() > 0 &&
          std::isfinite(largest_absolute * policy.initial_absolute_factor),
        dealii::ExcMessage(
          "Tolerance ramp requires positive finite absolute tolerances and finite scaled tolerances."));
      auto original = ida.reinit_vector;
      if (const auto *previous =
            original.template target<ToleranceRampSource>())
        original = previous->reinitialize;
      ida.reinit_vector = ToleranceRampSource(
        std::make_shared<State>(policy, data, base_absolute),
        std::move(original));
      // The componentwise vector is retained in State and used by WF weights;
      // only the wrapper's temporary SS/SV setup choice changes here.
      ida.get_local_tolerances = {};
    }

    static void
    register_weights(void *memory, IDA &ida)
    {
      auto *source = ida.reinit_vector.template target<ToleranceRampSource>();
      if (!source)
        return;
      AssertThrow(memory != nullptr,
                  dealii::ExcMessage("IDA memory is not initialized."));
      void *user_data = nullptr;
      check(IDAGetUserData(memory, &user_data), "IDAGetUserData");
      AssertThrow(user_data == &ida,
                  dealii::ExcMessage(
                    "Tolerance ramp must preserve IDA wrapper userdata."));
      auto &context    = *source->state;
      context.memory   = memory;
      context.armed    = false;
      context.released = false;
      context.error.clear();
      check(IDAWFtolerances(memory, &weight_function), "IDAWFtolerances");
      ++context.diagnostics.registrations;
    }

    static void
    after_solve(IDA &ida)
    {
      auto *source = ida.reinit_vector.template target<ToleranceRampSource>();
      if (!source)
        return;
      auto &context = *source->state;
      AssertThrow(context.error.empty(), dealii::ExcMessage(context.error));
      AssertThrow(
        context.memory != nullptr,
        dealii::ExcMessage(
          "Tolerance ramp weight function was not registered before IDA initialization."));
      double time = 0.0;
      check(IDAGetCurrentTime(context.memory, &time), "IDAGetCurrentTime");
      const bool at_cutoff =
        time >= context.policy.end_time -
                  ToleranceRamp::time_tolerance(context.initial_time,
                                                context.policy.end_time);
      if (!context.armed && !at_cutoff)
        {
          // reset() writes its own final stop time after the differential hook.
          // Arm this cutoff only after reset/CalcIC, at the initial callback.
          check(IDASetStopTime(context.memory, context.policy.end_time),
                "IDASetStopTime");
          context.armed = true;
          ++context.diagnostics.stop_arms;
        }
      if (at_cutoff && !context.released)
        {
          check(IDASetStopTime(context.memory, context.final_time),
                "IDASetStopTime");
          context.released = true;
          ++context.diagnostics.stop_releases;
        }
    }

    static const Diagnostics &
    diagnostics(const IDA &ida)
    {
      const auto *source =
        ida.reinit_vector.template target<ToleranceRampSource>();
      AssertThrow(source != nullptr,
                  dealii::ExcMessage("Tolerance ramp is not attached."));
      return source->state->diagnostics;
    }

  private:
    struct State
    {
      State(const ToleranceRamp                &policy,
            const typename IDA::AdditionalData &data,
            const VectorType                   &absolute)
        : policy(policy)
        , initial_time(data.initial_time)
        , final_time(data.final_time)
        , relative(data.relative_tolerance)
        , absolute(absolute)
      {}

      ToleranceRamp policy;
      double        initial_time;
      double        final_time;
      double        relative;
      VectorType    absolute;
      void         *memory   = nullptr;
      bool          armed    = false;
      bool          released = false;
      std::string   error;
      Diagnostics   diagnostics;
    };

    ToleranceRampSource(std::shared_ptr<State>            state,
                        std::function<void(VectorType &)> reinitialize)
      : state(std::move(state))
      , reinitialize(std::move(reinitialize))
    {}

    static void
    check(const int status, const char *operation)
    {
      AssertThrow(status == IDA_SUCCESS,
                  dealii::ExcMessage(std::string(operation) +
                                     " failed with status " +
                                     std::to_string(status)));
    }

    static int
    weight_function(N_Vector y, N_Vector weights, void *user_data) noexcept
    {
      if (!user_data)
        return -1;
      auto &ida    = *static_cast<IDA *>(user_data);
      auto *source = ida.reinit_vector.template target<ToleranceRampSource>();
      if (!source)
        return -1;
      auto &context = *source->state;
      try
        {
          double time = 0.0;
          check(IDAGetCurrentTime(context.memory, &time), "IDAGetCurrentTime");
          AssertThrow(std::isfinite(time),
                      dealii::ExcMessage("IDA returned a non-finite time."));
          const auto scales = context.policy.scales(context.initial_time, time);
#if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
          auto absolute =
            dealii::SUNDIALS::internal::make_nvector_view(context.absolute,
                                                          y->sunctx);
#else
          auto absolute =
            dealii::SUNDIALS::internal::make_nvector_view(context.absolute);
#endif
          N_VAbs(y, weights);
          N_VScale(context.relative * scales.relative, weights, weights);
          N_VLinearSum(1.0, weights, scales.absolute, absolute, weights);
          N_VInv(weights, weights);
          const auto *state_values =
            dealii::SUNDIALS::internal::unwrap_nvector_const<VectorType>(y);
          const auto *weight_values =
            dealii::SUNDIALS::internal::unwrap_nvector_const<VectorType>(
              weights);
          bool valid = true;
          for (const auto index : weight_values->locally_owned_elements())
            valid = valid && std::isfinite((*state_values)(index)) &&
                    std::isfinite((*weight_values)(index)) &&
                    (*weight_values)(index) > 0.0;
          if (dealii::Utilities::MPI::min(
                valid ? 1 : 0, context.absolute.get_mpi_communicator()) == 0)
            {
              context.error =
                "Non-finite state or non-positive/non-finite IDA error weights.";
              return -1;
            }
          ++context.diagnostics.weight_evaluations;
          context.diagnostics.last_time   = time;
          context.diagnostics.last_scales = scales;
          return 0;
        }
      catch (const std::exception &exception)
        {
          context.error = exception.what();
        }
      catch (...)
        {
          context.error = "Tolerance ramp weight evaluation failed.";
        }
      return -1;
    }

    std::shared_ptr<State>            state;
    std::function<void(VectorType &)> reinitialize;
  };
} // namespace MetricFlowX
#endif
