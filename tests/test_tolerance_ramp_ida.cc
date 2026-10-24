// Exercise the real deal.II/IDAS bridge, including CalcIC, retained BDF
// history, componentwise weights and a strict cutoff at nonzero physical
// initial time.
#include <deal.II/base/mpi.h>

#include <metric_flow_x/blood_flow_system.h>
#include <metric_flow_x/tolerance_ramp.h>

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;
using Integrator = SUNDIALS::IDA<VectorType>;
using Source     = ToleranceRampSource<VectorType>;

struct Run
{
  std::vector<double> times;
  std::vector<double> values;
  long int            steps = 0;
};

std::array<double, 2>
values(const VectorType &v, const IndexSet &owned, MPI_Comm comm)
{
  std::array<double, 2> local{}, global{};
  for (const auto i : owned)
    local[i] = v(i);
  MPI_Allreduce(local.data(), global.data(), 2, MPI_DOUBLE, MPI_SUM, comm);
  return global;
}

Run
integrate(const bool   enabled,
          const double factor,
          const bool   invalid_state = false,
          const bool   componentwise = false)
{
  const MPI_Comm comm = MPI_COMM_WORLD;
  const auto     rank = Utilities::MPI::this_mpi_process(comm);
  const auto     size = Utilities::MPI::n_mpi_processes(comm);
  IndexSet       owned(2);
  for (unsigned int i = 0; i < 2; ++i)
    if (i % size == rank)
      owned.add_index(i);
  owned.compress();
  VectorType y(owned, comm), yp(owned, comm), absolute(owned, comm);
  y = 1.0;
  // Installed CalcIC uses a broken native minimum on empty local vectors.
  // Empty-rank coverage therefore uses the independently consistent pair,
  // matching the application native-IC policy; serial/MPI2 still run CalcIC.
  const bool use_calc_ic = size <= 2;
  yp                     = use_calc_ic ? 0.0 : -1.0;
  if (invalid_state && owned.is_element(0))
    y(0) = std::numeric_limits<double>::infinity();
  y.compress(VectorOperation::insert);
  for (const auto i : owned)
    absolute(i) = componentwise && i == 1 ? 2e-10 : 1e-10;
  absolute.compress(VectorOperation::insert);

  Integrator::AdditionalData data;
  data.initial_time = .25;
  data.final_time   = .35;
  // CalcIC scales derivative corrections by an artificial step. This value
  // resolves the deliberately wrong initial derivative even at factor100.
  data.initial_step_size  = 1e-3;
  data.output_period      = .01;
  data.absolute_tolerance = 1e-10;
  data.relative_tolerance = 1e-7;
  data.maximum_order      = 2;
  data.ic_type = use_calc_ic ? Integrator::AdditionalData::use_y_diff :
                               Integrator::AdditionalData::none;
  Integrator ida(data, comm);
  ida.reinit_vector = [&](VectorType &v) { v.reinit(owned, comm); };
  unsigned int local_tolerance_queries = 0;
  if (componentwise)
    ida.get_local_tolerances = [&]() -> VectorType & {
      ++local_tolerance_queries;
      return absolute;
    };
  ToleranceRamp ramp;
  ramp.enabled                 = enabled;
  ramp.initial_relative_factor = factor;
  ramp.initial_absolute_factor = factor;
  ramp.hold_duration           = .01;
  ramp.end_time                = .30;
  Source::attach(ida, ramp, data, absolute);
  unsigned int differential_queries = 0;
  ida.differential_components       = [&] {
    ++differential_queries;
    Source::register_weights(MetricFlowX::internal::native_ida_memory(ida),
                             ida);
    IndexSet differential(2);
    if (owned.is_element(0))
      differential.add_index(0);
    differential.compress();
    return differential;
  };
  ida.residual = [&](double,
                     const VectorType &state,
                     const VectorType &dot,
                     VectorType       &res) {
    const auto v = values(state, owned, comm), d = values(dot, owned, comm);
    for (const auto i : owned)
      res(i) = i == 0 ? d[0] + v[0] : v[1] - v[0];
    res.compress(VectorOperation::insert);
  };
  double alpha = 0;
  ida.setup_jacobian =
    [&](double, const VectorType &, const VectorType &, double a) {
      alpha = a;
    };
  ida.solve_with_jacobian =
    [&](const VectorType &rhs, VectorType &dst, double) {
      const auto   r  = values(rhs, owned, comm);
      const double z0 = r[0] / (1 + alpha);
      for (const auto i : owned)
        dst(i) = i == 0 ? z0 : r[1] + z0;
      dst.compress(VectorOperation::insert);
    };
  Run      result;
  long int previous_steps = 0, initial_backtracks = -1;
  bool     baseline_stop_released = false;
  ida.output_step                 = [&](double            t,
                        const VectorType &state,
                        const VectorType &dot,
                        unsigned int      output) {
    auto *memory = MetricFlowX::internal::native_ida_memory(ida);
    AssertThrow(memory != nullptr, ExcInternalError());
    Source::after_solve(ida);
    double internal_time = 0;
    AssertThrow(IDAGetCurrentTime(memory, &internal_time) == 0,
                ExcInternalError());
    // Identical stop boundaries isolate custom weights from changes in the
    // integrator's step endpoints when comparing factor-one with built-in SV.
    if (!enabled)
      {
        if (output == 0)
          AssertThrow(IDASetStopTime(memory, ramp.end_time) == 0,
                      ExcInternalError());
        if (!baseline_stop_released &&
            internal_time >=
              ramp.end_time -
                ToleranceRamp::time_tolerance(data.initial_time, ramp.end_time))
          {
            AssertThrow(IDASetStopTime(memory, data.final_time) == 0,
                        ExcInternalError());
            baseline_stop_released = true;
          }
      }
    if (output == 0)
      t = data.initial_time; // Installed wrapper passes literal zero.
    // Without CalcIC, weights are first evaluated on the first integration
    // step.
    if (invalid_state && output == 0)
      return;
    const auto v = values(state, owned, comm), d = values(dot, owned, comm);
    AssertThrow(std::abs(v[1] - v[0]) < 1e-7,
                ExcMessage("Algebraic constraint lost."));
    AssertThrow(
      std::abs(v[0] - std::exp(-(t - data.initial_time))) < 5e-6,
      ExcMessage(
        "Native DAE trajectory differs from its analytical solution."));
    if (output == 0)
      {
        AssertThrow(std::abs(d[0] + 1.0) < 1e-5,
                    ExcMessage(
                      "CalcIC did not construct the differential derivative."));
        AssertThrow(IDAGetNumBacktrackOps(memory, &initial_backtracks) == 0,
                    ExcInternalError());
      }
    long int steps = 0, backtracks = 0;
    AssertThrow(IDAGetNumSteps(memory, &steps) == 0 && steps >= previous_steps,
                ExcMessage("IDA history was reset."));
    AssertThrow(
      IDAGetNumBacktrackOps(memory, &backtracks) == 0 &&
        backtracks == initial_backtracks,
      ExcMessage(
        "Initial-condition backtracking counter changed during integration."));
    previous_steps = steps;
    result.times.push_back(t);
    result.values.push_back(v[0]);
    result.steps = steps;
  };
  try
    {
      ida.solve_dae(y, yp);
    }
  catch (const ExceptionBase &)
    {
      // Debug deal.II checks the native failure before the output callback.
      // Preserve the weight callback's more specific error in that case.
      Source::after_solve(ida);
      throw;
    }
  AssertThrow(differential_queries == 1,
              ExcMessage("IDA initialization/CalcIC setup repeated."));
  AssertThrow(result.times.size() == 11,
              ExcMessage("Cutoff changed the requested output grid."));
  for (unsigned int i = 0; i < result.times.size(); ++i)
    AssertThrow(std::abs(result.times[i] -
                         (data.initial_time + i * data.output_period)) < 2e-14,
                ExcMessage("Cutoff skipped a requested output time."));
  AssertThrow(result.steps > 2,
              ExcMessage("Native integration did not advance."));
  if (enabled)
    {
      const auto &d = Source::diagnostics(ida);
      AssertThrow(d.registrations == 1 && d.weight_evaluations > 0 &&
                    d.stop_arms == 1 && d.stop_releases == 1,
                  ExcMessage("Unexpected ramp registration/stop lifecycle."));
      AssertThrow(d.last_time >= ramp.end_time &&
                    d.last_scales.relative == 1.0 &&
                    d.last_scales.absolute == 1.0,
                  ExcMessage("Strict final tolerances were not reached."));
    }
  else
    AssertThrow(local_tolerance_queries == (componentwise ? 1u : 0u),
                ExcMessage(
                  "Built-in tolerance source was queried during integration."));
  return result;
}

void
validate_callback_configuration()
{
  IndexSet owned(1);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    owned.add_index(0);
  owned.compress();
  VectorType absolute(owned, MPI_COMM_WORLD);
  absolute = 1e-10;
  Integrator::AdditionalData data;
  data.initial_time                      = 0.0;
  data.final_time                        = 1.0;
  data.output_period                     = 0.1;
  data.ic_type                           = Integrator::AdditionalData::none;
  data.reset_type                        = Integrator::AdditionalData::none;
  data.ignore_algebraic_terms_for_errors = false;
  Integrator ida(data, MPI_COMM_WORLD);
  auto       original      = [&]() -> VectorType            &{ return absolute; };
  ida.get_local_tolerances = original;
  auto original_reinit     = [&](VectorType &v) {
    v.reinit(owned, MPI_COMM_WORLD);
  };
  ida.reinit_vector = original_reinit;
  ToleranceRamp policy;
  Source::attach(ida, policy, data, absolute);
  AssertThrow(ida.get_local_tolerances.target<decltype(original)>() != nullptr,
              ExcMessage(
                "Disabled ramp changed the existing tolerance callback."));
  AssertThrow(ida.reinit_vector.target<decltype(original_reinit)>() != nullptr,
              ExcMessage("Disabled ramp changed the vector callback."));
  policy.enabled  = true;
  policy.end_time = 0.5;
  bool rejected   = false;
  try
    {
      Source::attach(ida, policy, data, absolute);
    }
  catch (const ExceptionBase &)
    {
      rejected = true;
    }
  AssertThrow(
    rejected,
    ExcMessage(
      "Ramp accepted a configuration without the initialization hook."));
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  validate_callback_configuration();
  const auto baseline  = integrate(false, 1.0);
  const auto identical = integrate(true, 1.0);
  AssertThrow(baseline.times == identical.times, ExcInternalError());
  for (unsigned int i = 0; i < baseline.values.size(); ++i)
    AssertThrow(
      std::abs(baseline.values[i] - identical.values[i]) < 1e-12,
      ExcMessage(
        "Factor-one weights differ from native componentwise tolerances."));
  integrate(true, 100.0, false, true);
  // Installed deal.II's native SV setup dereferences an empty local range.
  // Compare its componentwise formula where every rank owns a component;
  // the WF implementation above is also exercised with empty ranks.
  if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) <= 2)
    {
      const auto builtin = integrate(false, 1.0, false, true);
      const auto custom  = integrate(true, 1.0, false, true);
      AssertThrow(builtin.times == custom.times, ExcInternalError());
      for (unsigned int i = 0; i < builtin.values.size(); ++i)
        AssertThrow(std::abs(builtin.values[i] - custom.values[i]) < 1e-12,
                    ExcMessage("Componentwise factor-one weights differ."));
    }
  bool invalid_weights_rejected = false;
  try
    {
      integrate(true, 100.0, true, true);
    }
  catch (const ExceptionBase &exception)
    {
      invalid_weights_rejected =
        std::string(exception.what()).find("IDA error weights") !=
        std::string::npos;
    }
  AssertThrow(invalid_weights_rejected,
              ExcMessage("Invalid native error weights were not rejected."));
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      initlog();
      deallog
        << "Native IDA tolerance ramp, retained history and cutoff: PASSED"
        << std::endl;
    }
}
