// Observe every accepted step, and preserve the public output-only helpers.
#include <deal.II/base/mpi.h>

#include <metric_flow_x/blood_flow_system.h>
#include <metric_flow_x/ida_error_guard.h>
#include <metric_flow_x/tolerance_ramp.h>

#include <numeric>

#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

void
test()
{
  const MPI_Comm comm = MPI_COMM_WORLD;
  for (const bool observe_each_step : {false, true})
    {
      ParameterAcceptor::clear();
      BloodFlowSystem<1, 3> flow(comm);
      flow.output_directory = observe_each_step ? "observed" : "output_only";
      flow.print_statistics = false;
      flow.open_output_files();
      const auto owned =
        Utilities::MPI::create_evenly_distributed_partitioning(comm, 2);
      VectorType y(owned, comm), dot(owned, comm);
      y                = 1.;
      dot              = 0.;
      using Integrator = SUNDIALS::IDA<VectorType>;
      Integrator::AdditionalData data;
      data.initial_time      = .25;
      data.final_time        = .375;
      data.output_period     = .0625;
      data.initial_step_size = .03125;
      data.ic_type           = Integrator::AdditionalData::none;
      const double  maximum  = 1. / 256;
      IDAErrorGuard guard(comm);
      Integrator    ida(data, comm);
      ida.reinit_vector = [&](VectorType &v) { v.reinit(owned, comm); };
      ida.differential_components = [&] {
        guard.install(ida, 10000, 0., maximum);
        return owned;
      };
      ida.residual = [&](double,
                         const VectorType &,
                         const VectorType &rate,
                         VectorType       &residual) {
        if (observe_each_step)
          flow.poll_ida_internal_step(ida, false, true);
        residual = rate;
      };
      double alpha = 0.;
      ida.setup_jacobian =
        [&](double, const VectorType &, const VectorType &, double a) {
          ++flow.ida_jacobian_evaluations;
          alpha = a;
        };
      ida.solve_with_jacobian =
        [&](const VectorType &rhs, VectorType &dst, double) {
          ++flow.ida_linear_solves;
          dst = rhs;
          dst /= alpha;
        };
      ida.output_step = [&](double            t,
                            const VectorType &state,
                            const VectorType &rate,
                            unsigned int      output) {
        guard.check();
        if (observe_each_step)
          flow.poll_ida_internal_step(ida, true, true);
        flow.record_ida_step(ida, output == 0 ? data.initial_time : t);
        const auto record = flow.ida_output_history().back();
        if (output == 0)
          AssertThrow(!record.step_extrema_available &&
                        record.last_step_size == 0.,
                      ExcMessage("Initial output has no accepted timestep."));
        else
          {
            AssertThrow(
              record.step_extrema_available == observe_each_step,
              ExcMessage(
                "Output-only clients must not receive invented extrema."));
            AssertThrow(
              state.l2_norm() == std::sqrt(2.) && rate.linfty_norm() == 0.,
              ExcMessage("Constant analytical solution was changed."));
            if (observe_each_step)
              AssertThrow(
                std::abs(record.minimum_step_size / maximum - 1.) < 1e-12 &&
                  std::abs(record.maximum_step_size / maximum - 1.) < 1e-12,
                ExcMessage("Native hmax was not used or extrema are wrong."));
          }
      };
      ida.solve_dae(y, dot);
      guard.check();
      // Existing callers only record at outputs and then finalize. This must
      // remain valid even though many internal steps were not observed.
      flow.finalize_output(ida, 0.);
      const auto final = flow.current_ida_statistics(ida, data.final_time);
      AssertThrow(
        final.steps_cumulative == 32 &&
          final.step_extrema_available == observe_each_step &&
          final.jacobian_evals_cumulative > 0 &&
          final.jacobian_evals_cumulative == final.lin_setups_cumulative &&
          final.linear_solves_cumulative > 0 &&
          std::accumulate(flow.bdf_steps.begin(), flow.bdf_steps.end(), 0L) ==
            32,
        ExcMessage(
          "Native limits, counters or compatible BDF accounting failed."));
      if (observe_each_step)
        {
          std::ifstream input(flow.output_directory +
                              "/ida_internal_steps.csv");
          AssertThrow(input.good(), ExcMessage("Missing accepted-step CSV."));
          std::string line;
          std::getline(input, line);
          unsigned int rows = 0;
          while (std::getline(input, line))
            {
              const auto first  = line.find(','),
                         second = line.find(',', first + 1);
              const double accepted_time =
                std::stod(line.substr(first + 1, second - first - 1));
              AssertThrow(
                accepted_time > data.initial_time &&
                  accepted_time <= data.final_time,
                ExcMessage(
                  "Accepted times were not resynchronized to the external integrator's t0."));
              ++rows;
            }
          AssertThrow(rows == 32, ExcInternalError());
        }
    }
  deallog.depth_file(10);
  deallog
    << "Exact accepted-step statistics, native hmax and output-only compatibility: PASSED"
    << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  mpi_initlog();
  test();
}
