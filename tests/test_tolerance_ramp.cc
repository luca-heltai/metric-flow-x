#include <metric_flow_x/tolerance_ramp.h>

#include <cmath>
#include <functional>
#include <limits>

#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

void
rejected(const std::function<void()> &operation)
{
  bool failed = false;
  try
    {
      operation();
    }
  catch (const ExceptionBase &)
    {
      failed = true;
    }
  AssertThrow(failed, ExcMessage("Invalid ramp policy was accepted."));
}

int
main()
{
  initlog();
  ToleranceRamp ramp;
  ramp.validate(2.0,
                1.0,
                -1.0); // Disabled policy leaves existing IDA behavior alone.
  AssertThrow(ramp.scales(2.0, 2.0).absolute == 1.0, ExcInternalError());
  ramp.enabled                 = true;
  ramp.initial_relative_factor = 100.0;
  ramp.initial_absolute_factor = 10000.0;
  ramp.hold_duration           = 0.25;
  ramp.end_time                = 3.0;
  ramp.validate(2.0, 4.0, 0.25);
  for (const double t : {1.0, 2.0, 2.25})
    {
      const auto s = ramp.scales(2.0, t);
      AssertThrow(s.relative == 100.0 && s.absolute == 10000.0,
                  ExcInternalError());
    }
  const auto midpoint = ramp.scales(2.0, 2.625);
  AssertThrow(std::abs(midpoint.relative - 10.0) < 1e-12 &&
                std::abs(midpoint.absolute - 100.0) < 1e-11,
              ExcInternalError());
  for (const double t : {3.0, 3.5, 4.0})
    {
      const auto s = ramp.scales(2.0, t);
      AssertThrow(s.relative == 1.0 && s.absolute == 1.0, ExcInternalError());
    }
  auto previous = ramp.scales(2.0, 2.0);
  for (unsigned int i = 1; i <= 100; ++i)
    {
      const auto next = ramp.scales(2.0, 2.0 + i / 100.0);
      AssertThrow(next.relative <= previous.relative &&
                    next.absolute <= previous.absolute &&
                    next.relative >= 1.0 && next.absolute >= 1.0,
                  ExcInternalError());
      previous = next;
    }
  rejected([&] { ramp.validate(2.0, 2.0, 0.25); });
  rejected([&] { ramp.validate(2.0, 4.0, 0.0); });
  ramp.end_time = 3.1;
  rejected([&] { ramp.validate(2.0, 4.0, 0.25); });
  ramp.end_time = 4.25;
  rejected([&] { ramp.validate(2.0, 4.0, 0.25); });
  ramp.end_time = 2.25;
  rejected([&] { ramp.validate(2.0, 4.0, 0.25); });
  ramp.end_time      = 3.0;
  ramp.hold_duration = -1.0;
  rejected([&] { ramp.validate(2.0, 4.0, 0.25); });
  ramp.hold_duration           = 0.25;
  ramp.initial_relative_factor = 0.5;
  rejected([&] { ramp.validate(2.0, 4.0, 0.25); });
  ramp.initial_relative_factor = std::numeric_limits<double>::infinity();
  rejected([&] { ramp.validate(2.0, 4.0, 0.25); });
  ramp.initial_relative_factor = 100.0;
  ramp.initial_absolute_factor = std::numeric_limits<double>::quiet_NaN();
  rejected([&] { ramp.validate(2.0, 4.0, 0.25); });
  deallog << "Tolerance ramp policy endpoints and validation: PASSED"
          << std::endl;
}
