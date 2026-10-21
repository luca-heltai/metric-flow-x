#ifndef METRIC_FLOW_X_INFLOW_SERIES_H
#define METRIC_FLOW_X_INFLOW_SERIES_H

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace MetricFlowX
{
  /** SI inflow data. Linear interpolation, with no implicit extrapolation. */
  class InflowSeries
  {
  public:
    void
    read(const std::string &filename, const double period = 0.);

    bool
    empty() const
    {
      return samples.empty();
    }

    double
    value(double time) const
    {
      if (samples.empty() || !std::isfinite(time))
        throw std::invalid_argument("Invalid inflow evaluation");
      if (periodicity > 0. &&
          (time < samples.front().first || time > samples.back().first))
        {
          long double phase =
            std::fmod(static_cast<long double>(time) - samples.front().first,
                      static_cast<long double>(periodicity));
          if (phase < 0.)
            phase += periodicity;
          time = static_cast<double>(samples.front().first + phase);
        }
      if (time < samples.front().first || time > samples.back().first)
        throw std::out_of_range("Inflow extrapolation is forbidden");
      auto right = std::upper_bound(samples.begin(),
                                    samples.end(),
                                    time,
                                    [](double t, const auto &sample) {
                                      return t < sample.first;
                                    });
      if (right == samples.end())
        return samples.back().second;
      const auto &a = *(right - 1);
      const auto &b = *right;
      return a.second +
             (b.second - a.second) * ((time - a.first) / (b.first - a.first));
    }

  private:
    std::vector<std::pair<double, double>> samples;
    double                                 periodicity = 0.;
  };
} // namespace MetricFlowX
#endif
