#ifndef METRIC_FLOW_X_INFLOW_SERIES_H
#define METRIC_FLOW_X_INFLOW_SERIES_H

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
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
    read(const std::string &filename, const double period = 0.)
    {
      samples.clear();
      periodicity = period;
      if (!std::isfinite(period) || period < 0.)
        throw std::invalid_argument(
          "Inflow period must be finite and nonnegative");
      std::ifstream input(filename);
      std::string   line;
      if (!std::getline(input, line))
        throw std::invalid_argument("Cannot read inflow CSV: " + filename);
      if (!line.empty() && line.back() == '\r')
        line.pop_back();
      if (line != "time_s,flow_m3_s")
        throw std::invalid_argument(
          "Expected inflow CSV header time_s,flow_m3_s");
      while (std::getline(input, line))
        {
          if (!line.empty() && line.back() == '\r')
            line.pop_back();
          std::istringstream row(line);
          double             time, flow;
          char               comma;
          if (!(row >> time >> comma >> flow) || comma != ',' ||
              !std::isfinite(time) || !std::isfinite(flow) ||
              (row >> std::ws && !row.eof()) ||
              (!samples.empty() && time <= samples.back().first))
            throw std::invalid_argument("Invalid or unordered inflow CSV row");
          samples.emplace_back(time, flow);
        }
      if (samples.size() < 2)
        throw std::invalid_argument("Inflow CSV requires at least two samples");
      if (!std::isfinite(samples.back().first - samples.front().first))
        throw std::invalid_argument("Inflow time span must be representable");
      if (period > 0. &&
          (std::abs(samples.back().first - samples.front().first - period) >
             1e-10 * period ||
           std::abs(samples.front().second - samples.back().second) >
             1e-14 + 1e-8 * std::abs(samples.front().second)))
        throw std::invalid_argument(
          "Periodic inflow must span one closed period");
    }

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
