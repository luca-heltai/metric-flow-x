#include <metric_flow_x/inflow_series.h>

#include <fstream>
#include <locale>
#include <set>
#include <sstream>

#include "rapidcsv.h"

namespace MetricFlowX
{
  void
  InflowSeries::read(const std::string &filename, const double period)
  {
    samples.clear();
    periodicity = period;
    if (!std::isfinite(period) || period < 0.)
      throw std::invalid_argument(
        "Inflow period must be finite and nonnegative");

    std::ifstream input(filename);
    if (!input)
      throw std::invalid_argument("Cannot read inflow CSV: " + filename);
    rapidcsv::SeparatorParams separators(',', true);
    separators.mQuotedLinebreaks = true;
    const rapidcsv::Document csv(input, rapidcsv::LabelParams(), separators);
    const auto               columns = csv.GetColumnNames();
    std::set<std::string>    names;
    for (const auto &name : columns)
      if (name.empty() || !names.insert(name).second)
        throw std::invalid_argument(
          "Inflow CSV column names must be nonempty and unique");
    if (csv.GetColumnIdx("time_s") < 0 || csv.GetColumnIdx("flow_m3_s") < 0)
      throw std::invalid_argument(
        "Inflow CSV requires time_s and flow_m3_s columns");

    const auto parse_value = [](const std::string &text, double &value) {
      std::istringstream field(text);
      field.imbue(std::locale::classic());
      if (!(field >> value) || !std::isfinite(value) ||
          (field >> std::ws && !field.eof()))
        throw std::invalid_argument("Invalid inflow CSV numeric value: " +
                                    text);
    };
    samples.reserve(csv.GetRowCount());
    for (std::size_t row = 0; row < csv.GetRowCount(); ++row)
      {
        if (csv.GetRow<std::string>(row).size() != columns.size())
          throw std::invalid_argument("Malformed inflow CSV row " +
                                      std::to_string(row + 2));
        const double time = csv.GetCell<double>("time_s", row, parse_value);
        const double flow = csv.GetCell<double>("flow_m3_s", row, parse_value);
        if (!samples.empty() && time <= samples.back().first)
          throw std::invalid_argument(
            "Inflow CSV times must be strictly increasing");
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
} // namespace MetricFlowX
