#include <deal.II/base/mpi.h>

#include <metric_flow_x/inflow_series.h>

#include <fstream>
#include <iomanip>

#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

void
test()
{
  const auto write = [](const std::string &content) {
    std::ofstream file("inflow.csv", std::ios::binary);
    file << content;
  };
  write("time_s,flow_m3_s\r\n0,0\r\n0.5,1e-8\r\n1,0\r\n");
  InflowSeries data;
  data.read("inflow.csv");
  for (const double time : {0.0, 0.25, 0.5, 0.75, 1.0})
    AssertThrow(std::abs(data.value(time) -
                         (1e-8 - 2e-8 * std::abs(time - 0.5))) < 1e-23,
                ExcInternalError());
  for (const double time : {-0.01, 1.01})
    {
      bool rejected = false;
      try
        {
          data.value(time);
        }
      catch (const std::out_of_range &)
        {
          rejected = true;
        }
      AssertThrow(rejected, ExcInternalError());
    }
  data.read("inflow.csv", 1.0);
  AssertThrow(data.value(1.25) == data.value(0.25), ExcInternalError());
  AssertThrow(data.value(-0.25) == data.value(0.75), ExcInternalError());
  for (const auto &content : {"time,flow\n0,0\n1,0\n",
                              "time_s,flow_m3_s\n0,0\n",
                              "time_s,flow_m3_s\n0,0\n0,1\n",
                              "time_s,flow_m3_s\n1,0\n0,1\n",
                              "time_s,flow_m3_s\n0,0\n1,nan\n",
                              "time_s,flow_m3_s\n0,0\n1,0,7\n",
                              "time_s,flow_m3_s\n0,0\n1,\n",
                              "time_s,flow_m3_s\n0,0\n1,1e-8suffix\n",
                              "time_s,flow_m3_s\n0,0\n1suffix,1e-8\n",
                              "time_s,flow_m3_s\n0,0\n1,inf\n",
                              "time_s,flow_m3_s\n0,0\n1,1e999\n",
                              "time_s,flow_m3_s\n0,0\n1\n",
                              "time_s,time_s,flow_m3_s\n0,0,0\n1,1,0\n",
                              "time_s,flow_m3_s,note,note\n0,0,a,b\n1,0,a,b\n",
                              "time_s,flow_m3_s,\n0,0,a\n1,0,a\n"})
    {
      write(content);
      bool rejected = false;
      try
        {
          data.read("inflow.csv");
        }
      catch (const std::invalid_argument &)
        {
          rejected = true;
        }
      AssertThrow(rejected, ExcInternalError());
    }
  write("time_s,flow_m3_s\n0,0\n1,1e-8\n");
  for (const double period :
       {-1.0, 0.5, 1.0, std::numeric_limits<double>::infinity()})
    {
      bool rejected = false;
      try
        {
          data.read("inflow.csv", period);
        }
      catch (const std::invalid_argument &)
        {
          rejected = true;
        }
      AssertThrow(rejected, ExcInternalError());
    }
  // Named columns and quoting are handled by the same rapidcsv dependency as
  // the terminal-parameter table. Only the two selected columns are numeric.
  write("\xef\xbb\xbf\"note\", \"flow_m3_s\", \"time_s\"\r\n"
        "\"unused, multiline\nmetadata\", \"0\", \"0\"\r\n"
        "ignored,1e-8,1\r\n");
  data.read("inflow.csv");
  AssertThrow(std::abs(data.value(0.25) - 2.5e-9) < 1e-23, ExcInternalError());
  bool missing_rejected = false;
  try
    {
      data.read("missing_inflow.csv");
    }
  catch (const std::invalid_argument &)
    {
      missing_rejected = true;
    }
  AssertThrow(missing_rejected, ExcInternalError());

  // A 20-second history at 1 ms with 40 fields uses named time/flow columns;
  // the other 38 fields need not be numbers.
  {
    std::ofstream file("inflow.csv");
    file << "flow_m3_s,time_s";
    for (unsigned int column = 0; column < 38; ++column)
      file << ",extra_" << column;
    file << '\n' << std::setprecision(17);
    for (unsigned int step = 0; step <= 20000; ++step)
      {
        const double t = step * 1e-3;
        file << 1e-8 + 2e-9 * t << ',' << t;
        for (unsigned int column = 0; column < 38; ++column)
          file << ",not_numeric";
        file << '\n';
      }
  }
  data.read("inflow.csv");
  for (const double t : {0., 0.0005, 0.001, 10.1235, 19.9995, 20.})
    AssertThrow(std::abs(data.value(t) - (1e-8 + 2e-9 * t)) < 1e-20,
                ExcInternalError());
  deallog << "SI inflow interpolation and validation: PASSED" << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
