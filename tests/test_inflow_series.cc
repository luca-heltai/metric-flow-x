#include <deal.II/base/mpi.h>

#include <metric_flow_x/inflow_series.h>

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
                              "time_s,flow_m3_s\n0,0\n1,0,7\n"})
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
  deallog << "SI inflow interpolation and validation: PASSED" << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
