#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>

#include <metric_flow_x/steady_network.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

void
test()
{
  const auto close = [](const double actual, const double expected) {
    AssertThrow(std::abs(actual - expected) <=
                  2e-12 * std::max(1.0, std::abs(expected)),
                ExcMessage("Incorrect analytical network solution."));
  };
  const auto check_balance = [&close](const SteadyNetworkSolution &state) {
    for (const auto error : state.mass_balance_residual)
      close(error, 0.0);
  };
  // One tube, Q=3; the terminal reservoir pressure is not zero.
  auto state =
    solve_steady_network({{0, 1, 2.0}}, {3.0, 0.0}, {{1, 4.0, 10.0}});
  close(state.pressures[0], 28.0);
  close(state.pressures[1], 22.0);
  close(state.edge_flows[0], 3.0);
  close(state.outlet_flows[0], 3.0);
  check_balance(state);

  // Series resistances and reversed edge orientation.
  state = solve_steady_network({{0, 1, 2.0}, {2, 1, 3.0}},
                               {2.0, 0.0, 0.0},
                               {{2, 4.0, 5.0}});
  close(state.pressures[0], 23.0);
  close(state.pressures[1], 19.0);
  close(state.pressures[2], 13.0);
  close(state.edge_flows[1], -2.0);
  check_balance(state);

  // Parallel edges: conductances add, so the drop is 2 Pa for Q=3.
  state = solve_steady_network({{0, 1, 1.0}, {0, 1, 2.0}},
                               {3.0, 0.0},
                               {{1, 1.0, 0.0}});
  close(state.pressures[0], 5.0);
  close(state.pressures[1], 3.0);
  close(state.edge_flows[0], 2.0);
  close(state.edge_flows[1], 1.0);
  check_balance(state);

  // Bifurcation: two downstream branches with total resistance 2 and 4.
  state = solve_steady_network({{0, 1, 1.0}, {1, 2, 1.0}, {1, 3, 2.0}},
                               {3.0, 0.0, 0.0, 0.0},
                               {{2, 1.0, 10.0}, {3, 2.0, 10.0}});
  close(state.pressures[0], 17.0);
  close(state.pressures[1], 14.0);
  close(state.edge_flows[0], 3.0);
  close(state.edge_flows[1], 2.0);
  close(state.edge_flows[2], 1.0);
  check_balance(state);

  // A triangle has R_eq=1||(1+1)=2/3. This is not a tree solve.
  state = solve_steady_network({{0, 1, 1.0}, {1, 2, 1.0}, {0, 2, 1.0}},
                               {3.0, 0.0, 0.0},
                               {{2, 2.0, 0.0}});
  close(state.pressures[0], 8.0);
  close(state.pressures[1], 7.0);
  close(state.pressures[2], 6.0);
  close(state.edge_flows[0], 1.0);
  close(state.edge_flows[1], 1.0);
  close(state.edge_flows[2], 2.0);
  check_balance(state);

  // Independent components and negative prescribed injection are valid.
  state = solve_steady_network({{0, 1, 1.0}},
                               {2.0, 0.0, -1.0},
                               {{1, 2.0, 3.0}, {2, 4.0, 8.0}});
  close(state.pressures[0], 9.0);
  close(state.pressures[1], 7.0);
  close(state.pressures[2], 4.0);
  close(state.outlet_flows[1], -1.0);
  check_balance(state);
  state = solve_steady_network({}, {0.0}, {{0, 1.0, 0.0}});
  close(state.pressures[0], 0.0);
  check_balance(state);

  // Unequal reservoir pressures cause a circulating flow, even without an
  // explicit injection: two terminal resistors in series, Q=(9-3)/(2+4)=1.
  state = solve_steady_network({}, {0.0}, {{0, 2.0, 9.0}, {0, 4.0, 3.0}});
  close(state.pressures[0], 7.0);
  close(state.outlet_flows[0], -1.0);
  close(state.outlet_flows[1], 1.0);
  check_balance(state);

  const double pi = numbers::PI;
  close(tube_resistance(2.0, 0.5, 0.5, 3.0), 768.0 / pi);
  close(tube_resistance(3.0, 1.0, 2.0, 1.0), 7.0 / pi);
  close(tube_resistance(3.0, 2.0, 1.0, 1.0), 7.0 / pi);
  close(tube_resistance(3.0, 1.0, 2.0, 1.0, 0.0), 3.5 / pi);
  close(tube_resistance(2.0, 0.5, std::nextafter(0.5, 1.0), 3.0), 768.0 / pi);

  const auto reject = [](const std::function<void()> &operation) {
    bool failed = false;
    try
      {
        operation();
      }
    catch (const std::exception &)
      {
        failed = true;
      }
    AssertThrow(failed, ExcMessage("Invalid hydraulic data were accepted."));
  };
  reject([] { solve_steady_network({}, {}, {}); });
  reject([] { solve_steady_network({{0, 1, 1.0}}, {1.0, 0.0}, {}); });
  reject([] { solve_steady_network({}, {0.0, 0.0}, {{0, 1.0, 0.0}}); });
  reject([] { solve_steady_network({{0, 0, 1.0}}, {0.0}, {{0, 1.0, 0.0}}); });
  reject([] { solve_steady_network({{0, 2, 1.0}}, {0.0}, {{0, 1.0, 0.0}}); });
  reject([] {
    solve_steady_network({{0, 1, 0.0}}, {1.0, 0.0}, {{1, 1.0, 0.0}});
  });
  reject([] { solve_steady_network({}, {0.0}, {{0, -1.0, 0.0}}); });
  reject([] { solve_steady_network({}, {0.0}, {{1, 1.0, 0.0}}); });
  reject([] {
    solve_steady_network({},
                         {std::numeric_limits<double>::infinity()},
                         {{0, 1.0, 0.0}});
  });
  reject([] {
    solve_steady_network({},
                         {0.0},
                         {{0, 1.0, std::numeric_limits<double>::quiet_NaN()}});
  });
  reject([] { tube_resistance(1.0, 0.0, 1.0, 1.0); });
  reject([] { tube_resistance(1.0, 1.0, 1.0, 0.0); });
  reject([] { tube_resistance(1.0, 1.0, 1.0, 1.0, -2.0); });
  reject([] {
    tube_resistance(1.0, 1.0, 1.0, std::numeric_limits<double>::infinity());
  });
  deallog << "Steady hydraulic graph, tapered resistance and validation: PASSED"
          << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
