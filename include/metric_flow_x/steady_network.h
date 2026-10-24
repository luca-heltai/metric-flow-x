#ifndef METRIC_FLOW_X_STEADY_NETWORK_H
#define METRIC_FLOW_X_STEADY_NETWORK_H

#include <vector>

namespace MetricFlowX
{
  /** An oriented hydraulic edge; resistance is in Pa s / m^3. */
  struct HydraulicEdge
  {
    unsigned int node0, node1;
    double       resistance;
  };

  /** A resistor from a graph node to a reservoir at the given pressure. */
  struct HydraulicOutlet
  {
    unsigned int node;
    double       resistance, pressure;
  };

  struct SteadyNetworkSolution
  {
    std::vector<double> pressures;
    std::vector<double> edge_flows;
    std::vector<double> outlet_flows;
    // Outgoing edge/outlet flows minus prescribed injection, in m^3/s.
    std::vector<double> mass_balance_residual;
  };

  /** Solve the steady resistive network in SI units.
   *
   * nodal_injections defines the node count. Positive injection adds fluid;
   * edge flow is positive from node0 to node1 and outlet flow is positive
   * towards its reservoir. Every connected component needs an outlet, which
   * removes the pressure nullspace. Parallel edges, cycles and independently
   * anchored disconnected components are supported.
   *
   * Uses sparse UMFPACK when available, otherwise CG with SSOR. Invalid data,
   * missing pressure references, solver nonconvergence and an excessive
   * componentwise backward residual are errors. This rigid graph solution is
   * a warm start; it does not imply consistency of the compliant HDG DAE.
   */
  SteadyNetworkSolution
  solve_steady_network(const std::vector<HydraulicEdge>   &edges,
                       const std::vector<double>          &nodal_injections,
                       const std::vector<HydraulicOutlet> &outlets);

  /** Resistance for a straight tube whose radius varies linearly.
   *
   * Integrates 2*(profile+2)*mu/(pi*r(s)^4) along the tube, matching the 1D
   * velocity-equation friction convention. profile=2 gives Poiseuille flow.
   * Length, radii and viscosity must be positive; profile must exceed -2.
   */
  double
  tube_resistance(double length,
                  double radius0,
                  double radius1,
                  double viscosity,
                  double profile = 2.0);
} // namespace MetricFlowX

#endif
