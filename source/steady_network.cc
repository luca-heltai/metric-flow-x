#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <metric_flow_x/steady_network.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace MetricFlowX
{
  using namespace dealii;

  double
  tube_resistance(const double length,
                  const double radius0,
                  const double radius1,
                  const double viscosity,
                  const double profile)
  {
    for (const double value : {length, radius0, radius1, viscosity})
      AssertThrow(
        std::isfinite(value) && value > 0.0,
        ExcMessage(
          "Tube length, radii and viscosity must be finite and positive."));
    AssertThrow(std::isfinite(profile) && profile > -2.0,
                ExcMessage(
                  "Tube friction profile must be finite and greater than -2."));
    const long double a = radius0, b = radius1;
    // The divided difference of r^-3, expanded to avoid cancellation when
    // the endpoint radii coincide or differ only by roundoff.
    const long double integral = static_cast<long double>(length) *
                                 (a * a + a * b + b * b) /
                                 (3.0L * a * a * a * b * b * b);
    const double resistance =
      static_cast<double>(2.0L * (static_cast<long double>(profile) + 2.0L) *
                          viscosity / numbers::PI * integral);
    AssertThrow(std::isfinite(resistance) && resistance > 0.0,
                ExcMessage(
                  "Tube resistance is outside the finite positive range."));
    return resistance;
  }

  SteadyNetworkSolution
  solve_steady_network(const std::vector<HydraulicEdge>   &edges,
                       const std::vector<double>          &nodal_injections,
                       const std::vector<HydraulicOutlet> &outlets)
  {
    AssertThrow(!nodal_injections.empty() &&
                  nodal_injections.size() <=
                    std::numeric_limits<unsigned int>::max() / 10,
                ExcMessage(
                  "A steady network needs a nonempty, indexable node set."));
    const unsigned int n = nodal_injections.size();
    for (const double value : nodal_injections)
      AssertThrow(std::isfinite(value),
                  ExcMessage("Network injections must be finite."));
    const auto conductance = [](const double resistance) {
      AssertThrow(
        std::isfinite(resistance) && resistance > 0.0 &&
          std::isfinite(1.0 / resistance),
        ExcMessage(
          "Hydraulic resistance and conductance must be finite and positive."));
      return 1.0 / resistance;
    };
    std::vector<unsigned int> component(n);
    std::iota(component.begin(), component.end(), 0);
    const auto find_root = [&component](unsigned int node) {
      while (component[node] != node)
        {
          component[node] = component[component[node]];
          node            = component[node];
        }
      return node;
    };
    DynamicSparsityPattern dsp(n);
    for (unsigned int i = 0; i < n; ++i)
      dsp.add(i, i);
    for (const auto &edge : edges)
      {
        AssertThrow(
          edge.node0 < n && edge.node1 < n && edge.node0 != edge.node1,
          ExcMessage(
            "Hydraulic edges must connect two different valid nodes."));
        conductance(edge.resistance);
        component[find_root(edge.node0)] = find_root(edge.node1);
        dsp.add(edge.node0, edge.node1);
        dsp.add(edge.node1, edge.node0);
      }
    std::vector<bool> anchored(n, false);
    for (const auto &outlet : outlets)
      {
        AssertThrow(
          outlet.node < n && std::isfinite(outlet.pressure),
          ExcMessage(
            "Hydraulic outlets need a valid node and finite reservoir pressure."));
        conductance(outlet.resistance);
        anchored[find_root(outlet.node)] = true;
      }
    for (unsigned int i = 0; i < n; ++i)
      AssertThrow(
        anchored[find_root(i)],
        ExcMessage(
          "Every steady-network component needs a pressure reference through an outlet."));

    SparsityPattern sparsity;
    sparsity.copy_from(dsp);
    SparseMatrix<double> matrix(sparsity);
    Vector<double>       rhs(n), pressure(n);
    for (unsigned int i = 0; i < n; ++i)
      rhs[i] = nodal_injections[i];
    for (const auto &edge : edges)
      {
        const double g = conductance(edge.resistance);
        matrix.add(edge.node0, edge.node0, g);
        matrix.add(edge.node1, edge.node1, g);
        matrix.add(edge.node0, edge.node1, -g);
        matrix.add(edge.node1, edge.node0, -g);
      }
    for (const auto &outlet : outlets)
      {
        const double g = conductance(outlet.resistance);
        matrix.add(outlet.node, outlet.node, g);
        rhs[outlet.node] += g * outlet.pressure;
      }

    // Symmetric diagonal scaling preserves the SPD graph operator and keeps
    // SI conductances of very different magnitudes away from solver thresholds.
    Vector<double> scaling(n);
    for (unsigned int i = 0; i < n; ++i)
      {
        AssertThrow(std::isfinite(matrix.diag_element(i)) &&
                      matrix.diag_element(i) > 0.0 && std::isfinite(rhs[i]),
                    ExcMessage("Steady-network assembly overflowed."));
        scaling[i] = 1.0 / std::sqrt(matrix.diag_element(i));
        rhs[i] *= scaling[i];
        AssertThrow(std::isfinite(rhs[i]),
                    ExcMessage("Scaled network load overflowed."));
      }
    for (unsigned int i = 0; i < n; ++i)
      for (auto entry = matrix.begin(i); entry != matrix.end(i); ++entry)
        entry->value() =
          static_cast<double>(static_cast<long double>(entry->value()) *
                              scaling[i] * scaling[entry->column()]);

#ifdef DEAL_II_WITH_UMFPACK
    SparseDirectUMFPACK inverse;
    inverse.initialize(matrix);
    inverse.vmult(pressure, rhs);
#else
    if (rhs.l2_norm() != 0.0)
      {
        SolverControl                          control(std::max(1000u, 10 * n),
                              64 * std::numeric_limits<double>::epsilon() *
                                rhs.l2_norm());
        SolverCG<Vector<double>>               solver(control);
        PreconditionSSOR<SparseMatrix<double>> preconditioner;
        preconditioner.initialize(matrix);
        solver.solve(matrix, pressure, rhs, preconditioner);
      }
#endif
    SteadyNetworkSolution result;
    result.pressures.resize(n);
    for (unsigned int i = 0; i < n; ++i)
      {
        result.pressures[i] = pressure[i] * scaling[i];
        AssertThrow(std::isfinite(result.pressures[i]),
                    ExcMessage(
                      "Steady-network solve produced a nonfinite pressure."));
      }
    result.mass_balance_residual.resize(n, 0.0);
    std::vector<long double> residual_scale(n);
    for (unsigned int i = 0; i < n; ++i)
      {
        result.mass_balance_residual[i] = -nodal_injections[i];
        residual_scale[i]               = std::abs(nodal_injections[i]);
      }
    for (const auto &edge : edges)
      {
        const double p0 = result.pressures[edge.node0];
        const double p1 = result.pressures[edge.node1];
        const double q  = (p0 - p1) / edge.resistance;
        result.edge_flows.push_back(q);
        result.mass_balance_residual[edge.node0] += q;
        result.mass_balance_residual[edge.node1] -= q;
        const long double scale =
          (static_cast<long double>(std::abs(p0)) + std::abs(p1)) /
          edge.resistance;
        residual_scale[edge.node0] += scale;
        residual_scale[edge.node1] += scale;
      }
    for (const auto &outlet : outlets)
      {
        const double p = result.pressures[outlet.node];
        const double q = (p - outlet.pressure) / outlet.resistance;
        result.outlet_flows.push_back(q);
        result.mass_balance_residual[outlet.node] += q;
        residual_scale[outlet.node] +=
          (static_cast<long double>(std::abs(p)) + std::abs(outlet.pressure)) /
          outlet.resistance;
      }
    for (unsigned int i = 0; i < n; ++i)
      AssertThrow(
        std::isfinite(result.mass_balance_residual[i]) &&
          std::abs(result.mass_balance_residual[i]) <=
            512 * std::numeric_limits<double>::epsilon() * residual_scale[i],
        ExcMessage(
          "Steady-network solve failed its componentwise mass-balance backward-error check."));
    return result;
  }
} // namespace MetricFlowX
