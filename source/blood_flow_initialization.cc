#include <deal.II/base/mpi.h>

#include <deal.II/lac/full_matrix.h>

#include <metric_flow_x/blood_flow_system.h>
#include <metric_flow_x/steady_network.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>

namespace MetricFlowX
{
  using namespace dealii;

  /// @cond INTERNAL
  namespace
  {
    using Index = types::global_dof_index;

    void
    collective_check(const MPI_Comm comm, const std::string &error)
    {
      for (const auto &message : Utilities::MPI::all_gather(comm, error))
        AssertThrow(message.empty(), ExcMessage(message));
    }

    struct TraceRow
    {
      Index                                 index;
      double                                residual, state_scale, error_weight;
      std::vector<std::pair<Index, double>> entries;
      template <class Archive>
      void
      serialize(Archive &archive, const unsigned int)
      {
        archive &index &residual &state_scale &error_weight &entries;
      }
    };

    struct TraceCorrection
    {
      std::vector<std::pair<Index, double>> values, row_weights;
      double                                norm          = 0.;
      unsigned int                          largest_block = 0;
      template <class Archive>
      void
      serialize(Archive &archive, const unsigned int)
      {
        archive &values &row_weights &norm &largest_block;
      }
    };

    // Only algebraic columns are retained. The native equations couple one
    // physical face or junction at a time; continuity rows join duplicate
    // traces to the same block. No full-system factorization is needed.
    TraceCorrection
    trace_correction(const std::vector<std::vector<TraceRow>> &parts,
                     const double                              tolerance)
    {
      std::vector<TraceRow> rows;
      for (const auto &part : parts)
        rows.insert(rows.end(), part.begin(), part.end());
      std::map<Index, unsigned int> number;
      for (unsigned int i = 0; i < rows.size(); ++i)
        AssertThrow(number.emplace(rows[i].index, i).second,
                    ExcMessage(
                      "An initial algebraic row has multiple owners."));
      std::vector<unsigned int> parent(rows.size());
      std::iota(parent.begin(), parent.end(), 0);
      const auto find = [&parent](unsigned int i) {
        while (parent[i] != i)
          {
            parent[i] = parent[parent[i]];
            i         = parent[i];
          }
        return i;
      };
      for (const auto &row : rows)
        {
          AssertThrow(
            std::isfinite(row.residual) && std::isfinite(row.state_scale) &&
              row.state_scale > 0. && std::isfinite(row.error_weight) &&
              row.error_weight > 0.,
            ExcMessage("Non-finite or invalid initial algebraic scaling."));
          for (const auto &[column, value] : row.entries)
            {
              (void)column;
              AssertThrow(std::isfinite(value),
                          ExcMessage("Non-finite initial algebraic Jacobian."));
            }
        }
      std::vector<double> weights(rows.size(), 0.);
      for (unsigned int i = 0; i < rows.size(); ++i)
        for (const auto &[column, value] : rows[i].entries)
          if (value != 0. && number.count(column))
            {
              const unsigned int j = number.at(column);
              parent[find(i)]      = find(j);
              weights[i] += std::abs(value) * rows[j].error_weight;
            }
      TraceCorrection result;
      for (unsigned int i = 0; i < rows.size(); ++i)
        {
          AssertThrow(std::isfinite(rows[i].residual) &&
                        std::isfinite(weights[i]),
                      ExcMessage("Non-finite initial algebraic equation."));
          weights[i] = std::max(weights[i], std::numeric_limits<double>::min());
          result.row_weights.emplace_back(rows[i].index, weights[i]);
          result.norm =
            std::max(result.norm, std::abs(rows[i].residual) / weights[i]);
        }
      if (result.norm <= tolerance)
        return result;
      std::map<unsigned int, std::vector<unsigned int>> blocks;
      for (unsigned int i = 0; i < rows.size(); ++i)
        blocks[find(i)].push_back(i);
      for (const auto &[key, block] : blocks)
        {
          (void)key;
          const unsigned int n = block.size();
          result.largest_block = std::max(result.largest_block, n);
          FullMatrix<double>            matrix(n, n);
          Vector<double>                rhs(n), update(n);
          std::map<Index, unsigned int> local;
          for (unsigned int i = 0; i < n; ++i)
            local.emplace(rows[block[i]].index, i);
          for (unsigned int i = 0; i < n; ++i)
            {
              const auto k = block[i];
              rhs[i]       = -rows[k].residual / weights[k];
              for (const auto &[column, value] : rows[k].entries)
                if (local.count(column))
                  {
                    const auto j = local.at(column);
                    matrix(i, j) +=
                      value * rows[block[j]].state_scale / weights[k];
                  }
            }
          matrix.gauss_jordan();
          matrix.vmult(update, rhs);
          for (unsigned int i = 0; i < n; ++i)
            {
              const double value = update[i] * rows[block[i]].state_scale;
              AssertThrow(std::isfinite(value),
                          ExcMessage("Singular initial face/junction block."));
              result.values.emplace_back(rows[block[i]].index, value);
            }
        }
      return result;
    }
  } // namespace
  /// @endcond

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::compute_steady_network_initial_solution(
    VectorType  &state,
    const double t)
  {
    TimerOutput::Scope timer(computing_timer, "initialize_steady_network");
    AssertThrow(
      outlet_type == "RCR",
      ExcMessage(
        "Steady network initialization requires RCR/resistive terminal boundaries."));
    AssertThrow(!initial_graph_cells_.empty(),
                ExcMessage("The coarse hydraulic graph is unavailable."));
    std::vector<HydraulicEdge>                    edges;
    std::map<types::coarse_cell_id, unsigned int> edge_number;
    std::vector<double>          injections(initial_graph_points_.size(), 0.);
    std::vector<HydraulicOutlet> outlets;
    const auto                   area = [this](const InitialGraphCell &edge,
                             const Point<spacedim>  &point) {
      const auto &v = vessel_map.at(edge.vessel_id);
      if (v.r_in <= 0. || v.r_out <= 0.)
        return v.a_d;
      const auto direction = (initial_graph_points_[edge.node1] -
                              initial_graph_points_[edge.node0]) /
                             initial_graph_points_[edge.node1].distance(
                               initial_graph_points_[edge.node0]);
      const auto [lo, hi] = vessel_s_bounds.at(edge.vessel_id);
      const double x =
        hi - lo > 1e-14 ?
                            std::clamp((point * direction - lo) / (hi - lo), 0., 1.) :
                            .5;
      const double r = v.r_in + x * (v.r_out - v.r_in);
      return numbers::PI * r * r;
    };
    for (const auto &[id, edge] : initial_graph_cells_)
      {
        const auto &p0 = initial_graph_points_[edge.node0];
        const auto &p1 = initial_graph_points_[edge.node1];
        edge_number.emplace(id, edges.size());
        edges.push_back(
          {edge.node0,
           edge.node1,
           tube_resistance(p0.distance(p1),
                           std::sqrt(area(edge, p0) / numbers::PI),
                           std::sqrt(area(edge, p1) / numbers::PI),
                           viscosity,
                           friction_profile)});
      }
    inflow_function.set_time(t);
    const double inflow = inflow_series.empty() ?
                            inflow_function.value(Point<1>(0.)) :
                            inflow_series.value(t);
    for (const auto &[node, bid] : initial_graph_boundaries_)
      if (bid == 0)
        injections[node] += inflow;
      else
        {
          AssertThrow(
            rcr_map.count(bid),
            ExcMessage(
              "Steady network initialization needs a resistance for every terminal."));
          const auto &r = rcr_map.at(bid);
          outlets.push_back({node, r.R1 + r.R2, r.P_out});
        }
    SteadyNetworkSolution graph;
    std::string           error;
    if (this_mpi_process == 0)
      try
        {
          graph = solve_steady_network(edges, injections, outlets);
        }
      catch (const std::exception &e)
        {
          error = e.what();
        }
    error = Utilities::MPI::broadcast(mpi_communicator_, error, 0);
    AssertThrow(error.empty(),
                ExcMessage("Steady network solve failed: " + error));
    graph.pressures =
      Utilities::MPI::broadcast(mpi_communicator_, graph.pressures, 0);
    graph.edge_flows =
      Utilities::MPI::broadcast(mpi_communicator_, graph.edge_flows, 0);
    state = 0.;
    std::vector<std::pair<Index, double>> values;
    std::vector<Index>                    ldofs(fe_->n_dofs_per_cell());
    try
      {
        for (const auto &cell : dof_handler_.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const auto   id      = cell->id().get_coarse_cell_id();
              const auto  &edge    = initial_graph_cells_.at(id);
              const auto  &p0      = initial_graph_points_[edge.node0];
              const auto  &p1      = initial_graph_points_[edge.node1];
              const auto   tangent = (p1 - p0) / p0.distance(p1);
              const auto   n       = edge_number.at(id);
              const double r0      = std::sqrt(area(edge, p0) / numbers::PI);
              const double r1      = std::sqrt(area(edge, p1) / numbers::PI);
              const double sign =
                compute_directional_vector(cell) * tangent >= 0. ? 1. : -1.;
              const auto &v = vessel_map.at(edge.vessel_id);
              cell->get_dof_indices(ldofs);
              for (unsigned int i = 0; i < ldofs.size(); ++i)
                {
                  const auto   point = support_point(cell, i);
                  const double x =
                    std::clamp(((point - p0) * tangent) / p0.distance(p1),
                               0.,
                               1.);
                  const double fraction = x == 0. ?
                                            0. :
                                            tube_resistance(p0.distance(p1) * x,
                                                            r0,
                                                            r0 + x * (r1 - r0),
                                                            viscosity,
                                                            friction_profile) /
                                              edges[n].resistance;
                  const double pressure =
                    graph.pressures[edge.node0] +
                    fraction * (graph.pressures[edge.node1] -
                                graph.pressures[edge.node0]);
                  const double ad = compute_a_d_at_point(cell, point);
                  const double root =
                    std::sqrt(ad) +
                    ad / v.beta_p *
                      (pressure - v.p0 - v.p_d -
                       external_pressure(
                         {t, point, edge.vessel_id, cell->id()}));
                  AssertThrow(
                    std::isfinite(root) && root > 0.,
                    ExcMessage(
                      "The graph pressure has no admissible positive-area tube-law inverse."));
                  const double a = root * root;
                  const auto   component =
                    fe_->system_to_component_index(i).first;
                  values.emplace_back(ldofs[i],
                                      component == 0 || component == 2 ?
                                        a :
                                        sign * graph.edge_flows[n] / a);
                }
            }
      }
    catch (const std::exception &e)
      {
        error = e.what();
      }
    collective_check(mpi_communicator_, error);
    for (const auto &[i, value] : values)
      state(i) = value;
    for (const auto &[node, bid] : initial_graph_boundaries_)
      if (rcr_pc_dof.count(bid) &&
          locally_owned_dofs_.is_element(rcr_pc_dof.at(bid)))
        {
          const auto  &r = rcr_map.at(bid);
          const double q = (graph.pressures[node] - r.P_out) / (r.R1 + r.R2);
          state(rcr_pc_dof.at(bid)) = r.P_out + r.R2 * q;
        }
    state.compress(VectorOperation::insert);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::correct_initial_traces(VectorType  &state,
                                                         const double t)
  {
    TimerOutput::Scope timer(computing_timer, "initialize_trace_unknowns");
    unsigned int       invalid_state = 0;
    for (const auto i : locally_owned_dofs_)
      if (!std::isfinite(state(i)) || ((component_dofs_[0].is_element(i) ||
                                        component_dofs_[2].is_element(i)) &&
                                       state(i) <= 0.))
        invalid_state = 1;
    AssertThrow(Utilities::MPI::max(invalid_state, mpi_communicator_) == 0,
                ExcMessage("Non-finite or inadmissible initial state."));
    VectorType g = make_state(), delta = make_state(), trial = make_state();
    VectorType scales = make_state();
    const VectorType atol =
      use_componentwise_tolerances ? make_absolute_tolerances() : make_state();
    reinit_state(initial_trace_weights_);
    initial_trace_weights_ = 0.;
    std::vector<Index> ldofs(fe_->n_dofs_per_cell());
    for (const auto &cell : dof_handler_.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(ldofs);
          for (unsigned int i = 0; i < ldofs.size(); ++i)
            {
              const auto   component = fe_->system_to_component_index(i).first;
              const double ad =
                compute_a_d_at_point(cell, support_point(cell, i));
              scales(ldofs[i]) =
                component == 0 || component == 2 ?
                  ad :
                  compute_wave_speed(ad, cell->material_id(), ad);
            }
        }
    scales.compress(VectorOperation::insert);
    const auto assemble_g = [this, t](const VectorType &y,
                                      VectorType       &residual) {
      update_ghosted_vectors(y);
      residual = 0.;
      assemble_trace_interior_equations(t, y_relevant, residual);
      assemble_trace_boundary_equations(t, y_relevant, residual);
      assemble_trace_junction_equations(t, y_relevant, residual);
      assemble_trace_continuity_equations(y_relevant, residual);
      residual.compress(VectorOperation::add);
    };
    initial_state_diagnostics_.largest_trace_block = 0;
    for (unsigned int iteration = 0;
         iteration < initial_consistency_max_iterations;
         ++iteration)
      {
        assemble_g(state, g);
        jacobian_matrix = 0.;
        assemble_jacobian_trace_interior_block(y_relevant);
        assemble_jacobian_trace_boundary_block(t, y_relevant);
        assemble_jacobian_trace_junction_block(y_relevant);
        assemble_jacobian_trace_continuity_block();
        jacobian_matrix.compress(VectorOperation::add);
        std::vector<TraceRow> local;
        for (const auto i : trace_dofs_owned)
          {
            TraceRow row;
            row.index       = i;
            row.residual    = g(i);
            row.state_scale = scales(i);
            row.error_weight =
              (use_componentwise_tolerances ?
                 atol(i) :
                 ida_parameters.absolute_tolerance) +
              ida_parameters.relative_tolerance * std::abs(state(i));
            for (auto entry = jacobian_matrix.begin(i);
                 entry != jacobian_matrix.end(i);
                 ++entry)
              if (entry->value() != 0.)
                row.entries.emplace_back(entry->column(), entry->value());
            local.push_back(std::move(row));
          }
        const auto parts = Utilities::MPI::gather(mpi_communicator_, local, 0);
        TraceCorrection correction;
        std::string     error;
        if (this_mpi_process == 0)
          try
            {
              correction =
                trace_correction(parts, initial_consistency_tolerance);
            }
          catch (const std::exception &e)
            {
              error = e.what();
            }
        error = Utilities::MPI::broadcast(mpi_communicator_, error, 0);
        AssertThrow(error.empty(),
                    ExcMessage("Initial face/junction correction failed: " +
                               error));
        correction =
          Utilities::MPI::broadcast(mpi_communicator_, correction, 0);
        initial_state_diagnostics_.trace_iterations = iteration;
        initial_state_diagnostics_.algebraic_defect = correction.norm;
        initial_state_diagnostics_.largest_trace_block =
          std::max(initial_state_diagnostics_.largest_trace_block,
                   correction.largest_block);
        for (const auto &[i, weight] : correction.row_weights)
          if (locally_owned_dofs_.is_element(i))
            initial_trace_weights_(i) = weight;
        initial_trace_weights_.compress(VectorOperation::insert);
        if (verbosity > 0)
          pcout << "Initial trace iteration " << iteration
                << ", scaled residual " << correction.norm << '\n';
        if (correction.norm <= initial_consistency_tolerance)
          {
            jacobian_matrix = 0.;
            jacobian_matrix.compress(VectorOperation::insert);
            return;
          }
        delta = 0.;
        for (const auto &[i, value] : correction.values)
          if (locally_owned_dofs_.is_element(i))
            delta(i) = value;
        delta.compress(VectorOperation::insert);
        bool accepted = false;
        for (double damping = 1.; damping >= std::ldexp(1., -16); damping *= .5)
          {
            trial = state;
            trial.add(damping, delta);
            unsigned int invalid = 0;
            for (const auto i : locally_owned_dofs_)
              if (!std::isfinite(trial(i)) ||
                  ((component_dofs_[0].is_element(i) ||
                    component_dofs_[2].is_element(i)) &&
                   trial(i) <= 0.))
                invalid = 1;
            if (Utilities::MPI::max(invalid, mpi_communicator_) != 0)
              continue;
            assemble_g(trial, g);
            double norm = 0.;
            invalid     = 0;
            for (const auto i : trace_dofs_owned)
              {
                const double weight = initial_trace_weights_(i);
                if (!std::isfinite(g(i)) || !std::isfinite(weight) ||
                    weight <= 0.)
                  invalid = 1;
                else
                  {
                    const double scaled = std::abs(g(i)) / weight;
                    if (!std::isfinite(scaled))
                      invalid = 1;
                    else
                      norm = std::max(norm, scaled);
                  }
              }
            if (Utilities::MPI::max(invalid, mpi_communicator_) != 0)
              continue;
            norm = Utilities::MPI::max(norm, mpi_communicator_);
            if (std::isfinite(norm) &&
                (norm <= (1. - 1e-4 * damping) * correction.norm ||
                 norm <= initial_consistency_tolerance))
              {
                state    = trial;
                accepted = true;
                break;
              }
          }
        AssertThrow(
          accepted,
          ExcMessage(
            "No admissible decreasing initial trace correction; check the seed and subsonic boundary/junction branch."));
      }
    AssertThrow(
      false,
      ExcMessage(
        "Initial face/junction consistency did not converge within its iteration limit."));
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::reconstruct_initial_derivative(
    const VectorType &state,
    VectorType       &state_dot,
    const double      t)
  {
    TimerOutput::Scope timer(computing_timer, "initialize_state_derivative");
    reinit_state(state_dot);
    state_dot           = 0.;
    VectorType residual = make_state();
    assemble_residual(t, state, state_dot, residual);
    std::vector<Index> ldofs(fe_->n_dofs_per_cell());
    Vector<double> local(fe_->n_dofs_per_cell()), rate(fe_->n_dofs_per_cell());
    for (const auto &cell : dof_handler_.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(ldofs);
          residual.extract_subvector_to(ldofs.begin(),
                                        ldofs.end(),
                                        local.begin());
          per_cell_mass_inv[cell->active_cell_index()].vmult(rate, local);
          for (unsigned int i = 0; i < ldofs.size(); ++i)
            if (fe_->system_to_component_index(i).first < 2)
              state_dot(ldofs[i]) = -rate[i];
        }
    for (const auto &[bid, index] : rcr_pc_dof)
      if (locally_owned_dofs_.is_element(index))
        state_dot(index) = -residual(index) / rcr_map.at(bid).C;
    state_dot.compress(VectorOperation::insert);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::verify_initial_pair(
    const VectorType &state,
    const VectorType &state_dot,
    const double      t)
  {
    unsigned int invalid = 0;
    for (const auto i : locally_owned_dofs_)
      if (!std::isfinite(state(i)) || !std::isfinite(state_dot(i)) ||
          ((component_dofs_[0].is_element(i) ||
            component_dofs_[2].is_element(i)) &&
           state(i) <= 0.))
        invalid = 1;
    AssertThrow(Utilities::MPI::max(invalid, mpi_communicator_) == 0,
                ExcMessage(
                  "Non-finite or inadmissible initial state/derivative."));
    VectorType residual = make_state();
    assemble_residual(t, state, state_dot, residual);
    for (const auto i : locally_owned_dofs_)
      if (!std::isfinite(residual(i)))
        invalid = 1;
    AssertThrow(Utilities::MPI::max(invalid, mpi_communicator_) == 0,
                ExcMessage("Non-finite initial residual."));
    VectorType atol = make_state();
    if (use_componentwise_tolerances)
      atol = make_absolute_tolerances();
    else
      atol = ida_parameters.absolute_tolerance;
    const auto weight = [&](const Index i) {
      return atol(i) + ida_parameters.relative_tolerance * std::abs(state(i));
    };
    const double horizon = ida_parameters.initial_step_size > 0. ?
                             ida_parameters.initial_step_size :
                             std::min(1., ida_parameters.final_time - t);
    AssertThrow(std::isfinite(horizon) && horizon > 0.,
                ExcMessage("Invalid initial consistency time horizon."));
    for (const auto i : locally_owned_dofs_)
      if (!std::isfinite(weight(i)) || weight(i) <= 0.)
        invalid = 1;
    for (const auto i : trace_dofs_owned)
      if (!std::isfinite(initial_trace_weights_(i)) ||
          initial_trace_weights_(i) <= 0.)
        invalid = 1;
    AssertThrow(Utilities::MPI::max(invalid, mpi_communicator_) == 0,
                ExcMessage(
                  "Non-finite or invalid initial consistency weights."));
    double algebraic = 0., differential = 0.;
    for (const auto i : trace_dofs_owned)
      algebraic =
        std::max(algebraic, std::abs(residual(i)) / initial_trace_weights_(i));
    std::vector<Index> ldofs(fe_->n_dofs_per_cell());
    Vector<double> local(fe_->n_dofs_per_cell()), rate(fe_->n_dofs_per_cell());
    for (const auto &cell : dof_handler_.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(ldofs);
          residual.extract_subvector_to(ldofs.begin(),
                                        ldofs.end(),
                                        local.begin());
          per_cell_mass_inv[cell->active_cell_index()].vmult(rate, local);
          for (unsigned int i = 0; i < ldofs.size(); ++i)
            if (fe_->system_to_component_index(i).first < 2)
              {
                const double scaled =
                  horizon * std::abs(rate[i]) / weight(ldofs[i]);
                if (!std::isfinite(rate[i]) || !std::isfinite(scaled))
                  invalid = 1;
                else
                  differential = std::max(differential, scaled);
              }
        }
    for (const auto &[bid, index] : rcr_pc_dof)
      if (locally_owned_dofs_.is_element(index))
        {
          const double capacitance = rcr_map.at(bid).C;
          const double scaled =
            horizon * std::abs(residual(index)) / capacitance / weight(index);
          if (!std::isfinite(capacitance) || capacitance <= 0. ||
              !std::isfinite(scaled))
            invalid = 1;
          else
            differential = std::max(differential, scaled);
        }
    AssertThrow(Utilities::MPI::max(invalid, mpi_communicator_) == 0,
                ExcMessage("Non-finite initial consistency rate."));
    algebraic    = Utilities::MPI::max(algebraic, mpi_communicator_);
    differential = Utilities::MPI::max(differential, mpi_communicator_);
    initial_state_diagnostics_.algebraic_defect    = algebraic;
    initial_state_diagnostics_.differential_defect = differential;
    AssertThrow(
      std::isfinite(algebraic) && std::isfinite(differential) &&
        std::max(algebraic, differential) <= initial_consistency_tolerance,
      ExcMessage(
        "The prepared initial state/derivative pair does not satisfy the native DAE consistency bounds."));
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::initialize_consistent_state(
    VectorType  &state,
    VectorType  &state_dot,
    const double t)
  {
    TimerOutput::Scope timer(computing_timer, "initialize_consistent_state");
    initial_state_diagnostics_ = InitialStateDiagnostics{};
    reinit_state(state);
    compute_initial_solution(state, t);
    correct_initial_traces(state, t);
    reconstruct_initial_derivative(state, state_dot, t);
    verify_initial_pair(state, state_dot, t);
  }

  template void
  BloodFlowSystem<1, 3>::compute_steady_network_initial_solution(VectorType &,
                                                                 const double);
  template void
  BloodFlowSystem<1, 3>::correct_initial_traces(VectorType &, const double);
  template void
  BloodFlowSystem<1, 3>::reconstruct_initial_derivative(const VectorType &,
                                                        VectorType &,
                                                        const double);
  template void
  BloodFlowSystem<1, 3>::verify_initial_pair(const VectorType &,
                                             const VectorType &,
                                             const double);
  template void
  BloodFlowSystem<1, 3>::initialize_consistent_state(VectorType &,
                                                     VectorType &,
                                                     const double);
} // namespace MetricFlowX
