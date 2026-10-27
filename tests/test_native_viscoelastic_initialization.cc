// SPDX-License-Identifier: MIT
// Check the full initial DAE residual, fixed differential seed, zero algebraic
// rates and conservative area equation for native viscoelastic initialization.
#include <metric_flow_x/blood_flow_system.h>

#include "tests.h"
using namespace MetricFlowX;

void
test()
{
  const auto   comm                        = MPI_COMM_WORLD;
  unsigned int nonlocal_structural_columns = 0;
  for (const std::string model : {"arterial", "venous", "mixed"})
    for (const unsigned int degree : {1u, 2u})
      for (const bool direct : {true, false})
        {
          if (Utilities::MPI::this_mpi_process(comm) == 0)
            {
              std::ofstream prm("native_viscoelastic.prm");
              prm << "subsection Functions\n set Inflow function = 1e-7\nend\n"
                  << "subsection MetricFlowSystem<1, 3>\n"
                  << " set Wall model = ADAVN " << model
                  << "\n set Viscoelastic wall = true\n"
                  << " set Initial condition policy = native\n"
                  << " set Use direct solver = " << (direct ? "true" : "false")
                  << "\n"
                  << " set Finite element degree = " << degree
                  << "\n set Number of global refinement = 0\n"
                  << " set Use componentwise absolute tolerances = "
                  << (degree == 2 ? "true" : "false") << "\n"
                  << " set Initial consistency tolerance = 1e-5\n"
                  << " set Vtk file path for mesh input = "
                  << METRIC_FLOW_X_SOURCE_DIR << "/parameters/"
                  << (model == "mixed" ? "wall_mixed_synthetic.vtk" :
                                         "wall_models_synthetic.vtk")
                  << "\nend\n";
            }
          MPI_Barrier(comm);
          ParameterAcceptor::clear();
          BloodFlowSystem<1, 3> flow(comm);
          flow.initialize_params("native_viscoelastic.prm");
          flow.set_external_pressure_provider([](const auto &point) {
            return 100. + 10. * point.time + 20. * point.point[0];
          });
          flow.n_global_refinements = degree == 2 ? 1 : 0;
          // Both fixtures have three cells. Native MUMPS needs a nonempty
          // partition on every rank; retain empty ranks in the iterative case.
          if (direct && Utilities::MPI::n_mpi_processes(comm) > 3)
            flow.n_global_refinements = 1;
          flow.setup();
          if (degree == 2)
            {
              // Sparse row graphs can contain structural-zero columns beyond
              // the FE ghost layer, depending on partitioner/deal.II version.
              // Force that case on this tiny fixture, independently of either.
              DynamicSparsityPattern pattern(flow.n_total_dofs,
                                             flow.n_total_dofs,
                                             flow.locally_owned_dofs_);
              for (const auto row : flow.locally_owned_dofs_)
                for (types::global_dof_index column = 0;
                     column < flow.n_total_dofs;
                     ++column)
                  {
                    pattern.add(row, column);
                    if (!flow.locally_relevant_dofs_.is_element(column))
                      ++nonlocal_structural_columns;
                  }
              for (auto *matrix : {&flow.jacobian_matrix,
                                   &flow.state_jacobian_matrix_,
                                   &flow.derivative_jacobian_matrix_,
                                   &flow.linear_system_matrix})
                matrix->reinit(flow.locally_owned_dofs_,
                               flow.locally_owned_dofs_,
                               pattern,
                               comm);
            }
          auto state = flow.make_state(), rate = flow.make_state(),
               seed = flow.make_state(), residual = flow.make_state();
          flow.compute_initial_solution(seed, .1);
          flow.initialize_consistent_state(state, rate, .1);
          const auto diagnostics = flow.initial_state_diagnostics_;
          AssertThrow(diagnostics.algebraic_defect <=
                          flow.initial_consistency_tolerance &&
                        diagnostics.differential_defect <=
                          flow.initial_consistency_tolerance,
                      ExcMessage(
                        "Native viscoelastic consistency bounds failed."));
          for (const auto i : flow.locally_owned_dofs_)
            if (flow.trace_dofs_owned.is_element(i))
              AssertThrow(rate(i) == 0.,
                          ExcMessage("Algebraic rates must stay zero."));
            else
              AssertThrow(
                state(i) == seed(i),
                ExcMessage(
                  "Native initialization changed the differential seed."));
          flow.assemble_residual(.1, state, rate, residual);
          double defect = 0.;
          // Independent area-conservation oracle: the viscosity operator has
          // no area rows, so M_A*Adot equals the hyperbolic area residual.
          flow.update_ghosted_vectors(state);
          auto hyperbolic = flow.make_state();
          hyperbolic      = 0.;
          flow.assemble_cell_residuals(.1, flow.y_relevant, hyperbolic);
          hyperbolic.compress(VectorOperation::add);
          std::vector<types::global_dof_index> indices(
            flow.fe_->n_dofs_per_cell());
          for (const auto &cell : flow.dof_handler_.active_cell_iterators())
            if (cell->is_locally_owned())
              {
                flow.get_system_dof_indices(cell, indices);
                Vector<double> local(indices.size()), mass_rate(indices.size());
                rate.extract_subvector_to(indices.begin(),
                                          indices.end(),
                                          local.begin());
                flow.per_cell_mass[cell->active_cell_index()].vmult(mass_rate,
                                                                    local);
                for (unsigned int i = 0; i < indices.size(); ++i)
                  if (flow.fe_->system_to_component_index(i).first == 0)
                    defect =
                      std::max(defect,
                               std::abs(mass_rate[i] - hyperbolic(indices[i])));
              }
          defect = Utilities::MPI::max(defect, comm);
          AssertThrow(defect < 1e-12,
                      ExcMessage("Initial area conservation failed."));
          // The zero-viscosity limit must recover the existing elastic
          // initializer, although the two methods solve different systems.
          for (auto &[id, vessel] : flow.vessel_map)
            vessel.wall.km = 0.;
          auto zero_state = flow.make_state(), zero_rate = flow.make_state();
          flow.initialize_consistent_state(zero_state, zero_rate, .1);
          flow.viscoelastic_wall               = false;
          flow.derivative_jacobian_is_current_ = false;
          auto elastic_state                   = flow.make_state(),
               elastic_rate                    = flow.make_state();
          flow.initialize_consistent_state(elastic_state, elastic_rate, .1);
          auto absolute = flow.make_state();
          if (flow.use_componentwise_tolerances)
            absolute = flow.make_absolute_tolerances();
          else
            absolute = flow.ida_parameters.absolute_tolerance;
          double       difference = 0.;
          const double horizon =
            flow.ida_parameters.initial_step_size > 0. ?
              flow.ida_parameters.initial_step_size :
              std::min(1., flow.ida_parameters.final_time - .1);
          for (const auto i : flow.locally_owned_dofs_)
            {
              const double weight =
                absolute(i) + flow.ida_parameters.relative_tolerance *
                                std::abs(elastic_state(i));
              difference =
                std::max(difference,
                         std::abs(zero_state(i) - elastic_state(i)) / weight);
              difference =
                std::max(difference,
                         horizon * std::abs(zero_rate(i) - elastic_rate(i)) /
                           weight);
            }
          difference = Utilities::MPI::max(difference, comm);
          AssertThrow(
            difference < 1.e-3,
            ExcMessage(
              "Zero-viscosity native initialization disagrees with the elastic limit."));
          std::cout << model << " degree=" << degree << " direct=" << direct
                    << " defects=" << diagnostics.algebraic_defect << ","
                    << diagnostics.differential_defect << std::endl;
        }
  if (Utilities::MPI::n_mpi_processes(comm) > 1)
    AssertThrow(
      Utilities::MPI::sum(nonlocal_structural_columns, comm) > 0,
      ExcMessage(
        "The MPI regression must exercise matrix columns beyond FE ghosts."));
  deallog.depth_file(10);
  deallog << "Native viscoelastic initialization: PASSED" << std::endl;
}
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  mpi_initlog();
  test();
}
