// SPDX-License-Identifier: MIT
// Two geometrically disconnected vessels coupled by a peripheral resistor.
// The venous port is at face zero, so inward flow and its characteristic have
// signs opposite to the arterial port. Exercise distant partitions and empty
// ranks, check independent port balances and both implicit Jacobian blocks.
#include <metric_flow_x/blood_flow_system.h>

#include "tests.h"
using namespace MetricFlowX;

void
test()
{
  for (const bool venous_capacitor : {false, true})
    {
      const auto comm = MPI_COMM_WORLD;
      if (Utilities::MPI::this_mpi_process(comm) == 0)
        {
          std::ofstream mesh("terminal_connections.vtk");
          mesh << "# vtk DataFile Version 3.0\nTwo components\nASCII\n"
               << "DATASET UNSTRUCTURED_GRID\nPOINTS 4 double\n"
               << "0 0 0\n.1 0 0\n0 1 0\n.1 1 0\n"
               << "CELLS 2 6\n2 0 1\n2 2 3\nCELL_TYPES 2\n3\n3\nCELL_DATA 2\n";
          for (const auto &[name, values] : std::map<std::string, std::string>{
                 {"vessel_id", "0 1"},
                 {"a0", ".0001 .0002"},
                 {"a_d", ".0001 .0002"},
                 {"E", "500000 500000"},
                 {"viscosity", ".001 .004"},
                 {"h_wall", ".001 .001"},
                 {"p_d", "10000 666.666"},
                 {"p0", "0 0"},
                 {"L", ".1 .1"},
                 {"r_d", ".005641895835 .007978845608"},
                 {"adavn_wall_type", "1 2"},
                 {"adavn_p_ref", "10000 666.666"},
                 {"adavn_E_e", "340000 0"},
                 {"adavn_E_c", "5000000 0"},
                 {"adavn_epsilon_0", ".25 0"},
                 {"adavn_epsilon_r", ".05 0"},
                 {"adavn_venous_K", "0 1000"},
                 {"adavn_venous_m", "0 10"},
                 {"adavn_venous_n", "0 -1.5"}})
            mesh << "SCALARS " << name << " double 1\nLOOKUP_TABLE default\n"
                 << values << '\n';
          mesh << "POINT_DATA 4\n";
          for (const auto &[name, values] : std::map<std::string, std::string>{
                 {"boundary_id", "0 1 2 3"},
                 {"R1",
                  venous_capacitor ? "0 1000000 2000000 0" : "0 1000000 0 0"},
                 {"R2", "0 100000000 100000000 10000000"},
                 {"C", venous_capacitor ? "0 1e-9 2e-9 0" : "0 1e-9 0 0"},
                 {"P_out", "0 0 0 666.666"}})
            mesh << "SCALARS " << name << " double 1\nLOOKUP_TABLE default\n"
                 << values << '\n';
          std::ofstream links("terminal_connections.csv");
          links << "boundary0,boundary1,resistance\n1,2,1e9\n";
          std::ofstream prm("terminal_connections.prm");
          prm
            << "subsection Functions\n set Inflow function = 1e-6\nend\n"
            << "subsection MetricFlowSystem<1, 3>\n"
            << " set Vtk file path for mesh input = terminal_connections.vtk\n"
            << " set Terminal connections CSV file = terminal_connections.csv\n"
            << " set Wall model = ADAVN mixed\n"
            << " set Outlet boundary condition type = RCR\n"
            << " set Number of global refinement = 0\n"
            << " set Finite element degree = 1\n"
            << " set Initial consistency tolerance = 1e-5\nend\n";
        }
      MPI_Barrier(comm);
      ParameterAcceptor::clear();
      BloodFlowSystem<1, 3> flow(comm);
      flow.initialize_params("terminal_connections.prm");
      flow.setup();
      AssertThrow(flow.vessel_map.at(0).dynamic_viscosity == .001 &&
                    flow.vessel_map.at(1).dynamic_viscosity == .004,
                  ExcMessage("Per-vessel viscosity was not preserved."));
      auto y = flow.make_state(), rate = flow.make_state();
      flow.compute_initial_solution(y, 0.);
      AssertThrow(flow.has_admissible_wall_states(y),
                  ExcMessage("Valid wall state rejected."));
      auto invalid_trial = flow.make_state();
      invalid_trial      = y;
      for (const auto &cell : flow.dof_handler_.active_cell_iterators())
        if (cell->is_locally_owned() && cell->material_id() == 0)
          {
            std::vector<types::global_dof_index> indices(
              flow.fe_->n_dofs_per_cell());
            cell->get_dof_indices(indices);
            for (unsigned int k = 0; k < indices.size(); ++k)
              if (flow.fe_->system_to_component_index(k).first == 0)
                invalid_trial(indices[k]) = 100 * flow.vessel_map.at(0).a_d;
          }
      invalid_trial.compress(VectorOperation::insert);
      AssertThrow(
        !flow.has_admissible_wall_states(invalid_trial),
        ExcMessage(
          "Non-hyperbolic trial must be rejected collectively, including empty ranks."));
      flow.initialize_trace_unknowns(y, 0.);
      for (const auto &cell : flow.dof_handler_.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            std::vector<types::global_dof_index> indices(
              flow.fe_->n_dofs_per_cell());
            cell->get_dof_indices(indices);
            for (unsigned int k = 0; k < indices.size(); ++k)
              {
                const auto component =
                  flow.fe_->system_to_component_index(k).first;
                y(indices[k])    = component % 2 == 0 ?
                                     flow.vessel_map.at(cell->material_id()).a_d *
                                    (1.01 + .001 * k) :
                                     .02 + .001 * k;
                rate(indices[k]) = component % 2 == 0 ? 1e-7 : .003;
              }
          }
      const auto pc = flow.rcr_pc_dof.at(1);
      if (flow.locally_owned_dofs_.is_element(pc))
        {
          y(pc)    = 9000.;
          rate(pc) = 100.;
        }
      if (venous_capacitor)
        {
          const auto venous_pc = flow.rcr_pc_dof.at(2);
          if (flow.locally_owned_dofs_.is_element(venous_pc))
            {
              y(venous_pc)    = 7000.;
              rate(venous_pc) = 200.;
            }
        }
      y.compress(VectorOperation::insert);
      rate.compress(VectorOperation::insert);
      auto residual = flow.make_state();
      flow.assemble_residual(.1, y, rate, residual);
      flow.update_ghosted_vectors(y);
      double a1 = 0., u1 = 0., a2 = 0., u2 = 0.;
      for (const auto &face : flow.boundary_faces)
        {
          const auto bid = face.cell->face(face.face_no)->boundary_id();
          if (bid == 1)
            {
              a1 = y(face.a_hat_dof);
              u1 = y(face.u_hat_dof);
            }
          if (bid == 2)
            {
              a2 = y(face.a_hat_dof);
              u2 = y(face.u_hat_dof);
            }
        }
      a1              = Utilities::MPI::sum(a1, comm);
      u1              = Utilities::MPI::sum(u1, comm);
      a2              = Utilities::MPI::sum(a2, comm);
      u2              = Utilities::MPI::sum(u2, comm);
      const auto  &v2 = flow.terminal_ports.at(2);
      const double p2 =
        flow.compute_pressure_value(a2, v2.vessel, v2.reference_area);
      const double downstream_pressure = venous_capacitor ? 7000. : p2;
      if (flow.locally_owned_dofs_.is_element(pc))
        AssertThrow(std::abs(residual(pc) -
                             (1e-9 * 100. - a1 * u1 +
                              (9000. - downstream_pressure) / 1e9)) < 1e-16,
                    ExcMessage("Incorrect arterial compartment balance."));
      if (flow.locally_owned_dofs_.is_element(v2.area))
        {
          const double boundary = venous_capacitor ?
                                    p2 + 2e6 * a2 * u2 - 7000. :
                                    p2 + 1e9 * a2 * u2 - 9000.;
          AssertThrow(std::abs(residual(v2.area) + boundary) < 1e-9,
                      ExcMessage("Incorrect inward venous port balance."));
        }
      if (venous_capacitor &&
          flow.locally_owned_dofs_.is_element(flow.rcr_pc_dof.at(2)))
        AssertThrow(std::abs(residual(flow.rcr_pc_dof.at(2)) -
                             (2e-9 * 200. + a2 * u2 + (7000. - 9000.) / 1e9)) <
                      1e-16,
                    ExcMessage("Incorrect venous compartment balance."));

      flow.assemble_state_jacobian(.1, y, rate);
      flow.assemble_derivative_jacobian(.1, y, rate);
      for (const bool derivative : {false, true})
        {
          const auto &matrix = derivative ? flow.derivative_jacobian_matrix() :
                                            flow.state_jacobian_matrix();
          auto        plus = flow.make_state(), minus = flow.make_state();
          auto        fp = flow.make_state(), fm = flow.make_state();
          auto        unit = flow.make_state(), column = flow.make_state();
          double      worst = 0.;
          for (types::global_dof_index j = 0; j < y.size(); ++j)
            {
              const double local = flow.locally_owned_dofs_.is_element(j) ?
                                     (derivative ? rate(j) : y(j)) :
                                     0.;
              const double value = Utilities::MPI::sum(local, comm);
              const double h     = 1e-5 * std::max(std::abs(value), 1e-6);
              plus               = derivative ? rate : y;
              minus              = plus;
              if (flow.locally_owned_dofs_.is_element(j))
                {
                  plus(j)  = value + h;
                  minus(j) = value - h;
                }
              plus.compress(VectorOperation::insert);
              minus.compress(VectorOperation::insert);
              flow.assemble_residual(.1,
                                     derivative ? y : plus,
                                     derivative ? plus : rate,
                                     fp);
              flow.assemble_residual(.1,
                                     derivative ? y : minus,
                                     derivative ? minus : rate,
                                     fm);
              unit = 0.;
              if (flow.locally_owned_dofs_.is_element(j))
                unit(j) = 1.;
              unit.compress(VectorOperation::insert);
              matrix.vmult(column, unit);
              for (const auto i : flow.locally_owned_dofs_)
                {
                  const double fd = (fp(i) - fm(i)) / (2 * h);
                  worst           = std::max(worst,
                                   std::abs(fd - column(i)) /
                                     (1. + std::abs(fd) + std::abs(column(i))));
                }
            }
          AssertThrow(Utilities::MPI::max(worst, comm) < 2e-5,
                      ExcMessage(
                        "Terminal-network implicit Jacobian mismatch."));
        }
      flow.initialize_consistent_state(y, rate, 0.);
      AssertThrow(flow.initial_state_diagnostics_.algebraic_defect <= 1e-5 &&
                    flow.initial_state_diagnostics_.differential_defect <= 1e-5,
                  ExcMessage("Coupled native initialization failed."));
      flow.initial_state_source = "steady network";
      flow.compute_initial_solution(y, 0.);
      const double venous_resistance =
        8 * numbers::PI * .004 * .1 / (.0002 * .0002);
      const double venous_proximal = venous_capacitor ? 2e6 : 0.;
      const double expected_pc =
        666.666 + (1e7 + venous_resistance + venous_proximal + 1e9) * 1e-6;
      if (flow.locally_owned_dofs_.is_element(pc))
        AssertThrow(std::abs(y(pc) - expected_pc) < 1e-7,
                    ExcMessage(
                      "Coupled graph seed has incorrect capacitor pressure."));
      for (const auto &face : flow.boundary_faces)
        {
          const auto   bid  = face.cell->face(face.face_no)->boundary_id();
          const double area = y(face.a_hat_dof), velocity = y(face.u_hat_dof);
          AssertThrow(
            std::abs(area * velocity - 1e-6) < 1e-16,
            ExcMessage(
              "Graph seed did not retain the analytical through-flow."));
          if (bid == 1 || bid == 2)
            {
              const auto  &port = flow.terminal_ports.at(bid);
              const double pressure =
                flow.compute_pressure_value(area,
                                            port.vessel,
                                            port.reference_area);
              const double expected =
                bid == 1 ? expected_pc + 1. :
                           expected_pc - 1000. - venous_proximal * 1e-6;
              AssertThrow(
                std::abs(pressure - expected) < 1e-7,
                ExcMessage(
                  "ADAVN inverse did not recover the analytical graph pressure."));
            }
        }
      flow.check_mass_conservation(y, 0.);
      AssertThrow(
        flow.mass_flow_scale < 1e-16,
        ExcMessage(
          "Boundary orientation corrupted the through-flow mass diagnostic."));
      flow.initialize_consistent_state(y, rate, 0.);
      AssertThrow(flow.initial_state_diagnostics_.algebraic_defect <= 1e-5 &&
                    flow.initial_state_diagnostics_.differential_defect <= 1e-5,
                  ExcMessage(
                    "Native initialization from the coupled graph failed."));
      flow.vessel_probe_ids = {0};
      flow.output_directory = "selected_probes";
      flow.open_csv_files();
      AssertThrow(Utilities::MPI::sum(static_cast<unsigned int>(
                                        flow.csv_vessel.is_open()),
                                      comm) == 1,
                  ExcMessage("Vessel probe selection opened extra files."));
      flow.close_csv_files();
      deallog.depth_file(10);
      deallog << "Terminal connections: PASSED" << std::endl;
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
