#include <deal.II/base/mpi.h>

#include <metric_flow_x/blood_flow_system.h>

#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

void
test()
{
  BloodFlowSystem<1, 3> flow;
  flow.initialize_params(PRM_DIR "constant.prm");
  flow.set_numerical_flux(BloodFlowSystem<1, 3>::NumericalFluxType::HLL_HDG);
  flow.create_triangulation();
  for (auto &[bid, rcr] : flow.rcr_map)
    rcr.C = 0.0;
  flow.setup_system();
  flow.build_per_cell_mass_inv();
  AssertThrow(flow.rcr_pc_dof.empty(), ExcInternalError());

  auto y = flow.make_state();
  flow.initialize_state(y, 0.0);
  types::global_dof_index area_row     = numbers::invalid_dof_index;
  types::global_dof_index velocity_row = numbers::invalid_dof_index;
  for (const auto &cell : flow.dof_handler_.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const unsigned int face : cell->face_indices())
        if (cell->face(face)->at_boundary() &&
            cell->face(face)->boundary_id() == 1)
          {
            const auto &trace =
              flow.face_dof_map.at(flow.canonical_face_key(cell, face));
            area_row     = trace.a_hat_dof;
            velocity_row = trace.u_hat_dof;
          }
  area_row     = Utilities::MPI::min(area_row, flow.mpi_communicator());
  velocity_row = Utilities::MPI::min(velocity_row, flow.mpi_communicator());
  AssertThrow(area_row != numbers::invalid_dof_index, ExcInternalError());

  const double area     = 1.1 * flow.vessel_map.at(0).a_d;
  const double velocity = 0.12;
  if (flow.locally_owned_dofs_.is_element(area_row))
    {
      y(area_row)     = area;
      y(velocity_row) = velocity;
    }
  for (const auto &cell : flow.dof_handler_.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const unsigned int face : cell->face_indices())
        if (cell->face(face)->at_boundary() &&
            cell->face(face)->boundary_id() == 1)
          {
            Point<1> unit_point;
            unit_point[0] = static_cast<double>(face);
            std::vector<types::global_dof_index> dofs(
              cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(dofs);
            for (unsigned int i = 0; i < dofs.size(); ++i)
              {
                if (cell->get_fe().shape_value_component(i, unit_point, 0) !=
                    0.0)
                  y(dofs[i]) = 0.95 * flow.vessel_map.at(0).a_d;
                if (cell->get_fe().shape_value_component(i, unit_point, 1) !=
                    0.0)
                  y(dofs[i]) = -0.03;
              }
          }
  y.compress(VectorOperation::insert);
  flow.update_ghosted_vectors(y);

  double numerical_flux = 0.0;
  double flux_direction = 0.0;
  auto   direction      = flow.make_state();
  direction             = 0.0;
  if (flow.locally_owned_dofs_.is_element(area_row))
    {
      direction(area_row) += area;
      direction(velocity_row) += 0.3;
    }
  for (const auto &face : flow.boundary_faces)
    if (face.cell->face(face.face_no)->boundary_id() == 1)
      {
        numerical_flux +=
          flow.numerical_area_flux(face.cell, face.face_no, flow.y_relevant);
        flux_direction += flow.numerical_area_flux_derivative(
          face.cell, face.face_no, flow.y_relevant, area, 0.3, area, 0.3);

        Point<1> unit_point;
        unit_point[0] = static_cast<double>(face.face_no);
        std::vector<types::global_dof_index> dofs(
          face.cell->get_fe().n_dofs_per_cell());
        face.cell->get_dof_indices(dofs);
        for (unsigned int i = 0; i < dofs.size(); ++i)
          direction(dofs[i]) +=
            area * face.cell->get_fe().shape_value_component(i, unit_point, 0) +
            0.3 * face.cell->get_fe().shape_value_component(i, unit_point, 1);
      }
  numerical_flux = Utilities::MPI::sum(numerical_flux, flow.mpi_communicator());
  flux_direction = Utilities::MPI::sum(flux_direction, flow.mpi_communicator());
  direction.compress(VectorOperation::add);

  // A removed capacitor leaves two series resistors. Include the legacy
  // R1=0 case in the same analytic check.
  for (const auto &[r1, r2] : {std::pair{2.0e7, 3.0e7}, std::pair{0.0, 3.0e7}})
    {
      auto &outlet  = flow.rcr_map.at(1);
      outlet.R1     = r1;
      outlet.R2     = r2;
      outlet.P_out  = 123.0;
      auto residual = flow.make_state();
      residual      = 0.0;
      flow.update_ghosted_vectors(y);
      flow.assemble_trace_boundary_equations(0.0, flow.y_relevant, residual);
      residual.compress(VectorOperation::add);
      const double expected =
        flow.pressure(area, 0) - (r1 + r2) * numerical_flux - 123.0;
      if (flow.locally_owned_dofs_.is_element(area_row))
        AssertThrow(std::abs(residual(area_row) - expected) < 1.0e-9,
                    ExcMessage("C=0 must retain both series resistances"));

      flow.jacobian_matrix = 0.0;
      flow.assemble_jacobian_trace_boundary_block(0.0, flow.y_relevant);
      flow.jacobian_matrix.compress(VectorOperation::add);
      auto product = flow.make_state();
      flow.jacobian_matrix.vmult(product, direction);
      const double derivative =
        flow.pressure_derivative(area, 0) * area - (r1 + r2) * flux_direction;
      if (flow.locally_owned_dofs_.is_element(area_row))
        AssertThrow(std::abs(product(area_row) - derivative) <
                      1.0e-10 * std::max(1.0, std::abs(derivative)),
                    ExcInternalError());

      constexpr double step = 1.0e-5;
      auto             yp   = y;
      auto             ym   = y;
      yp.add(step, direction);
      ym.add(-step, direction);
      auto fp = flow.make_state();
      auto fm = flow.make_state();
      fp      = 0.0;
      fm      = 0.0;
      flow.update_ghosted_vectors(yp);
      flow.assemble_trace_boundary_equations(0.0, flow.y_relevant, fp);
      fp.compress(VectorOperation::add);
      flow.update_ghosted_vectors(ym);
      flow.assemble_trace_boundary_equations(0.0, flow.y_relevant, fm);
      fm.compress(VectorOperation::add);
      if (flow.locally_owned_dofs_.is_element(area_row))
        AssertThrow(std::abs((fp(area_row) - fm(area_row)) / (2.0 * step) -
                             product(area_row)) <
                      1.0e-7 * std::max(1.0, std::abs(derivative)),
                    ExcInternalError());
    }
  deallog.depth_file(10);
  deallog << "Zero-capacitance RCR: PASSED" << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
