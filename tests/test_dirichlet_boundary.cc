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
  flow.outlet_type = "Dirichlet";
  flow.exact_solution.update_expression(
    "2e-4 + 1e-5*x + 2e-5*t; 0.1 + 0.2*x - 0.03*t; 0; 0");
  flow.create_triangulation();
  // Even a mesh containing capacitive terminals must carry no Pc unknowns
  // when both physical endpoints prescribe the manufactured state.
  for (auto &[bid, rcr] : flow.rcr_map)
    rcr.C = 1.0;
  flow.setup_system();
  flow.build_per_cell_mass_inv();
  AssertThrow(flow.rcr_pc_dof.empty(), ExcInternalError());
  AssertDimension(flow.n_total_dofs, flow.dof_handler_.n_dofs());

  auto y = flow.make_state();
  flow.compute_initial_solution(y, 0.0);
  auto ydot = flow.make_state();
  ydot      = 2.0;

  struct Boundary
  {
    types::global_dof_index area, velocity;
    double                  x;
  };
  std::vector<Boundary> boundaries;
  for (const auto &cell : flow.dof_handler_.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const unsigned int face : cell->face_indices())
        if (cell->face(face)->at_boundary() &&
            !flow.is_junction_face(cell->id(), face))
          {
            const auto &trace =
              flow.face_dof_map.at(flow.canonical_face_key(cell, face));
            boundaries.push_back(
              {trace.a_hat_dof, trace.u_hat_dof, cell->vertex(face)[0]});
            y(trace.a_hat_dof) = 3.0e-4;
            y(trace.u_hat_dof) = 0.4;
          }
  y.compress(VectorOperation::insert);
  AssertThrow(Utilities::MPI::sum(boundaries.size(), flow.mpi_communicator()) ==
                2,
              ExcMessage("Check both inlet and outlet"));

  for (const double time : {0.2, 0.7})
    {
      auto residual = flow.make_state();
      flow.assemble_residual(time, y, ydot, residual);
      flow.assemble_state_jacobian(time, y, ydot);
      flow.assemble_derivative_jacobian(time, y, ydot);
      for (const auto &boundary : boundaries)
        {
          // Public residual is M*ydot - R, hence prescribed minus candidate.
          const double area     = 2.0e-4 + 1.0e-5 * boundary.x + 2.0e-5 * time;
          const double velocity = 0.1 + 0.2 * boundary.x - 0.03 * time;
          AssertThrow(std::abs(residual(boundary.area) - (area - 3.0e-4)) <
                        1.0e-14,
                      ExcMessage("Time-dependent area trace at endpoint"));
          AssertThrow(std::abs(residual(boundary.velocity) - (velocity - 0.4)) <
                        1.0e-14,
                      ExcMessage("Time-dependent velocity trace at endpoint"));
          for (const auto row : {boundary.area, boundary.velocity})
            {
              const auto &state_matrix = flow.state_jacobian_matrix();
              AssertThrow(std::abs(state_matrix.el(row, row) + 1.0) < 1.0e-14,
                          ExcInternalError());
              for (auto it = state_matrix.begin(row);
                   it != state_matrix.end(row);
                   ++it)
                AssertThrow(
                  std::abs(it->value() - (it->column() == row ? -1.0 : 0.0)) <
                    1.0e-14,
                  ExcMessage("Prescribed trace has no cell coupling"));
              const auto &derivative = flow.derivative_jacobian_matrix();
              for (auto it = derivative.begin(row); it != derivative.end(row);
                   ++it)
                AssertThrow(std::abs(it->value()) < 1.0e-14,
                            ExcMessage("Prescribed trace is algebraic"));
            }
        }
    }
  deallog.depth_file(10);
  deallog << "Dirichlet boundary traces: PASSED" << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
