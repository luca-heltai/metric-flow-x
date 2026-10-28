// SPDX-License-Identifier: MIT
// Pc is a native fifth FE component, zero on FE_Nothing cells. Exercise
// mixed active elements, ghost cells, empty ranks, and repeated setup.
#include <metric_flow_x/blood_flow_system.h>

#include "tests.h"

using namespace MetricFlowX;

void
test()
{
  for (const unsigned int degree : {1u, 2u})
    for (const unsigned int refinements : {0u, 2u})
      {
        ParameterAcceptor::clear();
        BloodFlowSystem<1, 3> flow;
        flow.initialize_params(PRM_DIR "constant.prm");
        flow.fe_degree            = degree;
        flow.n_global_refinements = refinements;
        flow.create_triangulation();
        for (const auto &mode : {"RCR", "Reflection", "Dirichlet"})
          for (const double capacitance : {0., 1e-9})
            {
              flow.outlet_type = mode;
              for (auto &[bid, rcr] : flow.rcr_map)
                rcr.C = capacitance;
              flow.setup_system();
              const bool capacitor =
                flow.outlet_type == "RCR" && capacitance > 0.;
              const auto expected = capacitor ? 1u : 0u;
              AssertDimension(flow.rcr_pc_dof.size(), expected);
              AssertDimension(flow.make_state().size(),
                              flow.dof_handler_.n_dofs());
              AssertDimension(flow.dof_handler_.n_dofs(),
                              flow.triangulation_.n_global_active_cells() *
                                  (2 * (degree + 1) + 4) +
                                expected);
              AssertThrow(flow.locally_owned_dofs_ ==
                            flow.dof_handler_.locally_owned_dofs(),
                          ExcMessage("State must use native DoF ownership."));
              AssertThrow(flow.component_dofs_[4] == flow.rcr_dofs_owned,
                          ExcMessage(
                            "Capacitors must be the fifth FE component."));
              auto state = flow.make_state();
              state      = 0.;
              for (const auto index : flow.component_dofs_[4])
                state(index) = 42.;
              state.compress(VectorOperation::insert);
              flow.update_ghosted_vectors(state);
              hp::FEValues<1, 3>  values(flow.fe_collection_,
                                        hp::QCollection<1>(QGauss<1>(2)),
                                        update_values);
              std::vector<double> pressure(2);
              for (const auto &cell : flow.dof_handler_.active_cell_iterators())
                if (!cell->is_artificial())
                  {
                    bool terminal = false;
                    for (const auto f : cell->face_indices())
                      terminal |= cell->face(f)->at_boundary() &&
                                  cell->face(f)->boundary_id() == 1;
                    const bool active = capacitor && terminal;
                    AssertDimension(cell->active_fe_index(), active ? 1 : 0);
                    AssertDimension(cell->get_fe().n_dofs_per_cell(),
                                    2 * (degree + 1) + 4 + (active ? 1 : 0));
                    values.reinit(cell);
                    values
                      .get_present_fe_values()[FEValuesExtractors::Scalar(4)]
                      .get_function_values(flow.y_relevant, pressure);
                    for (const auto value : pressure)
                      AssertThrow(value == (active ? 42. : 0.),
                                  ExcMessage(
                                    "Incorrect FE terminal pressure."));
                    if (active)
                      {
                        std::vector<types::global_dof_index> indices(
                          cell->get_fe().n_dofs_per_cell());
                        cell->get_dof_indices(indices);
                        const auto pc =
                          indices[cell->get_fe().component_to_system_index(4,
                                                                           0)];
                        AssertThrow(flow.rcr_pc_dof.at(1) == pc,
                                    ExcInternalError());
                        AssertThrow(flow.locally_owned_dofs_.is_element(pc) ==
                                      cell->is_locally_owned(),
                                    ExcMessage(
                                      "Pc must belong to its terminal cell."));
                      }
                  }
            }
      }
  // A single input vessel with two capacitive endpoints needs two DG0 cells.
  ParameterAcceptor::clear();
  BloodFlowSystem<1, 3> two_terminals;
  two_terminals.initialize_params(PRM_DIR "constant.prm");
  two_terminals.n_global_refinements = 0;
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::ifstream input(two_terminals.vtk_file_path);
      std::string   mesh((std::istreambuf_iterator<char>(input)),
                       std::istreambuf_iterator<char>());
      const auto    ids = mesh.find("\n0 1\n");
      AssertThrow(ids != std::string::npos, ExcInternalError());
      mesh.replace(ids, 5, "\n2 1\n");
      std::ofstream output("two_terminals.vtk");
      output << mesh;
    }
  MPI_Barrier(MPI_COMM_WORLD);
  two_terminals.vtk_file_path             = "two_terminals.vtk";
  two_terminals.terminal_parameter_source = "Parameters";
  two_terminals.default_terminal_c        = 1e-9;
  two_terminals.setup();
  AssertDimension(two_terminals.triangulation_.n_global_active_cells(), 2);
  AssertDimension(two_terminals.rcr_pc_dof.size(), 2);
  AssertThrow(two_terminals.rcr_pc_dof.at(1) != two_terminals.rcr_pc_dof.at(2),
              ExcMessage(
                "Distinct terminals must have independent FE pressures."));
  deallog.depth_file(10);
  deallog << "HP terminal field: PASSED" << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
