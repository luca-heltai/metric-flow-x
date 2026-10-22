#include <deal.II/base/mpi.h>

#include <metric_flow_x/blood_flow_system.h>
#include <metric_flow_x/io/vtk_utils.h>

#include <filesystem>

#include "rapidcsv.h"
#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

void
test()
{
  const auto write = [](const std::string &name, const std::string &content) {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::ofstream file(name);
        file << content;
      }
    MPI_Barrier(MPI_COMM_WORLD);
  };
  const std::set<types::boundary_id> ids{1, 2};
  {
    BloodFlowSystem<1, 3> flow;
    flow.default_terminal_r1    = 2e7;
    flow.default_terminal_r2    = 3e7;
    flow.default_terminal_c     = 0;
    flow.default_terminal_p_out = 321;
    auto values                 = flow.read_terminal_parameters(ids);
    AssertThrow(values.size() == 2 && values.at(1).R1 == 2e7 &&
                  values.at(2).C == 0 && values.at(1).label == "boundary_1" &&
                  values.at(2).label == "boundary_2",
                ExcInternalError());
    flow.terminal_parameters_csv_file = "terminal.csv";
    write("terminal.csv", "boundary_id,R1,R2,C\n2,1e7,4e7,1e-8\n");
    values = flow.read_terminal_parameters(ids);
    AssertThrow(values.at(1).R1 == 2e7 && values.at(1).C == 0 &&
                  values.at(2).R1 == 1e7 && values.at(2).C == 1e-8 &&
                  values.at(2).P_out == 321,
                ExcInternalError());
    write("terminal.csv", "boundary_id,R1,R2,C\n1,7,8,0\n2,9,10,1e-8\n");
    values = flow.read_terminal_parameters(ids);
    AssertThrow(values.at(1).R1 == 7 && values.at(2).R2 == 10,
                ExcInternalError());
    // rapidcsv handles quoted fields, whitespace, UTF-8 BOM and CRLF.
    write("terminal.csv",
          "\xef\xbb\xbf\"boundary_id\",\"R1\",\"R2\",\"C\"\r\n"
          "\"2\", \"1e7\", \"4e7\", \"1e-8\"\r\n");
    values = flow.read_terminal_parameters(ids);
    AssertThrow(values.at(1).R1 == 2e7 && values.at(2).R1 == 1e7 &&
                  values.at(2).R2 == 4e7 && values.at(2).C == 1e-8 &&
                  values.at(2).P_out == 321,
                ExcInternalError());
    write("terminal.csv", "boundary_id,R1,R2,C\n");
    values = flow.read_terminal_parameters(ids);
    AssertThrow(values.at(1).R1 == 2e7 && values.at(2).R2 == 3e7,
                ExcInternalError());
    // Column labels, not their positions, select parameters. Other data are
    // ignored.
    write("terminal.csv",
          "C,note,boundary_id,R2,R1\n"
          "1e-8,\"external, measurement\nnot used by RCR\",2,4e7,1e7\n");
    values = flow.read_terminal_parameters(ids);
    AssertThrow(values.at(1).R1 == 2e7 && values.at(2).R1 == 1e7 &&
                  values.at(2).R2 == 4e7 && values.at(2).C == 1e-8,
                ExcInternalError());
    write("terminal.csv", "label,R2,boundary_id\nunused,4e7,2\n");
    values = flow.read_terminal_parameters(ids);
    AssertThrow(values.at(2).R1 == 2e7 && values.at(2).R2 == 4e7 &&
                  values.at(2).C == 0 && values.at(2).P_out == 321 &&
                  values.at(2).label == "unused",
                ExcInternalError());
    write("terminal.csv",
          "boundary_id,R1,R2,C,external_pressure\n"
          "1, ,4e7,,not_a_number\n2,1e7, ,1e-8,not_used\n");
    values = flow.read_terminal_parameters(ids);
    AssertThrow(values.at(1).R1 == 2e7 && values.at(1).R2 == 4e7 &&
                  values.at(1).C == 0 && values.at(2).R1 == 1e7 &&
                  values.at(2).R2 == 3e7 && values.at(2).C == 1e-8,
                ExcInternalError());
    write("terminal.csv",
          "boundary_id,label\n1, \n2,\"right, \"\"iliac\"\"\nterminal\"\n");
    values = flow.read_terminal_parameters(ids);
    AssertThrow(values.at(1).label == "boundary_1" &&
                  values.at(2).label == "right, \"iliac\"\nterminal" &&
                  values.at(2).R1 == 2e7,
                ExcInternalError());
    for (const auto &bad :
         {"boundary_id,R1,R2,C\n2,1,2,0\n2,1,2,0\n",
          "boundary_id,R1,R2,C\n0,1,2,0\n",
          "boundary_id,R1,R2,C\n999,1,2,0\n",
          "boundary_id,R1,R2,C\n1,-1,2,0\n",
          "boundary_id,R1,R2,C\n1,0,0,0\n",
          "boundary_id,R1,R2,C\n1,1,0,1\n",
          "boundary_id,R1,R2,C\n1,1,2,-1\n",
          "boundary_id,R1,R2,C\n1,nan,2,0\n",
          "boundary_id,R1,R2,C\n1,1,2,0,8\n",
          "boundary_id,R1,R2,C\n-1,1,2,0\n",
          "wrong_header\n1,1,2,0\n",
          "boundary_id,R1,R2,C\n1,1,2\n",
          "R1,R2,C\n1,2,0\n",
          "boundary_id,R1,R1\n1,2,3\n",
          "boundary_id,R1,note,note\n1,2,a,b\n",
          "boundary_id,R1,\n1,2,a\n",
          "boundary_id,pressure_pa\n1,12000\n",
          "boundary_id,R2,C\n1,0,1e-8\n",
          "boundary_id,R1,R2,C\n1,1e7suffix,2,0\n",
          "boundary_id,R1,R2,C\n1,1,2,0suffix\n",
          "boundary_id,R1,R2,C\n1suffix,1,2,0\n",
          "boundary_id,R1,R2,C\n18446744073709551616,1,2,0\n"})
      {
        write("terminal.csv", bad);
        bool rejected = false;
        try
          {
            flow.read_terminal_parameters(ids);
          }
        catch (const std::exception &)
          {
            rejected = true;
          }
        AssertThrow(rejected, ExcInternalError());
      }
    flow.terminal_parameters_csv_file = "missing-terminal.csv";
    bool rejected                     = false;
    try
      {
        flow.read_terminal_parameters(ids);
      }
    catch (const std::exception &)
      {
        rejected = true;
      }
    AssertThrow(rejected, ExcInternalError());
  }

  ParameterAcceptor::clear();
  const std::string mesh = std::string(METRIC_FLOW_X_SOURCE_DIR) +
                           "/notebooks/bifurcation_network.vtk";
  // The new mode must load geometry and boundary IDs without any RCR arrays.
  std::ifstream      original(mesh);
  std::ostringstream geometry;
  std::string        line;
  bool               skip = false;
  while (std::getline(original, line))
    {
      if (line.rfind("SCALARS ", 0) == 0)
        skip = line.rfind("SCALARS R1 ", 0) == 0 ||
               line.rfind("SCALARS R2 ", 0) == 0 ||
               line.rfind("SCALARS C ", 0) == 0 ||
               line.rfind("SCALARS P_out ", 0) == 0;
      if (!skip)
        geometry << line << '\n';
    }
  write("geometry.vtk", geometry.str());
  write("terminal.csv", "boundary_id,R1,R2,C\n2,1e7,4e7,1e-8\n");
  const auto absolute_mesh = std::filesystem::absolute("geometry.vtk").string();
  write(
    "terminal.prm",
    "subsection MetricFlowSystem<1, 3>\n"
    "set Vtk file path for mesh input = " +
      absolute_mesh +
      "\n"
      "set Terminal parameter source = Parameters\n"
      "set Terminal parameters CSV file = " +
      std::filesystem::absolute("terminal.csv").string() +
      "\n"
      "set Default terminal R1 = 2e7\nset Default terminal R2 = 3e7\n"
      "set Default terminal C = 0\nset Default terminal P_out = 321\n"
      "set Number of global refinement = 1\nset Verbosity (console depth) = 0\nend\n");
  {
    BloodFlowSystem<1, 3> flow;
    flow.initialize_params("terminal.prm");
    flow.setup();
    AssertThrow(flow.rcr_map.size() == 2 && flow.rcr_pc_dof.size() == 1,
                ExcInternalError());
    AssertThrow(flow.rcr_map.at(1).R1 == 2e7 && flow.rcr_map.at(2).R1 == 1e7 &&
                  flow.rcr_map.at(2).P_out == 321,
                ExcInternalError());
    AssertThrow(Utilities::MPI::sum(flow.rcr_dofs_owned.n_elements(),
                                    MPI_COMM_WORLD) == 1,
                ExcInternalError());
  }
  ParameterAcceptor::clear();
  // Match complete CSV input to a legacy VTK case, including nonzero-state
  // residuals and both Jacobian blocks. This requires no linear solve.
  std::string                               complete_csv;
  double                                    legacy_p_out = 0;
  std::map<types::global_dof_index, double> reference_residual;
  using Entry = std::pair<types::global_dof_index, types::global_dof_index>;
  std::map<Entry, double> reference_state, reference_derivative;
  for (unsigned int mode = 0; mode < 2; ++mode)
    {
      ParameterAcceptor::clear();
      BloodFlowSystem<1, 3> flow;
      flow.initialize_params(PRM_DIR "constant.prm");
      flow.vtk_file_path        = mode == 0 ? mesh : absolute_mesh;
      flow.n_global_refinements = 1;
      if (mode == 1)
        {
          flow.terminal_parameter_source    = "Parameters";
          flow.terminal_parameters_csv_file = "complete.csv";
          flow.default_terminal_p_out       = legacy_p_out;
          write("complete.csv", complete_csv);
        }
      flow.setup();
      if (mode == 0)
        {
          rapidcsv::Document csv;
          for (const auto &[index, name] :
               std::vector<std::pair<unsigned int, std::string>>{
                 {0, "boundary_id"}, {1, "R1"}, {2, "R2"}, {3, "C"}})
            csv.SetColumnName(index, name);
          legacy_p_out    = flow.rcr_map.begin()->second.P_out;
          std::size_t row = 0;
          for (const auto &[id, values] : flow.rcr_map)
            {
              AssertThrow(values.P_out == legacy_p_out, ExcInternalError());
              csv.SetCell(0, row, id);
              csv.SetCell(1, row, values.R1);
              csv.SetCell(2, row, values.R2);
              csv.SetCell(3, row, values.C);
              ++row;
            }
          std::ostringstream csv_text;
          csv.Save(csv_text);
          complete_csv = csv_text.str();
        }
      auto y        = flow.make_state();
      auto ydot     = flow.make_state();
      auto residual = flow.make_state();
      y             = 0.0;
      ydot          = 0.01;
      std::vector<types::global_dof_index> indices(flow.fe_->n_dofs_per_cell());
      for (const auto &cell : flow.dof_handler_.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(indices);
            for (unsigned int i = 0; i < indices.size(); ++i)
              y(indices[i]) =
                flow.fe_->system_to_component_index(i).first % 2 == 0 ?
                  1.1 * flow.vessel_map.at(cell->material_id()).a_d :
                  0.05;
          }
      for (const auto i : flow.rcr_dofs_owned)
        y(i) = 9000.0;
      y.compress(VectorOperation::insert);
      flow.assemble_residual(0.1, y, ydot, residual);
      flow.assemble_state_jacobian(0.1, y, ydot);
      flow.assemble_derivative_jacobian(0.1, y, ydot);
      for (const auto i : flow.locally_owned_dofs_)
        {
          if (mode == 0)
            reference_residual[i] = residual(i);
          else
            AssertThrow(std::abs(residual(i) - reference_residual.at(i)) <
                          1e-12 * std::max(1.0, std::abs(residual(i))),
                        ExcInternalError());
        }
      const auto compare = [&](const MatrixType        &matrix,
                               std::map<Entry, double> &reference) {
        for (const auto row : flow.locally_owned_dofs_)
          for (auto entry = matrix.begin(row); entry != matrix.end(row);
               ++entry)
            {
              const Entry key{row, entry->column()};
              if (mode == 0)
                reference[key] = entry->value();
              else
                AssertThrow(std::abs(entry->value() - reference.at(key)) <
                              1e-12 * std::max(1.0, std::abs(entry->value())),
                            ExcInternalError());
            }
      };
      compare(flow.state_jacobian_matrix(), reference_state);
      compare(flow.derivative_jacobian_matrix(), reference_derivative);
    }
  deallog.depth_file(10);
  deallog
    << "Terminal CSV defaults, overrides, validation and capacitor ownership: PASSED"
    << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
