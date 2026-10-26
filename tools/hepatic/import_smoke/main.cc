// SPDX-License-Identifier: MIT
// Check generated line networks against the current native reader, including
// the geometry/array ordering assumed by BloodFlowSystem.
#include <metric_flow_x/io/vtk_utils.h>
#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkUnstructuredGridReader.h>

#include <cmath>
#include <iostream>
#include <string>

int
main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  try
    {
      AssertThrow(argc > 1, dealii::ExcMessage("Supply generated VTK paths"));
      for (int argument = 1; argument < argc; ++argument)
        {
          const std::string filename = argv[argument];
          auto raw = vtkSmartPointer<vtkUnstructuredGridReader>::New();
          raw->SetFileName(filename.c_str());
          raw->ReadAllScalarsOn();
          raw->Update();
          auto                         *grid = raw->GetOutput();
          MetricFlowX::VTKUtils::Reader reader(filename);
          dealii::Triangulation<1, 3>   tria;
          reader.read_vtk(tria, false);
          AssertThrow(tria.n_active_cells() == grid->GetNumberOfCells(),
                      dealii::ExcMessage("Cell count changed"));
          AssertThrow(tria.n_vertices() == grid->GetNumberOfPoints(),
                      dealii::ExcMessage("Point count changed"));
          unsigned int index = 0;
          for (const auto &cell : tria.active_cell_iterators())
            {
              for (unsigned int v = 0; v < 2; ++v)
                {
                  const auto expected = grid->GetCell(index)->GetPointId(v);
                  AssertThrow(cell->vertex_index(v) == expected,
                              dealii::ExcMessage("Cell ordering changed"));
                  double point[3];
                  grid->GetPoint(expected, point);
                  for (unsigned int d = 0; d < 3; ++d)
                    AssertThrow(std::abs(cell->vertex(v)[d] - point[d]) < 1e-12,
                                dealii::ExcMessage("Coordinates changed"));
                }
              ++index;
            }
          for (const std::string name : {"vessel_id",
                                         "a0",
                                         "a_d",
                                         "r_d",
                                         "E",
                                         "h_wall",
                                         "p_d",
                                         "p0",
                                         "L"})
            {
              dealii::Vector<double> values;
              reader.read_cell_data(name, values);
              auto *expected = grid->GetCellData()->GetArray(name.c_str());
              AssertThrow(expected && values.size() == tria.n_active_cells(),
                          dealii::ExcMessage(
                            "Missing or mis-sized cell array"));
              for (unsigned int i = 0; i < values.size(); ++i)
                AssertThrow(values[i] == expected->GetComponent(i, 0),
                            dealii::ExcMessage("Cell array changed"));
              if (name == "vessel_id")
                for (unsigned int i = 0; i < values.size(); ++i)
                  AssertThrow(values[i] == i,
                              dealii::ExcMessage(
                                "Invalid property-array index"));
            }
          for (const std::string name :
               {"boundary_id", "R1", "R2", "C", "P_out"})
            {
              dealii::Vector<double> values;
              reader.read_vertex_data(name, values);
              auto *expected = grid->GetPointData()->GetArray(name.c_str());
              AssertThrow(expected && values.size() == tria.n_vertices(),
                          dealii::ExcMessage(
                            "Missing or mis-sized point array"));
              for (unsigned int i = 0; i < values.size(); ++i)
                AssertThrow(values[i] == expected->GetComponent(i, 0),
                            dealii::ExcMessage("Point array changed"));
            }
          std::cout << filename << ": " << tria.n_active_cells()
                    << " cells, native import passed\n";
        }
    }
  catch (const std::exception &error)
    {
      std::cerr << error.what() << '\n';
      return 1;
    }
}
