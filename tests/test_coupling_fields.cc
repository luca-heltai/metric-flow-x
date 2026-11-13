// SPDX-License-Identifier: MIT
// Coupling field output: a history stores its layout once as a native
// checkpoint, and only selected fields at each time. A loaded object reads
// each field exactly, without parameter parsing or system setup, and rejects
// fields of a different mesh, missing fields and damaged directories.
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <metric_flow_x/blood_flow_system.h>

#include <filesystem>
#include <fstream>
#include <map>

#include "tests.h"

using namespace MetricFlowX;
using Flow = BloodFlowSystem<1, 3>;

void
make_mesh(Flow &flow, const double length)
{
  Triangulation<1, 3>      mesh;
  std::vector<Point<3>>    vertices{{0., 0., 0.}, {length, 0., 0.}};
  std::vector<CellData<1>> cells(1);
  cells[0].vertices[0] = 0;
  cells[0].vertices[1] = 1;
  mesh.create_triangulation(vertices, cells, SubCellData());
  for (const auto &cell : mesh.active_cell_iterators())
    for (const auto face : cell->face_indices())
      cell->face(face)->set_boundary_id(face);
  mesh.refine_global(3);
  GridTools::partition_triangulation_zorder(
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), mesh, false);
  flow.triangulation().create_triangulation(
    TriangulationDescription::Utilities::create_description_from_triangulation(
      mesh, MPI_COMM_WORLD));
}

void
setup(Flow &flow, const double length)
{
  flow.initialize_params("coupling-input.prm", "");
  make_mesh(flow, length);
  Flow::NetworkPhysics physics;
  auto                &v = physics.vessels[0];
  v.r_d = v.r_in = v.r_out = .003;
  v.a0 = v.a_d         = numbers::PI * v.r_d * v.r_d;
  v.E                  = 5.e5;
  v.h_wall             = .0003;
  v.L                  = length;
  v.p_d                = 100.;
  v.dynamic_viscosity  = .0035;
  physics.terminals[1] = {1.e7, 1.e8, 1.e-9, 100., "capacitive outlet"};
  flow.setup(physics);
}

template <class Operation>
void
expect_rejection(const Operation &operation, const std::string &what)
{
  bool rejected = false;
  try
    {
      operation();
    }
  catch (const std::exception &)
    {
      rejected = true;
    }
  AssertThrow(Utilities::MPI::sum(rejected ? 1u : 0u, MPI_COMM_WORLD) ==
                Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD),
              ExcMessage("Not rejected collectively: " + what));
}

void
test()
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  if (rank == 0)
    {
      for (const auto *path : {"coupling-history",
                               "relocated-coupling-history",
                               "shifted-fields",
                               "truncated-fields"})
        std::filesystem::remove_all(path);
      std::ofstream prm("coupling-input.prm");
      prm << "subsection MetricFlowSystem<1, 3>\n"
          << " set Verbosity (console depth) = 0\n"
          << " set Wall model = Square root\n"
          << " set Outlet boundary condition type = RCR\n"
          << " set Finite element degree = 2\n"
          << " set Number of global refinement = 0\nend\n";
    }
  MPI_Barrier(comm);

  const std::vector<double>                 times{.125, .25};
  const std::vector<std::string>            all{"area", "velocity", "pressure"};
  const std::map<std::string, unsigned int> component{{"area", 0},
                                                      {"velocity", 1},
                                                      {"pressure", 0}};
  // Expected field values per step and field on locally owned DoFs.
  std::vector<std::map<std::string, std::map<types::global_dof_index, double>>>
    expected(times.size());
  {
    ParameterAcceptor::clear();
    Flow flow(comm);
    setup(flow, .1);
    for (unsigned int step = 0; step < times.size(); ++step)
      {
        auto state = flow.make_state(), rate = flow.make_state(),
             pressure = flow.make_state();
        flow.initialize_consistent_state(state, rate, times[step]);
        // Distinct, nonuniform values in every component.
        for (const auto i : flow.locally_owned_dofs())
          state[i] = state[i] * (1. + 1.e-3 * step) + 1.e-9 * i;
        state.compress(VectorOperation::insert);
        flow.compute_pressure(state, rate, pressure, times[step]);
        for (const auto &field : all)
          for (const auto i : flow.component_dofs_[component.at(field)])
            expected[step][field][i] =
              field == "pressure" ? pressure[i] : state[i];
        if (step == 0)
          flow.save_checkpoint("coupling-history/layout",
                               times[step],
                               state,
                               rate);
        // The last time stores only the area.
        const std::vector<std::string> fields =
          step == 0 ? all : std::vector<std::string>{"area"};
        const auto path = "coupling-history/fields-" + std::to_string(step);
        flow.save_coupling_fields(path, times[step], state, rate, fields);
        expect_rejection(
          [&] {
            flow.save_coupling_fields(path, times[step], state, rate, fields);
          },
          "existing field directory");
      }
    auto state = flow.make_state(), rate = flow.make_state();
    expect_rejection(
      [&] {
        flow.save_coupling_fields("unknown-field", 0., state, rate, {"flow"});
      },
      "unknown field");
    expect_rejection(
      [&] {
        flow.save_coupling_fields(
          "duplicate-field", 0., state, rate, {"area", "area"});
      },
      "duplicate field");
  }
  {
    // Same topology and element association, different vertices.
    ParameterAcceptor::clear();
    Flow shifted(comm);
    setup(shifted, .2);
    auto state = shifted.make_state(), rate = shifted.make_state();
    shifted.initialize_consistent_state(state, rate, 0.);
    shifted.save_coupling_fields("shifted-fields", 0., state, rate, {"area"});
  }

  if (rank == 0)
    {
      std::filesystem::remove("coupling-input.prm");
      std::filesystem::rename("coupling-history", "relocated-coupling-history");
      std::filesystem::copy("relocated-coupling-history/fields-0",
                            "truncated-fields",
                            std::filesystem::copy_options::recursive);
      std::filesystem::resize_file("truncated-fields/area-0.bin", 0);
      // The fields carry no mesh: they are much smaller than the layout.
      std::uintmax_t layout = 0, fields = 0;
      for (const auto &entry : std::filesystem::recursive_directory_iterator(
             "relocated-coupling-history/layout"))
        if (entry.is_regular_file())
          layout += entry.file_size();
      for (const auto &entry : std::filesystem::directory_iterator(
             "relocated-coupling-history/fields-1"))
        fields += entry.file_size();
      AssertThrow(2 * fields < layout,
                  ExcMessage("Coupling fields are not smaller than a "
                             "checkpoint."));
    }
  MPI_Barrier(comm);

  ParameterAcceptor::clear();
  Flow reader(comm);
  reader.load_checkpoint("relocated-coupling-history/layout");
  auto values = reader.make_state();
  for (unsigned int step = 0; step < times.size(); ++step)
    for (const auto &[field, entries] : expected[step])
      {
        const auto path =
          "relocated-coupling-history/fields-" + std::to_string(step);
        if (step == 1 && field != "area")
          {
            expect_rejection(
              [&] { reader.load_coupling_field(path, field, values); },
              "field that was not saved");
            continue;
          }
        values = 1.;
        AssertThrow(reader.load_coupling_field(path, field, values) ==
                      times[step],
                    ExcMessage("Coupling field time changed."));
        for (const auto i : reader.locally_owned_dofs())
          {
            const auto match = entries.find(i);
            AssertThrow(
              values[i] == (match == entries.end() ? 0. : match->second),
              ExcMessage("Coupling field " + field + " is not exact."));
          }
      }

  values = 0.;
  for (const std::string path : {"missing-fields",
                                 "truncated-fields",
                                 "shifted-fields",
                                 "relocated-coupling-history/layout"})
    expect_rejection([&] { reader.load_coupling_field(path, "area", values); },
                     path);
  expect_rejection(
    [&] {
      reader.load_coupling_field("relocated-coupling-history/fields-0",
                                 "flow",
                                 values);
    },
    "unknown field name");
  AssertThrow(values.linfty_norm() == 0.,
              ExcMessage("Rejected read changed the vector."));

  deallog.depth_file(10);
  deallog << "MetricFlowX coupling fields: PASSED" << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  test();
}
