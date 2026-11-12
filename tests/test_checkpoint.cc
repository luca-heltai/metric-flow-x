// SPDX-License-Identifier: MIT
#if __has_include(<deal.II/distributed/solution_transfer.h>)
#  include <deal.II/distributed/solution_transfer.h>
#else
#  include <deal.II/numerics/solution_transfer.h>
#endif

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <metric_flow_x/blood_flow_system.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>

#include "tests.h"

using namespace MetricFlowX;
using Flow = BloodFlowSystem<1, 3>;
#if __has_include(<deal.II/distributed/solution_transfer.h>)
using AdditionalTransfer =
  dealii::parallel::distributed::SolutionTransfer<1, VectorType, 3>;
#else
using AdditionalTransfer = dealii::SolutionTransfer<1, VectorType, 3>;
#endif

void
make_mesh(Flow &flow)
{
  Triangulation<1, 3>      mesh;
  std::vector<Point<3>>    vertices{{0., 0., 0.}, {.1, 0., 0.}};
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
test()
{
  const auto comm  = MPI_COMM_WORLD;
  const auto rank  = Utilities::MPI::this_mpi_process(comm);
  const auto ranks = Utilities::MPI::n_mpi_processes(comm);
  if (rank == 0)
    {
      std::filesystem::remove_all("checkpoint-history");
      std::filesystem::remove_all("relocated-history");
      std::filesystem::remove_all("truncated-checkpoint");
      std::filesystem::remove_all("checkpoint.incomplete");
      std::filesystem::remove_all("truncated-checkpoint");
      std::filesystem::remove_all("checkpoint.incomplete");
      std::filesystem::create_directory("checkpoint-history");
      std::ofstream csv("checkpoint-inflow.csv");
      csv << "time_s,flow_m3_s\n0,1e-8\n1,3e-8\n";
      std::ofstream prm("checkpoint-input.prm");
      prm << "subsection Functions\n"
          << " set Inflow function = 7e-8*(1+t)\nend\n"
          << "subsection MetricFlowSystem<1, 3>\n"
          << " set Inflow CSV file = checkpoint-inflow.csv\n"
          << " set Verbosity (console depth) = 0\n"
          << " set Wall model = Square root\n"
          << " set Outlet boundary condition type = RCR\n"
          << " set Initial condition policy = native\n"
          << " set Initial consistency tolerance = 1e-5\n"
          << " set Finite element degree = 2\n"
          << " set Number of global refinement = 0\nend\n";
    }
  MPI_Barrier(comm);

  const std::vector<double>            times{.125, .25};
  std::vector<std::vector<double>>     states, rates, residuals;
  std::vector<types::global_dof_index> owned;
  std::map<std::string, std::vector<types::global_dof_index>> incidence;
  std::map<std::string, std::vector<Point<3>>>                geometry;
  std::map<std::string, types::material_id>                   materials;
  std::map<std::string, std::vector<types::boundary_id>>      boundaries;
  std::vector<unsigned int> component_sizes(5);

  {
    ParameterAcceptor::clear();
    Flow flow(comm);
    flow.initialize_params("checkpoint-input.prm", "");
    make_mesh(flow);
    Flow::NetworkPhysics physics;
    auto                &v = physics.vessels[0];
    v.r_d = v.r_in = v.r_out = .003;
    v.a0 = v.a_d         = numbers::PI * v.r_d * v.r_d;
    v.E                  = 5.e5;
    v.h_wall             = .0003;
    v.L                  = .1;
    v.p_d                = 100.;
    v.dynamic_viscosity  = .0035;
    physics.terminals[1] = {1.e7, 1.e8, 1.e-9, 100., "capacitive outlet"};
    flow.setup(physics);
    for (const auto i : flow.locally_owned_dofs_)
      owned.push_back(i);
    for (unsigned int c = 0; c < 5; ++c)
      component_sizes[c] =
        Utilities::MPI::sum(flow.component_dofs_[c].n_elements(), comm);
    AssertThrow(component_sizes[4] == 1,
                ExcMessage("Fixture must include a capacitor pressure DoF."));
    for (const auto &cell : flow.dof_handler_.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto id = cell->id().to_string();
          incidence[id].resize(cell->get_fe().n_dofs_per_cell());
          cell->get_dof_indices(incidence[id]);
          materials[id] = cell->material_id();
          for (const auto vertex : cell->vertex_indices())
            geometry[id].push_back(cell->vertex(vertex));
          for (const auto face : cell->face_indices())
            boundaries[id].push_back(cell->face(face)->boundary_id());
        }

    for (unsigned int step = 0; step < times.size(); ++step)
      {
        auto state = flow.make_state(), rate = flow.make_state(),
             residual = flow.make_state();
        flow.initialize_consistent_state(state, rate, times[step]);
        if (step == 1)
          {
            // This second pair tests serialization of every native component,
            // including algebraic derivatives that IDA need not use. It is
            // deliberately not a DAE-consistent initial pair; the separate
            // process-restart regression exercises actual time integration.
            for (unsigned int c = 0; c < 5; ++c)
              for (const auto i : flow.component_dofs_[c])
                {
                  state[i] =
                    state[i] + ((c == 0 || c == 2) ? 1.e-7 : .125 * (c + 1));
                  rate[i] = .25 * (c + 1) + .001 * i;
                }
            state.compress(VectorOperation::insert);
            rate.compress(VectorOperation::insert);
          }
        flow.assemble_residual(times[step], state, rate, residual);
        AssertThrow(state.linfty_norm() > 0. && rate.linfty_norm() > 0.,
                    ExcMessage("Checkpoint fixture is stationary or zero."));
        states.emplace_back();
        rates.emplace_back();
        residuals.emplace_back();
        for (const auto i : owned)
          {
            states.back().push_back(state[i]);
            rates.back().push_back(rate[i]);
            residuals.back().push_back(residual[i]);
          }
        const auto path = "checkpoint-history/step-" + std::to_string(step);
        // A coupled application can append its own native FE data to the
        // same mesh checkpoint. Values differ from both flow state vectors.
        auto extra = flow.make_state();
        for (const auto i : flow.locally_owned_dofs())
          extra[i] = .375 + .125 * i + step;
        extra.compress(VectorOperation::insert);
        VectorType relevant_extra(flow.locally_owned_dofs(),
                                  flow.locally_relevant_dofs(),
                                  comm);
        relevant_extra = extra;
        std::unique_ptr<AdditionalTransfer> extra_transfer;
        unsigned int                        prepare_calls = 0;
        flow.save_checkpoint(
          path, times[step], state, rate, [&](const std::string &staging) {
            ++prepare_calls;
            extra_transfer =
              std::make_unique<AdditionalTransfer>(flow.dof_handler());
            extra_transfer->prepare_for_serialization(relevant_extra);
            if (rank == 0)
              {
                std::ofstream metadata(std::filesystem::path(staging) /
                                       "coupled-metadata.txt");
                metadata << std::setprecision(17) << "coupled extension 1\n"
                         << times[step] << ' ' << 41 + step << '\n';
                metadata.close();
                AssertThrow(metadata.good(),
                            ExcMessage("Cannot write extension metadata."));
              }
          });
        AssertThrow(prepare_calls == 1,
                    ExcMessage("Checkpoint prepare hook was not called once."));
        std::ifstream manifest(std::filesystem::path(path) / "COMPLETE");
        std::string   header, payload;
        std::getline(manifest, header);
        std::uintmax_t bytes              = 0;
        bool           metadata_committed = false;
        while (manifest >> std::quoted(payload) >> bytes)
          metadata_committed |= payload == "coupled-metadata.txt" && bytes > 0;
        AssertThrow(metadata_committed,
                    ExcMessage(
                      "Extension metadata was not committed in manifest."));
        bool duplicate_rejected = false;
        try
          {
            flow.save_checkpoint(path, times[step], state, rate);
          }
        catch (const std::exception &)
          {
            duplicate_rejected = true;
          }
        AssertThrow(
          Utilities::MPI::sum(duplicate_rejected ? 1u : 0u, comm) == ranks,
          ExcMessage(
            "Existing checkpoint overwrite was not rejected collectively."));
      }
  }

  double changed = 0.;
  for (unsigned int i = 0; i < owned.size(); ++i)
    changed = std::max(changed, std::abs(states[0][i] - states[1][i]));
  AssertThrow(Utilities::MPI::max(changed, comm) > 0.,
              ExcMessage("History must contain different native states."));
  if (rank == 0)
    {
      std::filesystem::remove("checkpoint-input.prm");
      std::filesystem::remove("checkpoint-inflow.csv");
      std::filesystem::rename("checkpoint-history", "relocated-history");
    }
  MPI_Barrier(comm);

  for (unsigned int step = 0; step < times.size(); ++step)
    {
      ParameterAcceptor::clear();
      Flow                                restored(comm);
      VectorType                          extra;
      std::unique_ptr<AdditionalTransfer> extra_transfer;
      unsigned int                        restore_calls = 0;
      restored.load_checkpoint(
        "relocated-history/step-" + std::to_string(step),
        [&](const std::string &directory) {
          ++restore_calls;
          // Native state is available before the application restore hook.
          AssertThrow(restored.solution.linfty_norm() > 0.,
                      ExcMessage("Restore hook ran before native state load."));
          extra = restored.make_state();
          extra_transfer =
            std::make_unique<AdditionalTransfer>(restored.dof_handler());
          extra_transfer->deserialize(extra);
          std::ifstream metadata(std::filesystem::path(directory) /
                                 "coupled-metadata.txt");
          std::string   header;
          double        extension_time   = -1.;
          unsigned int  application_step = 0;
          std::getline(metadata, header);
          metadata >> extension_time >> application_step;
          AssertThrow(metadata && header == "coupled extension 1" &&
                        extension_time == times[step] &&
                        application_step == 41 + step,
                      ExcMessage("Relocated extension metadata changed."));
        });
      AssertThrow(restore_calls == 1,
                  ExcMessage("Checkpoint restore hook was not called once."));
      for (const auto i : restored.locally_owned_dofs())
        AssertThrow(extra[i] == .375 + .125 * i + step,
                    ExcMessage(
                      "Additional native FE vector did not round trip."));
      AssertThrow(restored.ida_parameters.initial_time == 0.,
                  ExcMessage(
                    "Checkpoint lost the original output-grid origin."));
      AssertThrow(restored.time == times[step],
                  ExcMessage("Checkpoint time changed."));
      AssertThrow(restored.triangulation_.n_global_active_cells() == 8,
                  ExcMessage("Checkpoint mesh changed."));
      AssertThrow(restored.fe_degree == 2 && restored.n_rcr_dofs == 1,
                  ExcMessage("Checkpoint discretization changed."));
      AssertThrow(restored.vessel_map.at(0).dynamic_viscosity == .0035 &&
                    restored.rcr_map.at(1).C == 1.e-9 &&
                    restored.rcr_map.at(1).label == "capacitive outlet",
                  ExcMessage(
                    "Checkpoint material or terminal metadata changed."));
      AssertThrow(std::abs(restored.prescribed_inflow(.75) - 2.5e-8) < 1.e-22,
                  ExcMessage("Checkpoint lost embedded inflow samples."));
      restored.inflow_function.set_time(.75);
      AssertThrow(std::abs(restored.inflow_function.value(Point<1>()) -
                           1.225e-7) < 1.e-21,
                  ExcMessage("Checkpoint lost parsed function configuration."));
      AssertThrow(restored.locally_owned_dofs_.n_elements() == owned.size(),
                  ExcMessage("Checkpoint partition changed."));
      for (unsigned int c = 0; c < 5; ++c)
        AssertThrow(
          Utilities::MPI::sum(restored.component_dofs_[c].n_elements(), comm) ==
            component_sizes[c],
          ExcMessage("Checkpoint component association changed."));
      for (const auto &cell : restored.dof_handler_.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const auto                           id = cell->id().to_string();
            std::vector<types::global_dof_index> indices(
              cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(indices);
            AssertThrow(indices == incidence.at(id) &&
                          cell->material_id() == materials.at(id),
                        ExcMessage("Checkpoint cell/DoF association changed."));
            for (const auto vertex : cell->vertex_indices())
              AssertThrow(cell->vertex(vertex) == geometry.at(id)[vertex],
                          ExcMessage("Checkpoint geometry changed."));
            for (const auto face : cell->face_indices())
              AssertThrow(
                cell->face(face)->boundary_id() == boundaries.at(id)[face],
                ExcMessage("Checkpoint boundary association changed."));
          }
      auto residual = restored.make_state();
      restored.assemble_residual(restored.time,
                                 restored.solution,
                                 restored.solution_dot,
                                 residual);
      for (unsigned int j = 0; j < owned.size(); ++j)
        {
          const auto i = owned[j];
          AssertThrow(restored.locally_owned_dofs_.is_element(i) &&
                        restored.solution[i] == states[step][j] &&
                        restored.solution_dot[i] == rates[step][j],
                      ExcMessage(
                        "Checkpoint state/rate round trip is not exact."));
          AssertThrow(std::abs(residual[i] - residuals[step][j]) <=
                        1.e-12 * std::max(1., std::abs(residuals[step][j])),
                      ExcMessage("Restored native equations changed."));
        }
    }

  if (rank == 0)
    {
      // Preserve the valid manifest while damaging one listed payload. All
      // ranks must reject before attempting native mesh/vector deserialization.
      std::filesystem::copy("relocated-history/step-0",
                            "truncated-checkpoint",
                            std::filesystem::copy_options::recursive);
      std::filesystem::resize_file("truncated-checkpoint/elements-0.bin", 0);
      std::filesystem::copy("relocated-history/step-0",
                            "checkpoint.incomplete",
                            std::filesystem::copy_options::recursive);
      std::ofstream manifest("relocated-history/step-1/COMPLETE");
      manifest << "unsupported checkpoint format\n";
    }
  MPI_Barrier(comm);
  for (const std::string path : {"missing-checkpoint",
                                 "relocated-history/step-1",
                                 "relocated-history/step-0",
                                 "checkpoint.incomplete",
                                 "truncated-checkpoint"})
    {
      ParameterAcceptor::clear();
      Flow flow(comm);
      bool rejected = false;
      try
        {
          // step-0 is otherwise valid, but requires the application restore
          // hook for its additional native data; a bare native load must fail.
          flow.load_checkpoint(path);
        }
      catch (const std::exception &)
        {
          rejected = true;
        }
      AssertThrow(
        Utilities::MPI::sum(rejected ? 1u : 0u, comm) == ranks,
        ExcMessage(
          "Invalid checkpoint or missing restore hook was not rejected collectively."));
    }
  deallog.depth_file(10);
  deallog << "MetricFlowX checkpoint history: PASSED" << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  test();
}
