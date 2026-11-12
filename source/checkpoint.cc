// SPDX-License-Identifier: LGPL-2.1-or-later
#include <metric_flow_x/blood_flow_system.h>

#if __has_include(<deal.II/distributed/solution_transfer.h>)
#  include <deal.II/distributed/solution_transfer.h>
#else
#  include <deal.II/numerics/solution_transfer.h>
#endif

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <filesystem>
#include <iomanip>
#include <sstream>

namespace MetricFlowX
{
  using dealii::ExcMessage;
  namespace
  {
    // File errors must reach every rank before the next collective operation.
    template <class Operation>
    void
    collective_file_operation(MPI_Comm comm, const Operation &operation)
    {
      std::string error;
      try
        {
          operation();
        }
      catch (const std::exception &exception)
        {
          error = exception.what();
        }
      const auto errors = dealii::Utilities::MPI::all_gather(comm, error);
      for (const auto &message : errors)
        AssertThrow(message.empty(),
                    dealii::ExcMessage("Checkpoint: " + message));
    }

    // Only the application envelope is new. Mesh and finite-element vectors
    // use deal.II's native serialization facilities below.
    template <int dim, int spacedim>
    struct CheckpointData
    {
      std::string magic          = "MetricFlowX checkpoint 1";
      std::string dealii_version = DEAL_II_PACKAGE_VERSION;
#ifdef METRIC_FLOW_X_WITH_PETSC
      std::string backend = "PETSc";
#else
      std::string backend = "Trilinos";
#endif
      unsigned int ranks           = 0;
      double       time            = 0.;
      bool         additional_data = false;
      typename BloodFlowSystem<dim, spacedim>::NetworkPhysics physics;
      std::map<unsigned int, std::set<types::boundary_id>>    vessel_terminals;
      InflowSeries                                            inflow;
      std::string                                             parameters;

      template <class Archive>
      void
      serialize(Archive &archive, const unsigned int)
      {
        archive & magic & dealii_version & backend & ranks & time &
          additional_data & physics & vessel_terminals & inflow & parameters;
      }
    };

    template <int dim, int spacedim>
    std::map<std::string, unsigned int>
    cell_elements(const dealii::DoFHandler<dim, spacedim> &dofs)
    {
      std::map<std::string, unsigned int> result;
      for (const auto &cell : dofs.active_cell_iterators())
        if (cell->is_locally_owned())
          result.emplace(cell->id().to_string(), cell->active_fe_index());
      return result;
    }

#if __has_include(<deal.II/distributed/solution_transfer.h>)
    template <int dim, int spacedim>
    using NativeTransfer = dealii::parallel::distributed::
      SolutionTransfer<dim, VectorType, spacedim>;
#else
    template <int dim, int spacedim>
    using NativeTransfer = dealii::SolutionTransfer<dim, VectorType, spacedim>;
#endif
  } // namespace

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::save_checkpoint(
    const std::string                              &directory,
    const double                                    t,
    const VectorType                               &state,
    const VectorType                               &derivative,
    const std::function<void(const std::string &)> &prepare_additional_data)
    const
  {
    const auto reference_directory =
      Utilities::MPI::broadcast(mpi_communicator_, directory, 0);
    AssertThrow(Utilities::MPI::min(directory == reference_directory ? 1u : 0u,
                                    mpi_communicator_) == 1,
                ExcMessage("Checkpoint directories differ between MPI ranks."));
    AssertThrow(
      Utilities::MPI::min(
        std::isfinite(t) && !external_pressure_provider &&
            dof_handler_.n_dofs() > 0 &&
            state.size() == dof_handler_.n_dofs() &&
            derivative.size() == state.size() &&
            state.locally_owned_elements() == locally_owned_dofs_ &&
            derivative.locally_owned_elements() == locally_owned_dofs_ ?
          1u :
          0u,
        mpi_communicator_) == 1,
      ExcMessage(
        "Checkpoint needs a finite time, native state pair, and no attached external pressure callable."));
    AssertThrow(Utilities::MPI::min(t, mpi_communicator_) ==
                  Utilities::MPI::max(t, mpi_communicator_),
                ExcMessage("Checkpoint times differ between MPI ranks."));
    AssertThrow(std::isfinite(state.l2_norm()),
                ExcMessage("Nonfinite checkpoint state."));
    AssertThrow(std::isfinite(derivative.l2_norm()),
                ExcMessage("Nonfinite checkpoint derivative."));
    const std::filesystem::path destination(directory),
      staging(directory + ".incomplete");
    collective_file_operation(mpi_communicator_, [&] {
      AssertThrow(!directory.empty(),
                  ExcMessage("Empty checkpoint directory."));
      if (this_mpi_process == 0)
        {
          AssertThrow(!std::filesystem::exists(destination),
                      ExcMessage("Checkpoint destination already exists."));
          if (!destination.parent_path().empty())
            std::filesystem::create_directories(destination.parent_path());
          AssertThrow(std::filesystem::create_directory(staging),
                      ExcMessage(
                        "Checkpoint staging directory already exists."));
        }
    });

    CheckpointData<dim, spacedim> data;
    data.ranks           = n_mpi_processes;
    data.time            = t;
    data.additional_data = static_cast<bool>(prepare_additional_data);
    for (const auto &[id, vessel] : vessel_map)
      data.physics.vessels.emplace(
        id, static_cast<const VesselProperties &>(vessel));
    data.physics.terminals        = rcr_map;
    data.physics.sealed_terminals = sealed_terminal_ids;
    for (const auto &[id, links] : terminal_links)
      for (const auto &[other, resistance] : links)
        if (id < other)
          data.physics.terminal_connections.push_back({id, other, resistance});
    data.vessel_terminals = vessel_terminals;
    data.inflow           = inflow_series;
    std::ostringstream parameters;
    this->prm.print_parameters(parameters,
                               ParameterHandler::PRM | ParameterHandler::Short);
    data.parameters = parameters.str();
    collective_file_operation(mpi_communicator_, [&] {
      if (this_mpi_process == 0)
        {
          std::ofstream file(staging / "application.bin", std::ios::binary);
          AssertThrow(file, ExcMessage("Cannot create application archive."));
          {
            boost::archive::binary_oarchive archive(file);
            archive << data;
          }
          file.close();
          AssertThrow(file, ExcMessage("Cannot finish application archive."));
        }
      std::ofstream file(staging / ("elements-" +
                                    std::to_string(this_mpi_process) + ".bin"),
                         std::ios::binary);
      AssertThrow(file,
                  ExcMessage("Cannot create element association archive."));
      {
        boost::archive::binary_oarchive archive(file);
        archive << cell_elements(dof_handler_);
      }
      file.close();
      AssertThrow(file,
                  ExcMessage("Cannot finish element association archive."));
    });

    VectorType relevant_state(locally_owned_dofs_,
                              locally_relevant_dofs_,
                              mpi_communicator_);
    VectorType relevant_derivative(locally_owned_dofs_,
                                   locally_relevant_dofs_,
                                   mpi_communicator_);
    relevant_state      = state;
    relevant_derivative = derivative;
    NativeTransfer<dim, spacedim> transfer(dof_handler_);
    transfer.prepare_for_serialization(
      std::vector<const VectorType *>{&relevant_state, &relevant_derivative});
    collective_file_operation(mpi_communicator_, [&] {
      if (prepare_additional_data)
        prepare_additional_data(staging.string());
    });
    triangulation_.save((staging / "mesh").string());
    AssertThrowMPI(MPI_Barrier(mpi_communicator_));

    // A committed directory is visible only after all mesh/vector and envelope
    // files have closed on every rank. Interrupted writes remain .incomplete.
    collective_file_operation(mpi_communicator_, [&] {
      if (this_mpi_process == 0)
        {
          std::ofstream manifest(staging / "COMPLETE");
          manifest << "MetricFlowX checkpoint 1\n";
          for (const auto &entry :
               std::filesystem::recursive_directory_iterator(staging))
            if (entry.is_regular_file() && entry.path() != staging / "COMPLETE")
              manifest
                << std::quoted(
                     std::filesystem::relative(entry.path(), staging).string())
                << ' ' << std::filesystem::file_size(entry.path()) << '\n';
          manifest.close();
          AssertThrow(manifest,
                      ExcMessage("Cannot finish checkpoint manifest."));
          AssertThrow(!std::filesystem::exists(destination),
                      ExcMessage(
                        "Checkpoint destination appeared during write."));
          std::filesystem::rename(staging, destination);
        }
    });
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::load_checkpoint(
    const std::string                              &directory,
    const std::function<void(const std::string &)> &restore_additional_data)
  {
    const auto reference_directory =
      Utilities::MPI::broadcast(mpi_communicator_, directory, 0);
    AssertThrow(Utilities::MPI::min(directory == reference_directory ? 1u : 0u,
                                    mpi_communicator_) == 1,
                ExcMessage("Checkpoint directories differ between MPI ranks."));
    AssertThrow(Utilities::MPI::min(triangulation_.n_global_active_cells() ==
                                          0 &&
                                        dof_handler_.n_dofs() == 0 ?
                                      1u :
                                      0u,
                                    mpi_communicator_) == 1,
                ExcMessage("Load checkpoint into a fresh BloodFlowSystem."));
    const std::filesystem::path   root(directory);
    CheckpointData<dim, spacedim> data;
    collective_file_operation(mpi_communicator_, [&] {
      AssertThrow(root.extension() != ".incomplete",
                  ExcMessage("Uncommitted checkpoint."));
      std::ifstream manifest(root / "COMPLETE");
      std::string   header;
      std::getline(manifest, header);
      AssertThrow(header == data.magic,
                  ExcMessage("Missing or incompatible checkpoint manifest."));
      std::string    name;
      std::uintmax_t bytes;
      unsigned int   entries = 0;
      while (manifest >> std::quoted(name) >> bytes)
        {
          const std::filesystem::path relative(name);
          bool safe = !relative.empty() && !relative.is_absolute();
          for (const auto &part : relative)
            safe = safe && part != "..";
          AssertThrow(safe &&
                        std::filesystem::is_regular_file(root / relative) &&
                        std::filesystem::file_size(root / name) == bytes,
                      ExcMessage("Missing or truncated checkpoint payload: " +
                                 name));
          ++entries;
        }
      AssertThrow(manifest.eof() && entries >= 4,
                  ExcMessage("Invalid checkpoint manifest."));
      std::ifstream file(root / "application.bin", std::ios::binary);
      boost::archive::binary_iarchive archive(file);
      archive >> data;
      const CheckpointData<dim, spacedim> expected;
      AssertThrow(data.magic == expected.magic &&
                    data.dealii_version == expected.dealii_version &&
                    data.backend == expected.backend &&
                    data.ranks == n_mpi_processes && std::isfinite(data.time),
                  ExcMessage(
                    "Incompatible checkpoint version, backend, or MPI size."));
      AssertThrow(
        !data.additional_data || static_cast<bool>(restore_additional_data),
        ExcMessage(
          "Checkpoint requires its owning application's additional-data restore hook."));
    });

    // The archived parameter text retains original input paths for provenance,
    // but no path is opened: mesh/physics/inflow come from native archives.
    // ParameterAcceptor's normal callbacks rebuild parsed functions/constants.
    loading_checkpoint_ = true;
    collective_file_operation(mpi_communicator_, [&] {
      std::istringstream input(data.parameters);
      ParameterAcceptor::declare_all_parameters(this->prm);
      // Native replay readers need not register unrelated standalone output or
      // solver controls. An owning application registers its own entries first.
      this->prm.parse_input(input, "checkpoint parameters", "", true);
      ParameterAcceptor::parse_all_parameters(this->prm);
    });
    // Reuse normal parameter validation and derived option conversion.
    finalize_parameter_initialization();
    loading_checkpoint_ = false;
    inflow_series       = data.inflow;
    triangulation_.load((root / "mesh").string());
    // Rebuild from the exact resolved native coefficients, rather than going
    // through file readers or imposing the separate in-memory input policy.
    vessel_map.clear();
    for (const auto &[id, vessel] : data.physics.vessels)
      {
        VesselPhysicalProperties value;
        static_cast<VesselProperties &>(value) = vessel;
        value.beta_p = compute_beta_p(value.E, value.h_wall);
        vessel_map.emplace(id, value);
      }
    rcr_map                    = data.physics.terminals;
    sealed_terminal_ids        = data.physics.sealed_terminals;
    input_terminal_connections = data.physics.terminal_connections;
    vessel_terminals           = data.vessel_terminals;
    in_memory_physics          = true;
    setup_system();
    initialize_terminal_capacitors();
    build_per_cell_mass_inv();
    collective_file_operation(mpi_communicator_, [&] {
      std::map<std::string, unsigned int> expected;
      std::ifstream file(root / ("elements-" +
                                 std::to_string(this_mpi_process) + ".bin"),
                         std::ios::binary);
      boost::archive::binary_iarchive archive(file);
      archive >> expected;
      AssertThrow(expected == cell_elements(dof_handler_),
                  ExcMessage(
                    "Checkpoint cell/finite-element association differs."));
    });
    NativeTransfer<dim, spacedim> transfer(dof_handler_);
    std::vector<VectorType *>     vectors{&solution, &solution_dot};
    transfer.deserialize(vectors);
    time = data.time;
    collective_file_operation(mpi_communicator_, [&] {
      if (restore_additional_data)
        restore_additional_data(root.string());
    });
  }

  template void
  BloodFlowSystem<1, 3>::save_checkpoint(
    const std::string &,
    double,
    const VectorType &,
    const VectorType &,
    const std::function<void(const std::string &)> &) const;
  template void
  BloodFlowSystem<1, 3>::load_checkpoint(
    const std::string &,
    const std::function<void(const std::string &)> &);
} // namespace MetricFlowX
