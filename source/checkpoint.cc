// SPDX-License-Identifier: LGPL-2.1-or-later
#include <metric_flow_x/blood_flow_system.h>

#if DEAL_II_VERSION_GTE(9, 7, 0)
#  include <deal.II/numerics/solution_transfer.h>
#else
#  include <deal.II/distributed/solution_transfer.h>
#endif

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <map>
#include <sstream>
#include <utility>
#include <vector>

namespace MetricFlowX
{
  using dealii::ExcMessage;
  namespace
  {
    /// @cond INTERNAL
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
        archive &magic &dealii_version &backend &ranks &time &additional_data
          &physics &vessel_terminals &inflow &parameters;
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

    void
    check_same_directory(MPI_Comm comm, const std::string &directory)
    {
      const auto reference =
        dealii::Utilities::MPI::broadcast(comm, directory, 0);
      AssertThrow(dealii::Utilities::MPI::min(directory == reference ? 1u : 0u,
                                              comm) == 1,
                  ExcMessage(
                    "Checkpoint directories differ between MPI ranks."));
    }

    // Checks that a committed directory lists only safe, complete payloads.
    void
    check_manifest(const std::filesystem::path &root,
                   const std::string           &magic,
                   const unsigned int           minimum_entries)
    {
      AssertThrow(root.extension() != ".incomplete",
                  ExcMessage("Uncommitted checkpoint."));
      std::ifstream manifest(root / "COMPLETE");
      std::string   header;
      std::getline(manifest, header);
      AssertThrow(header == magic,
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
      AssertThrow(manifest.eof() && entries >= minimum_entries,
                  ExcMessage("Invalid checkpoint manifest."));
    }

    // Rank 0 creates the staging directory of a new destination.
    std::filesystem::path
    create_staging(MPI_Comm comm, const std::string &directory)
    {
      const std::filesystem::path destination(directory),
        staging(directory + ".incomplete");
      collective_file_operation(comm, [&] {
        AssertThrow(!directory.empty(),
                    ExcMessage("Empty checkpoint directory."));
        if (dealii::Utilities::MPI::this_mpi_process(comm) == 0)
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
      return staging;
    }

    // A committed directory is visible only after all payload files have
    // closed on every rank. Interrupted writes remain .incomplete.
    void
    commit_staging(MPI_Comm                     comm,
                   const std::filesystem::path &staging,
                   const std::string           &directory,
                   const std::string           &magic)
    {
      AssertThrowMPI(MPI_Barrier(comm));
      const std::filesystem::path destination(directory);
      collective_file_operation(comm, [&] {
        if (dealii::Utilities::MPI::this_mpi_process(comm) == 0)
          {
            std::ofstream manifest(staging / "COMPLETE");
            manifest << magic << '\n';
            for (const auto &entry :
                 std::filesystem::recursive_directory_iterator(staging))
              if (entry.is_regular_file() &&
                  entry.path() != staging / "COMPLETE")
                manifest << std::quoted(
                              std::filesystem::relative(entry.path(), staging)
                                .string())
                         << ' ' << std::filesystem::file_size(entry.path())
                         << '\n';
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

    template <class Data>
    void
    write_archive(const std::filesystem::path &path, const Data &data)
    {
      std::ofstream file(path, std::ios::binary);
      AssertThrow(file, ExcMessage("Cannot create archive " + path.string()));
      {
        boost::archive::binary_oarchive archive(file);
        archive << data;
      }
      file.close();
      AssertThrow(file, ExcMessage("Cannot finish archive " + path.string()));
    }

    template <class Data>
    Data
    read_archive(const std::filesystem::path &path)
    {
      Data          data;
      std::ifstream file(path, std::ios::binary);
      AssertThrow(file, ExcMessage("Cannot open archive " + path.string()));
      boost::archive::binary_iarchive archive(file);
      archive >> data;
      return data;
    }

    // Envelope of a coupling field directory. Field values are stored per rank.
    struct CouplingFieldsData
    {
      std::string magic          = "MetricFlowX coupling fields 1";
      std::string dealii_version = DEAL_II_PACKAGE_VERSION;
#ifdef METRIC_FLOW_X_WITH_PETSC
      std::string backend = "PETSc";
#else
      std::string backend = "Trilinos";
#endif
      unsigned int             ranks = 0;
      double                   time  = 0.;
      std::vector<std::string> fields;

      template <class Archive>
      void
      serialize(Archive &archive, const unsigned int)
      {
        archive &magic &dealii_version &backend &ranks &time &fields;
      }
    };

    // Values of one field on the locally owned cells of one rank, in CellId
    // order. layout identifies the cells, their vertices and their elements.
    struct CouplingFieldBlock
    {
      std::uint64_t       layout = 0;
      std::vector<double> values;

      template <class Archive>
      void
      serialize(Archive &archive, const unsigned int)
      {
        archive &layout &values;
      }
    };

    // 64-bit FNV-1a (Fowler-Noll-Vo) hash. Unlike std::hash, its values are
    // fixed by the algorithm, so writer and reader processes agree. It detects
    // accidental layout mismatches; it is not a cryptographic check.
    //
    // The offset basis is the initial hash value, so that leading zero bytes
    // still change the hash. The prime, 2^40 + 2^8 + 0xb3, spreads every byte
    // over the hash bits; the product wraps modulo 2^64 by unsigned overflow.
    constexpr std::uint64_t fnv_offset_basis = 14695981039346656037ull;
    constexpr std::uint64_t fnv_prime        = 1099511628211ull;

    void
    hash_bytes(std::uint64_t &hash, const void *data, const std::size_t size)
    {
      const auto *bytes = static_cast<const unsigned char *>(data);
      for (std::size_t i = 0; i < size; ++i)
        {
          hash ^= bytes[i];
          hash *= fnv_prime;
        }
    }

    // Locally owned cells in CellId order, and a fingerprint of their ids,
    // vertices and active finite elements.
    template <int dim, int spacedim>
    std::pair<
      std::vector<
        typename dealii::DoFHandler<dim, spacedim>::active_cell_iterator>,
      std::uint64_t>
    owned_cell_layout(const dealii::DoFHandler<dim, spacedim> &dofs)
    {
      std::vector<
        typename dealii::DoFHandler<dim, spacedim>::active_cell_iterator>
        cells;
      for (const auto &cell : dofs.active_cell_iterators())
        if (cell->is_locally_owned())
          cells.push_back(cell);
      std::sort(cells.begin(), cells.end(), [](const auto &a, const auto &b) {
        return a->id() < b->id();
      });
      std::uint64_t hash = fnv_offset_basis;
      for (const auto &cell : cells)
        {
          const auto         id    = cell->id().to_string();
          const unsigned int index = cell->active_fe_index();
          hash_bytes(hash, id.data(), id.size());
          hash_bytes(hash, &index, sizeof(index));
          for (const auto v : cell->vertex_indices())
            for (unsigned int d = 0; d < spacedim; ++d)
              {
                const double x = cell->vertex(v)[d];
                hash_bytes(hash, &x, sizeof(x));
              }
        }
      return {cells, hash};
    }

    // Native component that holds a coupling field.
    unsigned int
    coupling_field_component(const std::string &field)
    {
      AssertThrow(field == "area" || field == "velocity" || field == "pressure",
                  ExcMessage("Unknown coupling field '" + field +
                             "'; use area, velocity or pressure."));
      return field == "velocity" ? 1 : 0;
    }

    std::map<std::string, unsigned int>
    read_cell_elements(const std::filesystem::path &root,
                       const unsigned int           rank)
    {
      return read_archive<std::map<std::string, unsigned int>>(
        root / ("elements-" + std::to_string(rank) + ".bin"));
    }

    // Validates the manifest and the application envelope of a committed
    // checkpoint on this rank. Call it inside collective_file_operation().
    template <int dim, int spacedim>
    CheckpointData<dim, spacedim>
    read_checkpoint_data(const std::filesystem::path &root,
                         const unsigned int           ranks)
    {
      CheckpointData<dim, spacedim> data;
      check_manifest(root, data.magic, 4);
      data =
        read_archive<CheckpointData<dim, spacedim>>(root / "application.bin");
      const CheckpointData<dim, spacedim> expected;
      AssertThrow(data.magic == expected.magic &&
                    data.dealii_version == expected.dealii_version &&
                    data.backend == expected.backend && data.ranks == ranks &&
                    std::isfinite(data.time),
                  ExcMessage(
                    "Incompatible checkpoint version, backend, or MPI size."));
      return data;
    }

#if DEAL_II_VERSION_GTE(9, 7, 0)
    template <int dim, int spacedim>
    using NativeTransfer = dealii::SolutionTransfer<dim, VectorType, spacedim>;
#else
    template <int dim, int spacedim>
    using NativeTransfer = dealii::parallel::distributed::
      SolutionTransfer<dim, VectorType, spacedim>;
#endif
    /// @endcond
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
    check_same_directory(mpi_communicator_, directory);
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
    const auto staging = create_staging(mpi_communicator_, directory);

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
        write_archive(staging / "application.bin", data);
      write_archive(staging /
                      ("elements-" + std::to_string(this_mpi_process) + ".bin"),
                    cell_elements(dof_handler_));
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
    commit_staging(mpi_communicator_,
                   staging,
                   directory,
                   CheckpointData<dim, spacedim>().magic);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::load_checkpoint(
    const std::string                              &directory,
    const std::function<void(const std::string &)> &restore_additional_data)
  {
    check_same_directory(mpi_communicator_, directory);
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
      data = read_checkpoint_data<dim, spacedim>(root, n_mpi_processes);
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
      AssertThrow(read_cell_elements(root, this_mpi_process) ==
                    cell_elements(dof_handler_),
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

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::load_checkpoint_state(
    const std::string &directory)
  {
    check_same_directory(mpi_communicator_, directory);
    AssertThrow(dof_handler_.n_dofs() > 0,
                ExcMessage(
                  "Load a checkpoint state into a set-up BloodFlowSystem."));
    const std::filesystem::path   root(directory);
    CheckpointData<dim, spacedim> data;
    collective_file_operation(mpi_communicator_, [&] {
      data = read_checkpoint_data<dim, spacedim>(root, n_mpi_processes);
      AssertThrow(read_cell_elements(root, this_mpi_process) ==
                    cell_elements(dof_handler_),
                  ExcMessage(
                    "Checkpoint cell/finite-element association differs."));
    });

    // Only the native state is needed. Load the archived mesh with its
    // attached vectors into a temporary triangulation, and rebuild the same
    // finite element association there without any system setup.
    Compatibility::FullyDistributedTriangulation<dim, spacedim> mesh(
      mpi_communicator_);
    mesh.load((root / "mesh").string());
    DoFHandler<dim, spacedim> dofs(mesh);

    using CellIterator =
      typename DoFHandler<dim, spacedim>::active_cell_iterator;
    std::map<CellId, CellIterator> current;
    for (const auto &cell : dof_handler_.active_cell_iterators())
      if (cell->is_locally_owned())
        current.emplace(cell->id(), cell);
    std::vector<std::pair<CellIterator, CellIterator>> pairs;

    bool same =
      mesh.n_global_active_cells() == triangulation_.n_global_active_cells();
    for (const auto &cell : dofs.active_cell_iterators())
      if (same && cell->is_locally_owned())
        {
          const auto match = current.find(cell->id());
          same             = match != current.end();
          if (!same)
            break;
          const auto &other = match->second;
          same              = cell->material_id() == other->material_id();
          for (const auto v : cell->vertex_indices())
            same = same && cell->vertex(v) == other->vertex(v);
          for (const auto f : cell->face_indices())
            same = same && cell->face(f)->boundary_id() ==
                             other->face(f)->boundary_id();
          cell->set_active_fe_index(other->active_fe_index());
          pairs.emplace_back(cell, other);
        }
    same = same && pairs.size() == current.size();
    AssertThrow(Utilities::MPI::min(same ? 1u : 0u, mpi_communicator_) == 1,
                ExcMessage("Checkpoint mesh differs from the loaded mesh."));
    dofs.distribute_dofs(fe_collection_);
    AssertThrow(dofs.n_dofs() == dof_handler_.n_dofs(),
                ExcMessage("Checkpoint degrees of freedom differ."));

    VectorType stored_state(dofs.locally_owned_dofs(), mpi_communicator_);
    VectorType stored_rate(dofs.locally_owned_dofs(), mpi_communicator_);
    {
      NativeTransfer<dim, spacedim> transfer(dofs);
      std::vector<VectorType *>     vectors{&stored_state, &stored_rate};
      transfer.deserialize(vectors);
    }

    // Every unknown belongs to one cell, so copy cell by cell through the
    // matching CellId instead of assuming the two numberings coincide.
    VectorType new_state(locally_owned_dofs_, mpi_communicator_);
    VectorType new_rate(locally_owned_dofs_, mpi_communicator_);
    std::vector<types::global_dof_index> from, to;
    for (const auto &[stored, other] : pairs)
      {
        from.resize(stored->get_fe().n_dofs_per_cell());
        to.resize(other->get_fe().n_dofs_per_cell());
        stored->get_dof_indices(from);
        other->get_dof_indices(to);
        for (unsigned int i = 0; i < to.size(); ++i)
          {
            new_state[to[i]] = stored_state[from[i]];
            new_rate[to[i]]  = stored_rate[from[i]];
          }
      }
    new_state.compress(VectorOperation::insert);
    new_rate.compress(VectorOperation::insert);

    solution     = new_state;
    solution_dot = new_rate;
    time         = data.time;
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::save_coupling_fields(
    const std::string              &directory,
    const double                    t,
    const VectorType               &state,
    const VectorType               &derivative,
    const std::vector<std::string> &fields) const
  {
    check_same_directory(mpi_communicator_, directory);
    AssertThrow(!fields.empty(), ExcMessage("No coupling fields selected."));
    for (unsigned int i = 0; i < fields.size(); ++i)
      {
        coupling_field_component(fields[i]);
        AssertThrow(std::find(fields.begin(), fields.begin() + i, fields[i]) ==
                      fields.begin() + i,
                    ExcMessage("Duplicate coupling field " + fields[i] + "."));
      }
    AssertThrow(
      Utilities::MPI::min(
        std::isfinite(t) && dof_handler_.n_dofs() > 0 &&
            state.size() == dof_handler_.n_dofs() &&
            derivative.size() == state.size() &&
            state.locally_owned_elements() == locally_owned_dofs_ &&
            derivative.locally_owned_elements() == locally_owned_dofs_ ?
          1u :
          0u,
        mpi_communicator_) == 1,
      ExcMessage("Coupling fields need a finite time and native state pair."));
    AssertThrow(Utilities::MPI::min(t, mpi_communicator_) ==
                  Utilities::MPI::max(t, mpi_communicator_),
                ExcMessage("Coupling field times differ between MPI ranks."));

    VectorType pressure;
    if (std::find(fields.begin(), fields.end(), "pressure") != fields.end())
      {
        pressure.reinit(locally_owned_dofs_, mpi_communicator_);
        compute_pressure(state, derivative, pressure, t);
      }
    const auto cell_layout = owned_cell_layout(dof_handler_);
    std::vector<CouplingFieldBlock>      blocks(fields.size());
    std::vector<types::global_dof_index> dofs;
    for (unsigned int f = 0; f < fields.size(); ++f)
      {
        const unsigned int component = coupling_field_component(fields[f]);
        const VectorType  &source = fields[f] == "pressure" ? pressure : state;
        blocks[f].layout          = cell_layout.second;
        for (const auto &cell : cell_layout.first)
          {
            dofs.resize(cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(dofs);
            for (unsigned int i = 0; i < dofs.size(); ++i)
              if (cell->get_fe().system_to_component_index(i).first ==
                  component)
                blocks[f].values.push_back(source[dofs[i]]);
          }
        AssertThrow(Utilities::MPI::min(std::all_of(blocks[f].values.begin(),
                                                    blocks[f].values.end(),
                                                    [](const double value) {
                                                      return std::isfinite(
                                                        value);
                                                    }) ?
                                          1u :
                                          0u,
                                        mpi_communicator_) == 1,
                    ExcMessage("Nonfinite coupling field " + fields[f] + "."));
      }

    const auto staging = create_staging(mpi_communicator_, directory);
    collective_file_operation(mpi_communicator_, [&] {
      if (this_mpi_process == 0)
        {
          CouplingFieldsData data;
          data.ranks  = n_mpi_processes;
          data.time   = t;
          data.fields = fields;
          write_archive(staging / "fields.bin", data);
        }
      for (unsigned int f = 0; f < fields.size(); ++f)
        write_archive(staging / (fields[f] + "-" +
                                 std::to_string(this_mpi_process) + ".bin"),
                      blocks[f]);
    });
    commit_staging(mpi_communicator_,
                   staging,
                   directory,
                   CouplingFieldsData().magic);
  }

  template <int dim, int spacedim>
  double
  BloodFlowSystem<dim, spacedim>::load_coupling_field(
    const std::string &directory,
    const std::string &field,
    VectorType        &values) const
  {
    check_same_directory(mpi_communicator_, directory);
    const unsigned int component = coupling_field_component(field);
    AssertThrow(dof_handler_.n_dofs() > 0 &&
                  values.size() == dof_handler_.n_dofs() &&
                  values.locally_owned_elements() == locally_owned_dofs_,
                ExcMessage("Load coupling fields into a set-up system and a "
                           "vector with the layout of make_state()."));
    const std::filesystem::path root(directory);
    const auto                  cell_layout = owned_cell_layout(dof_handler_);
    const auto                 &cells       = cell_layout.first;
    const std::uint64_t         layout      = cell_layout.second;
    CouplingFieldsData          data;
    CouplingFieldBlock          block;
    collective_file_operation(mpi_communicator_, [&] {
      check_manifest(root, data.magic, 2);
      data = read_archive<CouplingFieldsData>(root / "fields.bin");
      const CouplingFieldsData expected;
      AssertThrow(data.magic == expected.magic &&
                    data.dealii_version == expected.dealii_version &&
                    data.backend == expected.backend &&
                    data.ranks == n_mpi_processes && std::isfinite(data.time),
                  ExcMessage(
                    "Incompatible coupling fields version, backend, or MPI "
                    "size."));
      AssertThrow(std::find(data.fields.begin(), data.fields.end(), field) !=
                    data.fields.end(),
                  ExcMessage("Coupling field " + field + " was not saved."));
      block = read_archive<CouplingFieldBlock>(
        root / (field + "-" + std::to_string(this_mpi_process) + ".bin"));
      AssertThrow(block.layout == layout,
                  ExcMessage("Coupling field mesh or partition differs from "
                             "the loaded mesh."));
    });

    VectorType result(locally_owned_dofs_, mpi_communicator_);
    std::vector<types::global_dof_index> dofs;
    std::size_t                          next = 0;
    bool                                 fits = true;
    for (const auto &cell : cells)
      {
        dofs.resize(cell->get_fe().n_dofs_per_cell());
        cell->get_dof_indices(dofs);
        for (unsigned int i = 0; fits && i < dofs.size(); ++i)
          if (cell->get_fe().system_to_component_index(i).first == component)
            {
              fits = next < block.values.size();
              if (fits)
                result[dofs[i]] = block.values[next++];
            }
      }
    fits = fits && next == block.values.size();
    AssertThrow(Utilities::MPI::min(fits ? 1u : 0u, mpi_communicator_) == 1,
                ExcMessage("Coupling field size differs from the loaded "
                           "degrees of freedom."));
    result.compress(VectorOperation::insert);
    values = result;
    return data.time;
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
  template void
  BloodFlowSystem<1, 3>::load_checkpoint_state(const std::string &);
  template void
  BloodFlowSystem<1, 3>::save_coupling_fields(
    const std::string &,
    double,
    const VectorType &,
    const VectorType &,
    const std::vector<std::string> &) const;
  template double
  BloodFlowSystem<1, 3>::load_coupling_field(const std::string &,
                                             const std::string &,
                                             VectorType &) const;
} // namespace MetricFlowX
