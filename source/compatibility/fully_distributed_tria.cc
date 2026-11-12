// SPDX-License-Identifier: LGPL-2.1-or-later
// Adapted from deal.II 9.7.1 source/distributed/fully_distributed_tria.cc.
// Copyright (C) 2019 - 2025 by the deal.II authors.
#include <deal.II/base/mpi_large_count.h>

#include <deal.II/grid/tria_description.h>

#include <metric_flow_x/compatibility/fully_distributed_tria.h>

#include <fstream>

namespace MetricFlowX
{
  namespace Compatibility
  {
    using namespace dealii;
    template <int dim, int spacedim>
    void
    FullyDistributedTriangulation<dim, spacedim>::save(
      const std::string &filename) const
    {
#ifdef DEAL_II_WITH_MPI

      Assert(
        this->cell_attached_data.n_attached_deserialize == 0,
        ExcMessage(
          "Not all SolutionTransfer objects have been deserialized after the last call to load()."));
      Assert(this->n_cells() > 0,
             ExcMessage("Can not save() an empty Triangulation."));

      const int myrank =
        Utilities::MPI::this_mpi_process(this->mpi_communicator);
      const int mpisize =
        Utilities::MPI::n_mpi_processes(this->mpi_communicator);

      // Compute global offset for each rank.
      unsigned int n_locally_owned_cells = this->n_locally_owned_active_cells();

      unsigned int global_first_cell = 0;

      int ierr = MPI_Exscan(&n_locally_owned_cells,
                            &global_first_cell,
                            1,
                            MPI_UNSIGNED,
                            MPI_SUM,
                            this->mpi_communicator);
      AssertThrowMPI(ierr);

      // MPI_Exscan leaves the receive buffer undefined on rank zero.
      if (myrank == 0)
        global_first_cell = 0;

      if (myrank == 0)
        {
          std::string   fname = std::string(filename) + ".info";
          std::ofstream f(fname);
          f << "version nproc n_attached_fixed_size_objs n_attached_variable_size_objs n_global_active_cells"
            << std::endl
            << ::dealii::internal::CellAttachedDataSerializer<dim, spacedim>::
                 version_number
            << " " << Utilities::MPI::n_mpi_processes(this->mpi_communicator)
            << " " << this->cell_attached_data.pack_callbacks_fixed.size()
            << " " << this->cell_attached_data.pack_callbacks_variable.size()
            << " " << this->n_global_active_cells() << std::endl;
        }

      // The attached-data serializer expects a cell index, not a byte offset.
      this->save_attached_data(global_first_cell,
                               this->n_global_active_cells(),
                               filename);

      // Save triangulation description.
      {
        MPI_Info info;
        int      ierr = MPI_Info_create(&info);
        AssertThrowMPI(ierr);

        const std::string fname_tria = filename + "_triangulation.data";

        // Open file.
        MPI_File fh;
        ierr = MPI_File_open(this->mpi_communicator,
                             fname_tria.c_str(),
                             MPI_MODE_CREATE | MPI_MODE_WRONLY,
                             info,
                             &fh);
        AssertThrowMPI(ierr);

        ierr = MPI_File_set_size(fh, 0); // delete the file contents
        AssertThrowMPI(ierr);
        // this barrier is necessary, because otherwise others might already
        // write while one core is still setting the size to zero.
        ierr = MPI_Barrier(this->mpi_communicator);
        AssertThrowMPI(ierr);
        ierr = MPI_Info_free(&info);
        AssertThrowMPI(ierr);
        // ------------------

        // Create construction data.
        const auto construction_data = TriangulationDescription::Utilities::
          create_description_from_triangulation(
            *this,
            this->mpi_communicator,
            (this->is_multilevel_hierarchy_constructed() ?
               TriangulationDescription::Settings::
                 construct_multigrid_hierarchy :
               TriangulationDescription::Settings::default_setting));

        // Pack.
        std::vector<char> buffer;
        dealii::Utilities::pack(construction_data, buffer, false);

        // Write offsets to file.
        const std::uint64_t buffer_size = buffer.size();

        std::uint64_t offset = 0;

        ierr = MPI_Exscan(
          &buffer_size,
          &offset,
          1,
          Utilities::MPI::mpi_type_id_for_type<decltype(buffer_size)>,
          MPI_SUM,
          this->mpi_communicator);
        AssertThrowMPI(ierr);

        // Write offsets to file.
        ierr = MPI_File_write_at(
          fh,
          myrank * sizeof(std::uint64_t),
          &buffer_size,
          1,
          Utilities::MPI::mpi_type_id_for_type<decltype(buffer_size)>,
          MPI_STATUS_IGNORE);
        AssertThrowMPI(ierr);

        if (myrank == 0)
          offset = 0;

        // global position in file
        const std::uint64_t global_position =
          mpisize * sizeof(std::uint64_t) + offset;

        // Write buffers to file.
        ierr = dealii::Utilities::MPI::LargeCount::File_write_at_c(
          fh,
          global_position,
          buffer.data(),
          buffer.size(), // local buffer
          MPI_CHAR,
          MPI_STATUS_IGNORE);
        AssertThrowMPI(ierr);

        ierr = MPI_File_close(&fh);
        AssertThrowMPI(ierr);
      }
#else
      (void)filename;

      AssertThrow(false, ExcNeedsMPI());
#endif
    }



    template <int dim, int spacedim>
    void
    FullyDistributedTriangulation<dim, spacedim>::load(
      const std::string &filename)
    {
#ifdef DEAL_II_WITH_MPI
      Assert(this->n_cells() == 0,
             ExcMessage("load() only works if the Triangulation is empty!"));


      unsigned int version, numcpus, attached_count_fixed,
        attached_count_variable, n_global_active_cells;
      {
        std::string   fname = std::string(filename) + ".info";
        std::ifstream f(fname);
        AssertThrow(f.fail() == false, ExcIO());
        std::string firstline;
        getline(f, firstline);
        f >> version >> numcpus >> attached_count_fixed >>
          attached_count_variable >> n_global_active_cells;
      }

      const auto expected_version = ::dealii::internal::
        CellAttachedDataSerializer<dim, spacedim>::version_number;

      AssertThrow(version == expected_version,
                  ExcMessage("Incompatible version found in .info file."));

      // Load description and construct the triangulation.
      {
        const int myrank =
          Utilities::MPI::this_mpi_process(this->mpi_communicator);
        const int mpisize =
          Utilities::MPI::n_mpi_processes(this->mpi_communicator);

        AssertDimension(numcpus, mpisize);

        // Open file.
        MPI_Info info;
        int      ierr = MPI_Info_create(&info);
        AssertThrowMPI(ierr);

        const std::string fname_tria = filename + "_triangulation.data";

        MPI_File fh;
        ierr = MPI_File_open(this->mpi_communicator,
                             fname_tria.c_str(),
                             MPI_MODE_RDONLY,
                             info,
                             &fh);
        AssertThrowMPI(ierr);

        ierr = MPI_Info_free(&info);
        AssertThrowMPI(ierr);

        // Read offsets from file.
        std::uint64_t buffer_size;

        ierr = MPI_File_read_at(
          fh,
          myrank * sizeof(std::uint64_t),
          &buffer_size,
          1,
          Utilities::MPI::mpi_type_id_for_type<decltype(buffer_size)>,
          MPI_STATUS_IGNORE);
        AssertThrowMPI(ierr);

        std::uint64_t offset = 0;

        ierr = MPI_Exscan(
          &buffer_size,
          &offset,
          1,
          Utilities::MPI::mpi_type_id_for_type<decltype(buffer_size)>,
          MPI_SUM,
          this->mpi_communicator);
        AssertThrowMPI(ierr);

        if (myrank == 0)
          offset = 0;

        // global position in file
        const std::uint64_t global_position =
          mpisize * sizeof(std::uint64_t) + offset;

        // Read buffers from file.
        std::vector<char> buffer(buffer_size);
        ierr = dealii::Utilities::MPI::LargeCount::File_read_at_c(
          fh,
          global_position,
          buffer.data(),
          buffer.size(), // local buffer
          MPI_CHAR,
          MPI_STATUS_IGNORE);
        AssertThrowMPI(ierr);

        ierr = MPI_File_close(&fh);
        AssertThrowMPI(ierr);

        auto construction_data = dealii::Utilities::template unpack<
          TriangulationDescription::Description<dim, spacedim>>(buffer, false);

        // WARNING: serialization cannot handle the MPI communicator
        // which is the reason why we have to set it here explicitly
        construction_data.comm = this->mpi_communicator;

        this->create_triangulation(construction_data);
      }

      // Compute global offset for each rank.
      unsigned int n_locally_owned_cells = this->n_locally_owned_active_cells();

      unsigned int global_first_cell = 0;

      int ierr = MPI_Exscan(&n_locally_owned_cells,
                            &global_first_cell,
                            1,
                            MPI_UNSIGNED,
                            MPI_SUM,
                            this->mpi_communicator);
      AssertThrowMPI(ierr);

      // MPI_Exscan leaves the receive buffer undefined on rank zero.
      if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
        global_first_cell = 0;

      Assert(this->n_global_active_cells() == n_global_active_cells,
             ExcMessage("Number of global active cells differ!"));

      // clear all of the callback data, as explained in the documentation of
      // register_data_attach()
      this->cell_attached_data.n_attached_data_sets = 0;
      this->cell_attached_data.n_attached_deserialize =
        attached_count_fixed + attached_count_variable;

      // The attached-data serializer expects a cell index, not a byte offset.
      this->load_attached_data(global_first_cell,
                               this->n_global_active_cells(),
                               this->n_locally_owned_active_cells(),
                               filename,
                               attached_count_fixed,
                               attached_count_variable);

      // create_triangulation() has already rebuilt cell relations and caches.
      this->update_periodic_face_map();
#else
      (void)filename;

      AssertThrow(false, ExcNeedsMPI());
#endif
    }



    template class FullyDistributedTriangulation<1, 3>;
  } // namespace Compatibility
} // namespace MetricFlowX
