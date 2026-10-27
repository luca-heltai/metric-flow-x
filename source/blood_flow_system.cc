/* --------------------------------------------------------------------------
 * Blood flow simulation in 1D (dim=1, spacedim=3), run on a
 * parallel::fullydistributed::Triangulation.
 *
 * Unknown layout:
 *   FESystem( FE_DGQ(fe_degree), 2,   -> comps 0,1 : cell  A, U
 *             FE_DGQ(1),        2 )   -> comps 2,3 : trace A_hat, U_hat
 *   Each rank's system block contains its FE DoFs followed by its local
 *   terminal capacitor pressures.
 *
 * Ownership follows the mesh partition: the rank owning a cell owns that
 * cell's cell-DoFs and its trace-DoFs, because every one of them is a DoF of
 * `dof_handler_`.  The only bookkeeping this file adds on top of deal.II is
 * face_dof_map, which records *which* of the four trace DoFs at a face is the
 * canonical one -- it does not decide who owns them.
 *
 * Assembly rule, obeyed by every routine below: read from the ghosted
 * vectors, write only rows this rank owns, and close with one compress().
 * --------------------------------------------------------------------------
 */
#include <deal.II/base/function_parser.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_description.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/container/small_vector.hpp>

#include <idas/idas.h>
#include <metric_flow_x/blood_flow_system.h>
#include <metric_flow_x/ida_access.h>
#include <metric_flow_x/io/vtk_utils.h>

#ifdef METRIC_FLOW_X_WITH_PETSC
#  include <petscoptions.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <type_traits>

#include "rapidcsv.h"

namespace MetricFlowX
{
  using namespace dealii;

  namespace
  {
    template <typename T>
    std::string
    csv_cell(const T &value)
    {
      if constexpr (std::is_convertible_v<T, std::string>)
        return value;
      else
        {
          std::ostringstream text;
          text.imbue(std::locale::classic());
          text << std::setprecision(std::numeric_limits<double>::max_digits10)
               << value;
          return text.str();
        }
    }

    // Serialize only the current record, preserving bounded memory and live
    // output even for long simulations. Rapidcsv owns quoting and escaping.
    void
    csv_write_record(std::ostream                   &stream,
                     const std::vector<std::string> &values)
    {
      rapidcsv::Document row("", rapidcsv::LabelParams(-1, -1));
      row.SetRow(0, values);
      row.Save(stream);
      stream.flush();
      AssertThrow(stream.good(), ExcMessage("Cannot write CSV record."));
    }

    template <typename... Values>
    void
    csv_write_row(std::ostream &stream, const Values &...values)
    {
      csv_write_record(stream, {csv_cell(values)...});
    }

    std::string
    prepare_csv_path(const std::string &directory, const std::string &filename)
    {
      const auto path = std::filesystem::path(directory) / filename;
      if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
      return path.string();
    }

    // The action must not contain collectives: all ranks report I/O failures
    // before any caller can enter its next collective.
    template <typename Action>
    void
    collective_csv_action(const MPI_Comm communicator, const Action &action)
    {
      std::string error;
      try
        {
          action();
        }
      catch (const std::exception &exception)
        {
          error = exception.what();
        }
      for (const auto &message :
           Utilities::MPI::all_gather(communicator, error))
        AssertThrow(message.empty(), ExcMessage(message));
    }

    std::ofstream
    csv_append(const std::string              &filename,
               const std::vector<std::string> &header)
    {
      const bool empty = !std::filesystem::exists(filename) ||
                         std::filesystem::file_size(filename) == 0;
      std::ofstream stream(filename, std::ios::app);
      AssertThrow(stream, ExcMessage("Cannot open " + filename));
      if (empty)
        csv_write_record(stream, header);
      return stream;
    }
  } // namespace

  // ===========================================================================
  // IDA statistics access shim.
  //
  // SUNDIALS::IDA keeps its `ida_mem` handle private, but IDAGetNumSteps and
  // friends need it.  An explicit instantiation may name a private member, so
  // Planter<&IDA::ida_mem> captures the pointer-to-member and exposes it
  // through the ADL-found friend steal().  Same trick as the pre-MetricFlowX
  // driver; it lives in exactly one translation unit.
  // ===========================================================================
  namespace internal_ida_access
  {
    struct IdaMemAccessor
    {
      using type = void *dealii::SUNDIALS::IDA<VectorType>::*;
      friend type        steal(IdaMemAccessor);
    };

    template <IdaMemAccessor::type M>
    struct Planter
    {
      friend IdaMemAccessor::type
      steal(IdaMemAccessor)
      {
        return M;
      }
    };

    template struct Planter<&dealii::SUNDIALS::IDA<VectorType>::ida_mem>;

#if DEAL_II_SUNDIALS_VERSION_GTE(7, 0, 0)
    struct IdaContextAccessor
    {
      using type = SUNContext dealii::SUNDIALS::IDA<VectorType>::*;
      friend type             steal(IdaContextAccessor);
    };

    template <IdaContextAccessor::type M>
    struct ContextPlanter
    {
      friend IdaContextAccessor::type
      steal(IdaContextAccessor)
      {
        return M;
      }
    };

    template struct ContextPlanter<&dealii::SUNDIALS::IDA<VectorType>::ida_ctx>;
#endif

    inline void *
    get_ida_mem(const dealii::SUNDIALS::IDA<VectorType> &ida)
    {
      return ida.*steal(IdaMemAccessor{});
    }
  } // namespace internal_ida_access

  namespace internal
  {
    // Shared with the runner's supported SUNDIALS tolerance-weight adapter.
    void *
    native_ida_memory(const SUNDIALS::IDA<VectorType> &ida)
    {
      return internal_ida_access::get_ida_mem(ida);
    }

#if DEAL_II_SUNDIALS_VERSION_GTE(7, 0, 0)
    SUNContext
    native_ida_context(const SUNDIALS::IDA<VectorType> &ida)
    {
      return ida.*steal(internal_ida_access::IdaContextAccessor{});
    }
#endif
  } // namespace internal

  // ==========================================================================
  // Constructor
  // ==========================================================================
  template <int dim, int spacedim>
  BloodFlowSystem<dim, spacedim>::BloodFlowSystem(const MPI_Comm comm)
    : ParameterAcceptor("MetricFlowSystem<" + std::to_string(dim) + ", " +
                        std::to_string(spacedim) + ">")
    , mpi_communicator_(comm)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(comm))
    , this_mpi_process(Utilities::MPI::this_mpi_process(comm))
    , pcout(std::cout, this_mpi_process == 0)
    , computing_timer(mpi_communicator_,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
    , direct_solver_control(1000, 1e-10)
    , par("Metric Flow Parameters",
          {"rho", "mu", "xi", "m", "Rt"},
          {1060, 0.004, 2.0, 0.5, 0.5},
          {"Density",
           "Viscosity coefficient",
           "Profile constant for friction term",
           "Tube law exponent",
           "Reflection coefficient at outflow boundary"})
    , triangulation_(mpi_communicator_)
    , dof_handler_(triangulation_)
    , fe_(nullptr)
    , rhs_function("Functions",
                   "0.0; 0.0",
                   "RHS expression",
                   par,
                   dealii::FunctionParser<spacedim>::default_variable_names() +
                     ",t")
    , exact_solution(
        "Functions",
        "1e-4; 0.0; 0.0; 0.0",
        "Exact solution",
        par,
        dealii::FunctionParser<spacedim>::default_variable_names() + ",t")
    , inflow_function("Functions", "0.0", "Inflow function", par, "x,t")
  {
    add_parameter("Finite element degree", fe_degree);
    add_parameter("Problem constants", constants);
    add_parameter("Output filename", output_filename);
    add_parameter("Use direct solver", use_direct_solver);
    add_parameter("Initial state source", initial_state_source);
    add_parameter("Initial condition policy", initial_condition_policy);
    add_parameter("Initial consistency tolerance",
                  initial_consistency_tolerance);
    add_parameter("Initial consistency maximum iterations",
                  initial_consistency_max_iterations);
    add_parameter("Check Jacobian at startup", check_initial_jacobian);

    add_parameter("Use componentwise absolute tolerances",
                  use_componentwise_tolerances);
    add_parameter("Area absolute tolerance fraction",
                  area_absolute_tolerance_fraction);
    add_parameter("Velocity absolute tolerance", velocity_absolute_tolerance);
    add_parameter("Capacitor pressure absolute tolerance",
                  capacitor_absolute_tolerance);
    add_parameter(
      "Print current statistics",
      print_statistics,
      "Print IDA progress at every output callback, independently of file cadence.");
    add_parameter("Write flow regime diagnostics",
                  write_flow_regime_diagnostics);
    add_parameter("Mach warning threshold", mach_warning_threshold);

    add_parameter("Number of refinement cycles", n_refinement_cycles);
    add_parameter("Number of global refinement", n_global_refinements);
    add_parameter("Theta (penalty parameter)", theta);
    add_parameter("Theta Boundary (stability parameter)", theta_bd);
    add_parameter("Gamma (Total pressure factor)", gamma);
    add_parameter("Verbosity (console depth)", verbosity);
    add_parameter("Output directory", output_directory);
    add_parameter(
      "Output vessel CSV file",
      output_vessel_csv_file,
      "One midpoint probe CSV relative to Output directory, or absolute. Empty disables vessel probes.");
    add_parameter(
      "Vessel probe IDs",
      vessel_probe_ids,
      "Comma-separated vessel IDs for midpoint probes. Empty selects all vessels.");
    add_parameter(
      "Output time interval",
      time_interval,
      "Positive value sets the interval between written outputs; zero writes "
      "at every IDA callback.");
    add_parameter("Numerical flux type", numerical_flux_type_str);
    add_parameter("Use Riemann Invariants", use_riemann_invariants);
    add_parameter("Use junction mesh", use_junction_mesh);
    add_parameter("Mesh partitioning",
                  mesh_partitioning,
                  "Weighted vessel graph (requires METIS), or legacy zorder.",
                  this->prm,
                  Patterns::Selection("graph|zorder"));
    add_parameter("Partition junction weight",
                  partition_junction_weight,
                  "Positive graph edge weight at shared mesh vertices.",
                  this->prm,
                  Patterns::Integer(1));
    add_parameter("Partition terminal connection weight",
                  partition_terminal_weight,
                  "Positive graph edge weight for peripheral resistive links.",
                  this->prm,
                  Patterns::Integer(1));
    add_parameter("Partition imbalance tolerance",
                  partition_imbalance,
                  "Allowed maximum load excess relative to the mean.",
                  this->prm,
                  Patterns::Double(0., 1.));
    add_parameter("Outlet boundary condition type", outlet_type);
    add_parameter("Vtk file path for mesh input", vtk_file_path);
    add_parameter(
      "Wall model",
      wall_model,
      "Square root (default), ADAVN arterial, ADAVN venous, or ADAVN mixed; ADAVN requires constitutive CELL_DATA arrays.");
    add_parameter(
      "Viscoelastic wall",
      viscoelastic_wall,
      "Add the ADAVN Kelvin-Voigt pressure using adavn_K_m; supports native or IDA initialization.");
    add_parameter(
      "Terminal parameter source",
      terminal_parameter_source,
      "VTK preserves legacy mesh values; Parameters uses defaults and optional CSV overrides.",
      this->prm,
      Patterns::Selection("VTK|Parameters"));
    add_parameter("Terminal parameters CSV file", terminal_parameters_csv_file);
    add_parameter(
      "Terminal connections CSV file",
      terminal_connections_csv_file,
      "Optional undirected resistor graph: boundary0,boundary1,resistance. "
      "Connects capacitor pressures (C>0) or trace pressures (C=0); "
      "replaces the connected terminals' fixed downstream closure.");
    add_parameter("Default terminal R1", default_terminal_r1);
    add_parameter("Default terminal R2", default_terminal_r2);
    add_parameter("Default terminal C", default_terminal_c);
    add_parameter("Default terminal P_out", default_terminal_p_out);

    add_parameter("Inflow CSV file", inflow_csv_file);
    add_parameter("Inflow CSV period", inflow_csv_period);
    add_parameter(
      "Output terminal CSV file",
      output_terminal_csv_file,
      "CSV path relative to Output directory, or absolute; empty disables terminal output.");

    add_parameter(
      "Output network terminals CSV file",
      output_network_terminals_csv_file,
      "CSV path relative to Output directory, or absolute; empty disables output.");
    add_parameter(
      "Output mass conservation CSV file",
      output_mass_conservation_csv_file,
      "CSV path relative to Output directory, or absolute; empty disables output.");
    add_parameter(
      "Output IDA statistics CSV file",
      output_ida_statistics_csv_file,
      "CSV path relative to Output directory, or absolute; empty disables output.");
    add_parameter(
      "Output IDA step history CSV file",
      output_ida_step_history_csv_file,
      "CSV path relative to Output directory, or absolute; empty disables output.");
    add_parameter(
      "Output IDA internal steps CSV file",
      output_ida_internal_steps_csv_file,
      "CSV path relative to Output directory, or absolute; empty disables output.");
    add_parameter(
      "Output scaling timings CSV file",
      output_scaling_timings_csv_file,
      "CSV path relative to Output directory, or absolute; empty disables output.");
    add_parameter(
      "Output flow regime CSV file",
      output_flow_regime_csv_file,
      "CSV path relative to Output directory, or absolute; empty disables output.");

    add_parameter(
      "Jacobian export file (Matrix Market)",
      jacobian_export_filename,
      "If non-empty, the first assembled IDA Jacobian is written to this file "
      "in Matrix Market format. Leave empty to disable.");
    add_parameter("Jacobian export time",
                  jacobian_export_time,
                  "Export the first Jacobian assembled at or after this time.");
    this->enter_subsection("IDA parameters");
    this->enter_my_subsection(this->prm);
    ida_parameters.add_parameters(this->prm);
    this->leave_my_subsection(this->prm);
    this->leave_subsection();
  }

  // ============================================================================
  // initialize_params
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::initialize_params(const std::string &filename)
  {
    TimerOutput::Scope t(computing_timer, "initialize_params");
    derivative_jacobian_is_current_ = false;

    ParameterAcceptor::initialize(filename,
                                  "last_used_parameters.prm",
                                  ParameterHandler::Short,
                                  this->prm,
                                  ParameterHandler::Short);

    AssertThrow(
      wall_model == "Square root" || wall_model == "ADAVN arterial" ||
        wall_model == "ADAVN venous" || wall_model == "ADAVN mixed",
      ExcMessage(
        "Wall model must be Square root, ADAVN arterial, ADAVN venous, or ADAVN mixed."));
    AssertThrow(
      !viscoelastic_wall || wall_model != "Square root",
      ExcMessage(
        "Viscoelastic wall requires an ADAVN wall model and its adavn_K_m input field."));
    AssertThrow(std::isfinite(time_interval) && time_interval >= 0.0,
                ExcMessage("Output time interval must be finite and "
                           "non-negative."));
    if (time_interval > 0.0)
      AssertThrow(ida_parameters.initial_time + time_interval >
                      ida_parameters.initial_time &&
                    ida_parameters.final_time - time_interval <
                      ida_parameters.final_time,
                  ExcMessage("Output time interval is too small to be "
                             "represented at the simulation time scale."));

    // deallog is written by rank 0 only; every other rank stays silent so that
    // the console is readable and the log file is not written K times over.
    deallog.depth_console(this_mpi_process == 0 ? verbosity : 0);
    deallog.depth_file(this_mpi_process == 0 ? verbosity : 0);

    for (const double value : {area_absolute_tolerance_fraction,
                               velocity_absolute_tolerance,
                               capacitor_absolute_tolerance,
                               mach_warning_threshold})
      AssertThrow(
        std::isfinite(value) && value > 0.0,
        ExcMessage(
          "Component tolerance scales and Mach threshold must be finite and positive."));

    AssertThrow(
      terminal_parameter_source == "Parameters" ||
        terminal_parameters_csv_file.empty(),
      ExcMessage(
        "Terminal parameters CSV file requires Terminal parameter source = Parameters."));


    AssertThrow(initial_state_source == "rest" ||
                  initial_state_source == "steady network",
                ExcMessage(
                  "Initial state source must be rest or steady network."));
    AssertThrow(initial_condition_policy == "native" ||
                  initial_condition_policy == "IDA",
                ExcMessage("Initial condition policy must be native or IDA."));
    AssertThrow(
      std::isfinite(initial_consistency_tolerance) &&
        initial_consistency_tolerance > 0. &&
        initial_consistency_tolerance < 1. &&
        initial_consistency_max_iterations > 0,
      ExcMessage(
        "Initial consistency needs a tolerance in (0,1) and a positive iteration limit."));

    exact_solution.update_constants(par);
    rhs_function.update_constants(par);
    inflow_function.update_constants(par);
    AssertThrow(std::isfinite(inflow_csv_period) && inflow_csv_period >= 0.0,
                ExcMessage(
                  "Inflow CSV period must be finite and nonnegative."));
    AssertThrow(!inflow_csv_file.empty() || inflow_csv_period == 0.0,
                ExcMessage("Inflow CSV period requires an Inflow CSV file."));
    inflow_series = InflowSeries{};
    if (!inflow_csv_file.empty())
      {
        inflow_series.read(inflow_csv_file, inflow_csv_period);
        inflow_series.value(ida_parameters.initial_time);
        inflow_series.value(ida_parameters.final_time);
      }

    if (numerical_flux_type_str == "HLL")
      numerical_flux_type = NumericalFluxType::HLL;
    else if (numerical_flux_type_str == "HLL_HDG")
      numerical_flux_type = NumericalFluxType::HLL_HDG;
    else if (numerical_flux_type_str == "LAX_FRIEDRICHS")
      numerical_flux_type = NumericalFluxType::LAX_FRIEDRICHS;
    else
      AssertThrow(false,
                  ExcMessage("Unknown numerical flux type: " +
                             numerical_flux_type_str));
  }

  template <int dim, int spacedim>
  double
  BloodFlowSystem<dim, spacedim>::prescribed_inflow(const double t)
  {
    inflow_function.set_time(t);
    return inflow_series.empty() ? inflow_function.value(Point<1>(0.)) :
                                   inflow_series.value(t);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::setup()
  {
    setup(MeshInitializer{});
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::setup(const MeshInitializer &initialize_mesh)
  {
    create_triangulation(initialize_mesh);
    setup_system();
    initialize_terminal_capacitors();
    build_per_cell_mass_inv();
  }

  template <int dim, int spacedim>
  VectorType
  BloodFlowSystem<dim, spacedim>::make_state() const
  {
    return VectorType(locally_owned_dofs_, mpi_communicator_);
  }

  template <int dim, int spacedim>
  VectorType
  BloodFlowSystem<dim, spacedim>::make_absolute_tolerances() const
  {
    auto result = make_state();
    result      = velocity_absolute_tolerance;
    std::vector<types::global_dof_index> indices(fe_->n_dofs_per_cell());
    for (const auto &cell : dof_handler_.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          get_system_dof_indices(cell, indices);
          const double reference_area = std::min(compute_a_d_at_face(cell, 0),
                                                 compute_a_d_at_face(cell, 1));
          for (unsigned int i = 0; i < indices.size(); ++i)
            {
              const auto component = fe_->system_to_component_index(i).first;
              if (component == 0 || component == 2)
                result(indices[i]) =
                  area_absolute_tolerance_fraction * reference_area;
            }
        }
    for (const auto index : rcr_dofs_owned)
      result(index) = capacitor_absolute_tolerance;
    result.compress(VectorOperation::insert);
    return result;
  }

  template <int dim, int spacedim>
  std::map<unsigned int, double>
  BloodFlowSystem<dim, spacedim>::maximum_mach_numbers(const VectorType &state)
  {
    update_ghosted_vectors(state);
    std::map<unsigned int, unsigned int> offsets;
    for (const auto &[id, vessel] : vessel_map)
      {
        (void)vessel;
        const unsigned int next = offsets.size();
        offsets.emplace(id, next);
      }
    std::vector<double>     local(offsets.size(), 0.0), global(offsets.size());
    const QGauss<dim>       quadrature(fe_degree + 2);
    FEValues<dim, spacedim> values(*fe_, quadrature, update_values);
    const FEValuesExtractors::Scalar area(0), velocity(1);
    std::vector<double> areas(quadrature.size()), velocities(quadrature.size());
    for (const auto &cell : dof_handler_.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const unsigned int id   = cell->material_id();
          double            &peak = local[offsets.at(id)];
          values.reinit(cell);
          values[area].get_function_values(y_fe_relevant, areas);
          values[velocity].get_function_values(y_fe_relevant, velocities);
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            peak = std::max(peak,
                            std::abs(velocities[q]) /
                              compute_wave_speed(areas[q], cell));
          for (const auto face : cell->face_indices())
            {
              double a, u;
              get_face_trace(y_relevant, cell, face, a, u);
              peak = std::max(
                peak,
                std::abs(u) /
                  compute_wave_speed(a, id, compute_a_d_at_face(cell, face)));
            }
        }
    Utilities::MPI::max(ArrayView<const double>(local),
                        mpi_communicator_,
                        ArrayView<double>(global));
    std::map<unsigned int, double> result;
    for (const auto &[id, offset] : offsets)
      result.emplace(id, global[offset]);
    return result;
  }

  template <int dim, int spacedim>
  bool
  BloodFlowSystem<dim, spacedim>::has_admissible_wall_states(
    const VectorType &state) const
  {
    TimerOutput::Scope timer(computing_timer, "check_wall_domain");
    update_ghosted_vectors(state);
    unsigned int invalid = 0;
    for (const auto i : locally_owned_dofs_)
      if (!std::isfinite(state(i)))
        invalid = 1;
    const auto admissible = [this, &invalid](double       a,
                                             unsigned int vid,
                                             double       reference) {
      if (!std::isfinite(a) || a <= 0.)
        return false;
      a                    = std::max(a, 1e-10);
      const double modulus = a * compute_pressure_derivative(a, vid, reference);
      const bool   valid   = std::isfinite(modulus) && modulus > 0.;
      if (!valid && invalid == 0 && verbosity > 1)
        deallog << "Recoverable wall-domain rejection: vessel " << vid
                << ", A/Aref=" << a / reference << ", A*dP/dA=" << modulus
                << std::endl;
      return valid;
    };
    const QGauss<dim>                quadrature(fe_->tensor_degree() + 1);
    FEValues<dim, spacedim>          values(*fe_,
                                   quadrature,
                                   update_values | update_quadrature_points);
    const FEValuesExtractors::Scalar area(0);
    std::vector<double> areas(quadrature.size()), local(fe_->n_dofs_per_cell());
    std::vector<types::global_dof_index> indices(fe_->n_dofs_per_cell());
    for (const auto &cell : dof_handler_.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto vid = cell->material_id();
          values.reinit(cell);
          get_system_dof_indices(cell, indices);
          state.extract_subvector_to(indices, local);
          values[area].get_function_values_from_local_dof_values(local, areas);
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            if (!admissible(areas[q],
                            vid,
                            compute_a_d_at_point(cell,
                                                 values.quadrature_point(q))))
              invalid = 1;
          for (const auto f : cell->face_indices())
            {
              Point<dim> xi;
              xi[0]            = f;
              double cell_area = 0.;
              for (unsigned int i = 0; i < local.size(); ++i)
                if (fe_->system_to_component_index(i).first == 0)
                  cell_area += local[i] * fe_->shape_value(i, xi);
              const auto   trace     = face_trace_dofs(indices, f);
              const double reference = compute_a_d_at_face(cell, f);
              if (!admissible(cell_area, vid, reference) ||
                  !admissible(state(trace.first), vid, reference))
                invalid = 1;
            }
        }
    return Utilities::MPI::max(invalid, mpi_communicator_) == 0;
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::report_flow_regime(const VectorType &state,
                                                     const double      t)
  {
    if (!write_flow_regime_diagnostics)
      return;
    const auto maxima = maximum_mach_numbers(state);
    collective_csv_action(mpi_communicator_, [&]() {
      if (this_mpi_process == 0 && !output_flow_regime_csv_file.empty())
        {
          if (!flow_regime_log.is_open())
            {
              flow_regime_log.open(
                prepare_csv_path(output_directory,
                                 output_flow_regime_csv_file));
              AssertThrow(flow_regime_log,
                          ExcMessage("Cannot open flow regime CSV."));
              csv_write_row(flow_regime_log,
                            "time_s",
                            "vessel_id",
                            "max_abs_U_over_c");
            }
          for (const auto &[id, peak] : maxima)
            flow_regime_log << std::setprecision(17) << t << ',' << id << ','
                            << peak << '\n';
          flow_regime_log.flush();
          AssertThrow(flow_regime_log.good(),
                      ExcMessage("Cannot write flow regime CSV."));
        }
    });
    for (const auto &[id, peak] : maxima)
      {
        if (peak >= mach_warning_threshold)
          {
            if (mach_warning_active.insert(id).second)
              pcout << "WARNING: vessel " << id << " reaches |U|/c=" << peak
                    << " at t=" << t
                    << "; subsonic trace equations approach a sonic fold.\n";
          }
        else
          mach_warning_active.erase(id);
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::initialize_iterative_preconditioner(
    IterativePreconditioner &preconditioner,
    const MatrixType        &matrix) const
  {
#ifdef METRIC_FLOW_X_WITH_PETSC
    preconditioner.initialize(matrix);
    // A local DAE block may have zero pivots even when the global matrix is
    // invertible. Shift the local ILU factors, without changing the operator.
    const auto check = [](const PetscErrorCode error) {
      AssertThrow(error == 0,
                  ExcMessage("PETSc preconditioner setup error " +
                             std::to_string(error)));
    };
    const PC     pc            = preconditioner.get_pc();
    PetscBool    explicit_type = PETSC_FALSE;
    const char  *prefix        = nullptr;
    PetscOptions options       = nullptr;
    check(
      PetscObjectGetOptionsPrefix(reinterpret_cast<PetscObject>(pc), &prefix));
    check(PetscObjectGetOptions(reinterpret_cast<PetscObject>(pc), &options));
    check(PetscOptionsHasName(options, prefix, "-pc_type", &explicit_type));
    if (!explicit_type && n_mpi_processes > 1)
      {
        // A rank-local DAE block can be singular even when the global
        // Jacobian is invertible. Include neighboring equations before ILU.
        check(PCSetType(pc, PCASM));
        check(PCASMSetOverlap(pc, 1));
        check(PCSetFromOptions(pc));
      }
    PCType type;
    check(PCGetType(pc, &type));
    const std::string name(type);
    if (name != PCBJACOBI && name != PCASM)
      return; // Respect a different preconditioner explicitly selected by PETSc
              // options.
    check(PCSetUp(pc));
    PetscInt count;
    KSP     *subsolvers;
    if (name == PCASM)
      check(PCASMGetSubKSP(pc, &count, nullptr, &subsolvers));
    else
      check(PCBJacobiGetSubKSP(pc, &count, nullptr, &subsolvers));
    for (PetscInt i = 0; i < count; ++i)
      {
        PC subpc;
        check(KSPGetPC(subsolvers[i], &subpc));
        check(PCGetType(subpc, &type));
        if (std::string(type) == PCILU || std::string(type) == PCLU)
          {
            Mat      local_matrix;
            PetscInt rows, columns;
            check(KSPGetOperators(subsolvers[i], nullptr, &local_matrix));
            check(MatGetSize(local_matrix, &rows, &columns));
            if (rows == 0 && columns == 0)
              {
                // PETSc 3.19 ILU symbolic factorization divides by zero when
                // computing fill ratios for empty subdomains. Their action is
                // already empty: no factorization or shift is needed.
                check(PCSetType(subpc, PCNONE));
                continue;
              }
            check(PCFactorSetShiftType(subpc, MAT_SHIFT_NONZERO));
            // A near-zero shift can leave singular local DAE blocks with
            // enormous inverse entries and cause GMRES breakdown.
            check(PCFactorSetShiftAmount(subpc, 1e-5));
            // The coupled vessel equations need fill beyond their original
            // sparsity pattern for a useful local approximation.
            if (n_mpi_processes > 1 && std::string(type) == PCILU)
              check(PCFactorSetLevels(subpc, 1));
            check(PCSetFromOptions(subpc));
          }
      }
#else
    // Shift only the local ILU factors: algebraic DAE rows can have zero
    // diagonal entries even when the global Jacobian is invertible.
    typename IterativePreconditioner::AdditionalData data;
    data.ilu_atol = 1e-5;
    data.ilu_fill = n_mpi_processes > 1 ? 1 : 0;
    data.overlap  = 1;
    preconditioner.initialize(matrix, data);
#endif
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::reinit_state(VectorType &state) const
  {
    state.reinit(locally_owned_dofs_, mpi_communicator_);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::initialize_state(VectorType  &state,
                                                   const double t0)
  {
    reinit_state(state);
    compute_initial_solution(state, t0);
    initialize_trace_unknowns(state, t0);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::initialize_state_derivative(
    VectorType &state_dot,
    const double /*t0*/) const
  {
    reinit_state(state_dot);
    state_dot = 0.0;
  }

  template <int dim, int spacedim>
  parallel::fullydistributed::Triangulation<dim, spacedim> &
  BloodFlowSystem<dim, spacedim>::triangulation()
  {
    return triangulation_;
  }

  template <int dim, int spacedim>
  const parallel::fullydistributed::Triangulation<dim, spacedim> &
  BloodFlowSystem<dim, spacedim>::triangulation() const
  {
    return triangulation_;
  }

  template <int dim, int spacedim>
  const DoFHandler<dim, spacedim> &
  BloodFlowSystem<dim, spacedim>::dof_handler() const
  {
    return dof_handler_;
  }

  template <int dim, int spacedim>
  const FiniteElement<dim, spacedim> &
  BloodFlowSystem<dim, spacedim>::finite_element() const
  {
    AssertThrow(fe_ != nullptr, ExcMessage("BloodFlowSystem is not set up."));
    return *fe_;
  }

  template <int dim, int spacedim>
  const AffineConstraints<double> &
  BloodFlowSystem<dim, spacedim>::constraints() const
  {
    return constraints_;
  }

  template <int dim, int spacedim>
  MPI_Comm
  BloodFlowSystem<dim, spacedim>::mpi_communicator() const
  {
    return mpi_communicator_;
  }

  template <int dim, int spacedim>
  const IndexSet &
  BloodFlowSystem<dim, spacedim>::locally_owned_dofs() const
  {
    return locally_owned_dofs_;
  }

  template <int dim, int spacedim>
  const IndexSet &
  BloodFlowSystem<dim, spacedim>::locally_relevant_dofs() const
  {
    return locally_relevant_dofs_;
  }

  template <int dim, int spacedim>
  const IndexSet &
  BloodFlowSystem<dim, spacedim>::differential_dofs() const
  {
    return differential_dofs_;
  }

  template <int dim, int spacedim>
  const IndexSet &
  BloodFlowSystem<dim, spacedim>::algebraic_dofs() const
  {
    return algebraic_dofs_;
  }

  template <int dim, int spacedim>
  const IndexSet &
  BloodFlowSystem<dim, spacedim>::component_dofs(
    const Component component) const
  {
    return component_dofs_[static_cast<unsigned int>(component)];
  }

  template <int dim, int spacedim>
  FEValuesExtractors::Scalar
  BloodFlowSystem<dim, spacedim>::area_extractor() const
  {
    return FEValuesExtractors::Scalar(0);
  }

  template <int dim, int spacedim>
  FEValuesExtractors::Scalar
  BloodFlowSystem<dim, spacedim>::velocity_extractor() const
  {
    return FEValuesExtractors::Scalar(1);
  }

  // ============================================================================
  // create_triangulation
  //
  // Every rank reads the VTK file and builds the coarse mesh serially, then
  // partitions it and keeps only its own part plus a ghost layer.  For a 1-D
  // network the coarse mesh and the per-vessel data are small; what has to
  // scale is the refined mesh, and that is what the fully distributed
  // Triangulation stores.  If the coarse mesh ever stops fitting in one rank's
  // memory, this is the function to replace with
  // create_description_from_triangulation_in_groups().
  // ============================================================================
  template <int dim, int spacedim>
  std::map<types::boundary_id,
           typename BloodFlowSystem<dim, spacedim>::RCRPhysics>
  BloodFlowSystem<dim, spacedim>::read_terminal_parameters(
    const std::set<types::boundary_id> &terminal_ids) const
  {
    const auto validate = [](const RCRPhysics &rcr) {
      AssertThrow(
        std::isfinite(rcr.R1) && std::isfinite(rcr.R2) &&
          std::isfinite(rcr.C) && std::isfinite(rcr.P_out) && rcr.R1 >= 0.0 &&
          rcr.R2 >= 0.0 && rcr.C >= 0.0 && rcr.R1 + rcr.R2 > 0.0 &&
          std::isfinite(rcr.R1 + rcr.R2) && (rcr.C == 0.0 || rcr.R2 > 0.0),
        ExcMessage(
          "Terminal RCR values must be finite, resistances and compliance nonnegative, total resistance positive, and R2 positive when C > 0."));
    };
    RCRPhysics defaults;
    defaults.R1    = default_terminal_r1;
    defaults.R2    = default_terminal_r2;
    defaults.C     = default_terminal_c;
    defaults.P_out = default_terminal_p_out;
    validate(defaults);
    std::map<types::boundary_id, RCRPhysics> result;
    for (const auto id : terminal_ids)
      {
        auto value  = defaults;
        value.label = "boundary_" + std::to_string(id);
        result.emplace(id, value);
      }
    if (terminal_parameters_csv_file.empty())
      return result;
    rapidcsv::SeparatorParams separators(',', true);
    separators.mQuotedLinebreaks = true;
    const rapidcsv::Document csv(terminal_parameters_csv_file,
                                 rapidcsv::LabelParams(),
                                 separators);
    const auto               columns = csv.GetColumnNames();
    std::set<std::string>    column_names;
    for (const auto &name : columns)
      AssertThrow(!name.empty() && column_names.insert(name).second,
                  ExcMessage(
                    "Terminal CSV column names must be nonempty and unique."));
    AssertThrow(csv.GetColumnIdx("boundary_id") >= 0,
                ExcMessage("Terminal CSV requires a boundary_id column."));
    AssertThrow(
      csv.GetColumnIdx("R1") >= 0 || csv.GetColumnIdx("R2") >= 0 ||
        csv.GetColumnIdx("C") >= 0 || csv.GetColumnIdx("label") >= 0,
      ExcMessage(
        "Terminal CSV requires at least one of R1, R2, C or label; other columns are ignored."));
    const auto parse_value = [](const std::string &text, double &value) {
      std::size_t consumed = 0;
      value                = std::stod(text, &consumed);
      AssertThrow(consumed == text.size(),
                  ExcMessage("Invalid terminal CSV numeric value: " + text));
    };
    std::set<types::boundary_id> seen;
    for (std::size_t row = 0; row < csv.GetRowCount(); ++row)
      {
        const std::string context =
          "Malformed terminal CSV row " + std::to_string(row + 2);
        AssertThrow(csv.GetRow<std::string>(row).size() == columns.size(),
                    ExcMessage(context));
        const auto id_text = csv.GetCell<std::string>("boundary_id", row);
        AssertThrow(
          !id_text.empty() &&
            std::all_of(id_text.begin(),
                        id_text.end(),
                        [](const char c) { return c >= '0' && c <= '9'; }),
          ExcMessage(context + ": boundary_id must be an unsigned integer"));
        const auto raw_id = csv.GetCell<unsigned long long>("boundary_id", row);
        AssertThrow(raw_id <= std::numeric_limits<types::boundary_id>::max(),
                    ExcMessage(context + ": boundary_id is out of range"));
        RCRPhysics value          = defaults;
        const auto override_value = [&](const std::string &name,
                                        double            &target) {
          if (csv.GetColumnIdx(name) >= 0 &&
              !csv.GetCell<std::string>(name, row).empty())
            target = csv.GetCell<double>(name, row, parse_value);
        };
        override_value("R1", value.R1);
        override_value("R2", value.R2);
        override_value("C", value.C);
        const auto id = static_cast<types::boundary_id>(raw_id);
        AssertThrow(result.count(id) != 0,
                    ExcMessage(
                      "Terminal CSV references non-terminal boundary ID " +
                      std::to_string(id)));
        AssertThrow(seen.insert(id).second,
                    ExcMessage("Duplicate boundary ID in terminal CSV: " +
                               std::to_string(id)));
        value.label = result.at(id).label;
        if (csv.GetColumnIdx("label") >= 0 &&
            !csv.GetCell<std::string>("label", row).empty())
          value.label = csv.GetCell<std::string>("label", row);
        validate(value);
        result.at(id) = value;
      }
    return result;
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::create_triangulation()
  {
    create_triangulation(MeshInitializer{});
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::create_triangulation(
    const MeshInitializer &initialize_mesh)
  {
    AssertThrow(triangulation_.n_global_active_cells() == 0,
                ExcMessage(
                  "Cannot initialize an already populated flow mesh."));
    TimerOutput::Scope timer(computing_timer, "create_triangulation");

    Triangulation<dim, spacedim> serial_triangulation;

    VTKUtils::Reader vtk(vtk_file_path);
    vtk.read_vtk(serial_triangulation, false);
    if (verbosity > 0)
      pcout << "Coarse cells = " << serial_triangulation.n_active_cells()
            << std::endl;

    vtk.read_cell_data("vessel_id", cell_vessel_ids);
    vtk.read_cell_data("a0", cell_a0);
    vtk.read_cell_data("a_d", cell_a_d);
    vtk.read_cell_data("E", cell_E);
    vtk.read_cell_data("h_wall", cell_h_wall);
    vtk.read_cell_data("p_d", cell_p_d);
    vtk.read_cell_data("p0", cell_p0);
    vtk.read_cell_data("L", cell_L);
    vtk.read_cell_data("r_d", cell_r_d);

    cell_viscosity.reinit(0);
    if (vtk.has_cell_data("viscosity"))
      {
        vtk.read_cell_data("viscosity", cell_viscosity);
        AssertThrow(cell_viscosity.size() == cell_vessel_ids.size(),
                    ExcMessage(
                      "viscosity must contain one value per VTK cell."));
        std::map<unsigned int, double> seen;
        for (unsigned int i = 0; i < cell_viscosity.size(); ++i)
          {
            const double value = cell_viscosity[i];
            AssertThrow(std::isfinite(value) && value >= 0.,
                        ExcMessage(
                          "VTK viscosity must be finite and nonnegative."));
            const auto [it, inserted] = seen.emplace(cell_vessel_ids[i], value);
            AssertThrow(
              inserted || it->second == value,
              ExcMessage("viscosity differs between cells sharing vessel_id."));
          }
      }

    wall_input.clear();
    if (wall_model != "Square root")
      {
        const std::string context =
          "Wall model " + wall_model + " in " + vtk_file_path + ": ";
        const auto require_field = [&](const std::string &name) {
          AssertThrow(vtk.has_cell_data(name),
                      ExcMessage(context +
                                 "missing required CELL_DATA array '" + name +
                                 "'."));
          vtk.read_cell_data(name, wall_input[name]);
          AssertThrow(wall_input[name].size() == cell_vessel_ids.size(),
                      ExcMessage(context + name +
                                 " has the wrong number of entries."));
        };
        if (wall_model == "ADAVN mixed")
          require_field("adavn_wall_type");
        else if (vtk.has_cell_data("adavn_wall_type"))
          require_field("adavn_wall_type");
        bool artery = false, vein = false;
        for (unsigned int i = 0; i < cell_vessel_ids.size(); ++i)
          {
            const double type = wall_input.count("adavn_wall_type") ?
                                  wall_input.at("adavn_wall_type")[i] :
                                  (wall_model == "ADAVN arterial" ? 1. : 2.);
            AssertThrow(
              type == 1. || type == 2. || type == 3.,
              ExcMessage(
                context +
                "adavn_wall_type must be 1 (artery), 2 (vein), or 3 (dural sinus), cell " +
                std::to_string(i)));
            AssertThrow(wall_model != "ADAVN arterial" || type == 1.,
                        ExcMessage(context +
                                   "venous cell supplied to arterial model."));
            AssertThrow(wall_model != "ADAVN venous" || type != 1.,
                        ExcMessage(context +
                                   "arterial cell supplied to venous model."));
            artery |= type == 1.;
            vein |= type != 1.;
          }
        require_field("adavn_p_ref");
        if (artery)
          for (const auto &name :
               {"adavn_E_e", "adavn_E_c", "adavn_epsilon_0", "adavn_epsilon_r"})
            require_field(name);
        if (vein)
          for (const auto &name :
               {"adavn_venous_K", "adavn_venous_m", "adavn_venous_n"})
            require_field(name);
        if (viscoelastic_wall)
          require_field("adavn_K_m");
        std::map<unsigned int, unsigned int> first_cell;
        for (unsigned int i = 0; i < cell_vessel_ids.size(); ++i)
          {
            const double id = cell_vessel_ids[i];
            AssertThrow(
              std::isfinite(id) && id >= 0. && id == std::floor(id) &&
                id < std::numeric_limits<types::material_id>::max(),
              ExcMessage(
                context +
                "vessel_id must be a nonnegative representable integer at cell " +
                std::to_string(i)));
            const auto [it, inserted] =
              first_cell.emplace(static_cast<unsigned int>(id), i);
            if (!inserted)
              {
                AssertThrow(
                  cell_a_d[i] == cell_a_d[it->second],
                  ExcMessage(
                    context + "a_d differs between cells sharing vessel_id " +
                    std::to_string(it->first) +
                    "; split vessel IDs or use the supported taper arrays."));
                const bool arterial =
                  wall_input.count("adavn_wall_type") ?
                    wall_input.at("adavn_wall_type")[i] == 1. :
                    wall_model == "ADAVN arterial";
                AssertThrow(
                  !arterial || cell_h_wall[i] == cell_h_wall[it->second],
                  ExcMessage(
                    context +
                    "h_wall differs between arterial cells sharing vessel_id " +
                    std::to_string(it->first) +
                    "; use distinct vessel IDs for distinct wall thickness."));
              }
            if (!inserted)
              for (const auto &[name, values] : wall_input)
                AssertThrow(
                  values[i] == values[it->second],
                  ExcMessage(
                    context + name +
                    " differs between cells sharing vessel_id " +
                    std::to_string(it->first) +
                    "; use distinct vessel IDs for distinct wall properties."));
            const auto value = [&](const std::string &name) {
              const double x = wall_input.at(name)[i];
              AssertThrow(std::isfinite(x),
                          ExcMessage(context + name +
                                     " is not finite at cell " +
                                     std::to_string(i)));
              return x;
            };
            const double type = wall_input.count("adavn_wall_type") ?
                                  value("adavn_wall_type") :
                                  (wall_model == "ADAVN arterial" ? 1. : 2.);
            AssertThrow(std::isfinite(cell_a_d[i]) && cell_a_d[i] > 0.,
                        ExcMessage(context +
                                   "a_d must be finite and positive."));
            value("adavn_p_ref");
            if (viscoelastic_wall)
              AssertThrow(value("adavn_K_m") >= 0.,
                          ExcMessage(context +
                                     "adavn_K_m must be nonnegative."));
            if (type == 1.)
              AssertThrow(
                value("adavn_E_e") > 0. && value("adavn_E_c") >= 0. &&
                  value("adavn_epsilon_0") >= 0. &&
                  value("adavn_epsilon_r") > 0. &&
                  std::isfinite(cell_h_wall[i]) && cell_h_wall[i] > 0.,
                ExcMessage(
                  context +
                  "arterial moduli, activation width, or wall thickness are invalid at cell " +
                  std::to_string(i)));
            else
              AssertThrow(
                value("adavn_venous_K") > 0. && value("adavn_venous_m") > 0. &&
                  value("adavn_venous_n") <= 0.,
                ExcMessage(
                  context +
                  "venous K and m must be positive and n nonpositive at cell " +
                  std::to_string(i)));
          }
      }

    if (write_flow_regime_diagnostics)
      for (unsigned int i = 0; i < cell_a_d.size(); ++i)
        {
          const double from_radius = numbers::PI * cell_r_d[i] * cell_r_d[i];
          if (std::abs(cell_a_d[i] - from_radius) > 0.02 * from_radius)
            pcout << "WARNING: vessel " << cell_vessel_ids[i]
                  << " has a_d inconsistent with pi*r_d^2 by more than 2%.\n";
        }


    // Missing taper arrays are optional; malformed arrays still report errors.
    cell_r_in.reinit(0);
    cell_r_out.reinit(0);
    if (vtk.has_cell_data("r_in"))
      vtk.read_cell_data("r_in", cell_r_in);
    if (vtk.has_cell_data("r_out"))
      vtk.read_cell_data("r_out", cell_r_out);

    if (terminal_parameter_source == "VTK")
      {
        vtk.read_vertex_data("R1", point_R1);
        vtk.read_vertex_data("R2", point_R2);
        vtk.read_vertex_data("C", point_C);
        vtk.read_vertex_data("P_out", point_P_out);
      }
    vtk.read_vertex_data("boundary_id", point_boundary_id);

    std::vector<unsigned int> vertex_degree(serial_triangulation.n_vertices(),
                                            0);
    for (const auto &cell : serial_triangulation.active_cell_iterators())
      for (const auto vertex : cell->vertex_indices())
        ++vertex_degree[cell->vertex_index(vertex)];

    // Material IDs + boundary IDs from VTK.  These are attributes of the coarse
    // cells and are inherited by their children through refinement and carried
    // into the distributed description, so they are available on every rank
    // afterwards without any communication.
    {
      unsigned int cell_idx = 0;
      for (auto &cell : serial_triangulation.active_cell_iterators())
        {
          cell->set_material_id(
            static_cast<unsigned int>(cell_vessel_ids[cell_idx]));
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->face(f)->at_boundary())
              {
                const unsigned int v = cell->face(f)->vertex_index(0);
                cell->face(f)->set_boundary_id(
                  static_cast<types::boundary_id>(point_boundary_id[v]));
              }
          ++cell_idx;
        }
    }

    // RCR map, keyed by boundary id: global data, identical on every rank
    // because every rank reads the same file.
    if (terminal_parameter_source == "Parameters")
      {
        std::set<types::boundary_id> terminals;
        for (const auto &cell : serial_triangulation.active_cell_iterators())
          for (const auto face : cell->face_indices())
            if (cell->face(face)->at_boundary() &&
                vertex_degree[cell->face(face)->vertex_index(0)] == 1)
              {
                const auto id = cell->face(face)->boundary_id();
                if (id != 0)
                  AssertThrow(terminals.insert(id).second,
                              ExcMessage(
                                "Duplicate terminal boundary ID in mesh."));
              }
        rcr_map = read_terminal_parameters(terminals);
      }
    else
      {
        rcr_map.clear();
        for (const auto &cell : serial_triangulation.active_cell_iterators())
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->face(f)->at_boundary())
              {
                const types::boundary_id bid = cell->face(f)->boundary_id();
                if (bid == 0)
                  continue; // inflow
                const unsigned int v = cell->face(f)->vertex_index(0);
                RCRPhysics         rcr;
                rcr.R1    = point_R1[v];
                rcr.R2    = point_R2[v];
                rcr.C     = point_C[v];
                rcr.P_out = point_P_out[v];
                rcr.label = "boundary_" + std::to_string(bid);
                if (rcr.R2 > 0.0)
                  rcr_map[bid] = rcr;
              }
      }


    // Preserve the coarse hydraulic graph before refinement/partitioning.
    // File vertex identities distinguish coincident, disconnected vertices.
    initial_graph_points_.clear();
    std::map<unsigned int, unsigned int> graph_vertex_number;
    for (const auto &cell : serial_triangulation.active_cell_iterators())
      for (const auto f : cell->vertex_indices())
        if (graph_vertex_number
              .emplace(cell->vertex_index(f), initial_graph_points_.size())
              .second)
          initial_graph_points_.push_back(cell->vertex(f));
    initial_graph_cells_.clear();
    initial_graph_boundaries_.clear();
    for (const auto &cell : serial_triangulation.active_cell_iterators())
      {
        initial_graph_cells_.emplace(
          cell->id().get_coarse_cell_id(),
          InitialGraphCell{graph_vertex_number.at(cell->vertex_index(0)),
                           graph_vertex_number.at(cell->vertex_index(1)),
                           cell->material_id()});
        for (const auto f : cell->face_indices())
          if (vertex_degree[cell->vertex_index(f)] == 1)
            initial_graph_boundaries_.emplace(graph_vertex_number.at(
                                                cell->vertex_index(f)),
                                              cell->face(f)->boundary_id());
      }

    vessel_terminals.clear();
    for (const auto &cell : serial_triangulation.active_cell_iterators())
      for (const auto face : cell->face_indices())
        if (cell->face(face)->at_boundary() &&
            vertex_degree[cell->face(face)->vertex_index(0)] == 1)
          {
            const auto id = cell->face(face)->boundary_id();
            if (id != 0 && rcr_map.count(id))
              vessel_terminals[cell->material_id()].insert(id);
          }

    if (initialize_mesh)
      {
        initialize_mesh(serial_triangulation, n_global_refinements);
        AssertThrow(triangulation_.n_global_active_cells() > 0,
                    ExcMessage(
                      "The mesh initializer must populate the flow mesh."));
      }
    else
      {
        serial_triangulation.refine_global(n_global_refinements);

        if (mesh_partitioning == "graph")
          partition_vessels(serial_triangulation);
        else
          GridTools::partition_triangulation_zorder(n_mpi_processes,
                                                    serial_triangulation);

        const auto description = TriangulationDescription::Utilities::
          create_description_from_triangulation(serial_triangulation,
                                                mpi_communicator_);

        triangulation_.create_triangulation(description);
      }
    if (verbosity > 0)
      pcout << "Global active cells = "
            << triangulation_.n_global_active_cells() << std::endl;
    if (verbosity > 0)
      {
        const std::vector<unsigned int> owned =
          Utilities::MPI::gather(mpi_communicator_,
                                 triangulation_.n_locally_owned_active_cells());
        if (this_mpi_process == 0)
          for (unsigned int r = 0; r < owned.size(); ++r)
            pcout << "  rank " << r << " owns " << owned[r] << " cells"
                  << std::endl;
      }
  }

  // ============================================================================
  // build_global_vessel_data
  //
  // vessel_map and rcr_map describe the network, not the local part of it: a
  // rank needs the pressure law of every vessel whose cells it can see, and
  // compute_a_d_local() interpolates between a vessel's *global* end radii.  So
  // vessel_map comes from the replicated VTK arrays and the arc-length bounds
  // are reduced over the communicator.  Without the reduction a tapered vessel
  // split across two ranks would get a different a_d on each side of the cut.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_global_vessel_data()
  {
    TimerOutput::Scope timer(computing_timer, "build_global_vessel_data");

    vessel_map.clear();

    unsigned int max_vid = 0;
    for (unsigned int i = 0; i < cell_vessel_ids.size(); ++i)
      max_vid =
        std::max(max_vid, static_cast<unsigned int>(cell_vessel_ids[i]));
    const unsigned int n_vid = max_vid + 1;

    for (unsigned int i = 0; i < cell_vessel_ids.size(); ++i)
      {
        const unsigned int vid = static_cast<unsigned int>(cell_vessel_ids[i]);
        if (vessel_map.count(vid))
          continue;

        const unsigned int data_index = wall_model == "Square root" ? vid : i;
        VesselPhysicalProperties vp;
        vp.a0     = cell_a0[data_index];
        vp.E      = cell_E[data_index];
        vp.h_wall = cell_h_wall[data_index];
        vp.p_d    = cell_p_d[data_index];
        vp.p0     = cell_p0[data_index];
        vp.a_d    = cell_a_d[data_index];
        vp.L      = cell_L[data_index];
        vp.dynamic_viscosity =
          cell_viscosity.size() ? cell_viscosity[data_index] : viscosity;
        vp.r_d  = cell_r_d[data_index];
        vp.r_in = (cell_r_in.size() > data_index) ? cell_r_in[data_index] : 0.0;
        vp.r_out =
          (cell_r_out.size() > data_index) ? cell_r_out[data_index] : 0.0;
        vp.beta_p = compute_beta_p(vp.E, vp.h_wall);
        if (wall_model != "Square root")
          {
            const unsigned int type =
              wall_input.count("adavn_wall_type") ?
                static_cast<unsigned int>(wall_input.at("adavn_wall_type")[i]) :
                (wall_model == "ADAVN arterial" ? 1u : 2u);
            vp.wall.type               = static_cast<WallLaw::Type>(type);
            vp.wall.reference_pressure = wall_input.at("adavn_p_ref")[i];
            vp.wall.km = viscoelastic_wall ? wall_input.at("adavn_K_m")[i] : 0.;
            if (type == 1)
              {
                vp.wall.ee       = wall_input.at("adavn_E_e")[i];
                vp.wall.ec       = wall_input.at("adavn_E_c")[i];
                vp.wall.epsilon0 = wall_input.at("adavn_epsilon_0")[i];
                vp.wall.epsilonr = wall_input.at("adavn_epsilon_r")[i];
              }
            else
              {
                vp.wall.stiffness = wall_input.at("adavn_venous_K")[i];
                vp.wall.m         = wall_input.at("adavn_venous_m")[i];
                vp.wall.n         = wall_input.at("adavn_venous_n")[i];
              }
            AssertThrow(vp.wall.elastic(vp.a_d, vp.a_d, vp.h_wall).first > 0.,
                        ExcMessage(
                          "Non-hyperbolic ADAVN reference state in vessel " +
                          std::to_string(vid)));
          }


        vessel_map[vid] = vp;
      }

    // Arc-length bounds: local extrema first, then a global min/max so that
    // every rank ends up with the same [s_min, s_max] per vessel.
    std::vector<double> s_min(n_vid, std::numeric_limits<double>::max());
    std::vector<double> s_max(n_vid, std::numeric_limits<double>::lowest());

    for (const auto &cell : triangulation_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        const unsigned int vid = cell->material_id();

        const Tensor<1, spacedim> d_hat =
          (cell->vertex(1) - cell->vertex(0)) /
          cell->vertex(1).distance(cell->vertex(0));

        const double s0 = cell->vertex(0) * d_hat;
        const double s1 = cell->vertex(1) * d_hat;

        s_min[vid] = std::min(s_min[vid], std::min(s0, s1));
        s_max[vid] = std::max(s_max[vid], std::max(s0, s1));
      }

    std::vector<double> global_s_min(n_vid);
    std::vector<double> global_s_max(n_vid);

    MPI_Allreduce(s_min.data(),
                  global_s_min.data(),
                  n_vid,
                  MPI_DOUBLE,
                  MPI_MIN,
                  mpi_communicator_);

    MPI_Allreduce(s_max.data(),
                  global_s_max.data(),
                  n_vid,
                  MPI_DOUBLE,
                  MPI_MAX,
                  mpi_communicator_);

    s_min.swap(global_s_min);
    s_max.swap(global_s_max);
    vessel_s_bounds.clear();
    for (const auto &[vid, vp] : vessel_map)
      {
        (void)vp;
        if (s_min[vid] <= s_max[vid])
          vessel_s_bounds[vid] = std::make_pair(s_min[vid], s_max[vid]);
      }

    // Terminal boundary ids: derived from rcr_map, which is replicated, so this
    // set is identical on every rank and needs no reduction.
    terminal_boundary_ids.clear();
    for (const auto &[bid, rcr] : rcr_map)
      {
        (void)rcr;
        terminal_boundary_ids.insert(bid);
      }

    if (verbosity > 0)
      {
        pcout << "\n=== vessel arc lengths ===\n  vid   L[mm]\n";
        for (const auto &[vid, b] : vessel_s_bounds)
          pcout << "  " << std::setw(3) << vid << "   " << std::setw(10)
                << (b.second - b.first) * 1000.0 << "\n";
      }
  }

  // ============================================================================
  // detect_junctions
  //
  // Runs over the locally relevant cells (owned + ghost).  The ghost layer is
  // vertex-adjacent, so a rank owning any cell incident to a junction vertex
  // also stores every other cell incident to it: the JunctionInfo it builds is
  // complete, and no rank ever sees a partial junction.
  //
  // The half-faces are sorted by (CellId, face_no) so that every rank that sees
  // a junction orders its vessels identically.  That matters because the row of
  // each junction equation is chosen by position in this list -- without the
  // sort, two ranks could disagree about which row carries mass conservation.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::detect_junctions()
  {
    TimerOutput::Scope timer(computing_timer, "detect_junctions");

    junctions.clear();
    all_junction_faces.clear();

    // vertex -> list of (cell, local_face_no) pairs
    std::map<unsigned int,
             std::vector<std::pair<
               typename DoFHandler<dim, spacedim>::active_cell_iterator,
               unsigned int>>>
      vertex_to_half_faces;

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (cell->is_artificial())
          continue;

        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          vertex_to_half_faces[cell->vertex_index(v)].emplace_back(cell, v);
      }

    unsigned int n_local_junctions = 0;

    for (auto &[v_idx, half_faces] : vertex_to_half_faces)
      {
        std::sort(half_faces.begin(),
                  half_faces.end(),
                  [](const auto &a, const auto &b) {
                    if (a.first->id() != b.first->id())
                      return a.first->id() < b.first->id();
                    return a.second < b.second;
                  });

        const std::size_t n_inc = half_faces.size();

        // Classify by how many cells meet at the vertex:
        //   n_inc == 1                    -> boundary (inlet / terminal).
        //   n_inc == 2, same vessel id    -> ordinary interior face.
        //   n_inc == 2, different ids     -> two vessels joined end to end, an
        //                                    in-line connection, treated as a
        //                                    2-way junction.
        //   n_inc >= 3                    -> junction.
        if (n_inc == 1)
          continue;

        if (n_inc == 2)
          {
            const unsigned int vid_a = half_faces[0].first->material_id();
            const unsigned int vid_b = half_faces[1].first->material_id();
            if (vid_a == vid_b)
              continue;

            if (unify_two_way_junctions)
              {
                if (two_way_pressure_laws_match(vid_a, vid_b))
                  {
                    // Same p(A) on both sides: the K=2 junction conditions
                    // reduce to a single unique trace pair.  Demote to an
                    // interior face, handled by
                    // assemble_trace_interior_equations() plus the continuity
                    // rows like every other valence-<=2 face.
                    continue;
                  }
                pcout
                  << "WARNING: 2-way node between vessels " << vid_a << " and "
                  << vid_b
                  << " has differing pressure laws; keeping the per-vessel\n"
                     "         junction equations (a unique trace there "
                     "would violate total-pressure continuity).\n";
              }
          }

        JunctionInfo J;
        J.location = triangulation_.get_vertices()[v_idx];

        for (const auto &[cell, local_face] : half_faces)
          {
            JunctionHalfFace jhf;
            jhf.cell        = cell;
            jhf.face_no     = local_face;
            jhf.orientation = (local_face == 1) ? 1 : -1;

            J.half_faces.push_back(jhf);
            all_junction_faces.emplace(cell->id(), local_face);
          }

        junctions.push_back(std::move(J));
        ++n_local_junctions;
      }

    const unsigned int n_global_junctions =
      Utilities::MPI::sum(n_local_junctions, mpi_communicator_);
    if (verbosity > 0)
      pcout
        << "Detected " << n_global_junctions
        << " junction half-face groups (counting each junction once per rank "
           "that touches it)."
        << std::endl;
  }

  // ============================================================================
  // canonical_face_key
  //
  // Interior face: key = the half-face whose cell has the smaller CellId.
  // Boundary / junction face: key = the unique owning cell.
  //
  // CellId ordering does not depend on the partition, so all ranks agree on the
  // canonical side of a face that straddles a subdomain boundary.
  // ============================================================================
  template <int dim, int spacedim>
  std::pair<CellId, unsigned int>
  BloodFlowSystem<dim, spacedim>::canonical_face_key(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const unsigned int face_no) const
  {
    if (cell->face(face_no)->at_boundary() ||
        is_junction_face(cell->id(), face_no))
      return {cell->id(), face_no};

    const auto        &nb         = cell->neighbor(face_no);
    const unsigned int nb_face_no = cell->neighbor_of_neighbor(face_no);

    if (cell->id() < nb->id())
      return {cell->id(), face_no};

    return {nb->id(), nb_face_no};
  }

  // ============================================================================
  // build_face_dof_map
  //
  // Records, for every face this rank can reach from one of its locally owned
  // cells, which (A_hat, U_hat) pair is the canonical one.  The indices come
  // straight out of the FESystem, so this map answers "which DoF" and never
  // "whose DoF" -- ownership is read off locally_owned_dofs_ at the point of
  // use.
  //
  // Both sides of an interior face are locally relevant (the ghost layer is
  // vertex-adjacent), so the canonical indices can always be looked up, even
  // when the canonical cell belongs to another rank.
  //
  // The non-canonical side of an ordinary interior face is a duplicate: this
  // rank owns its two rows whenever it owns that cell, regardless of who owns
  // the canonical side, and ties them down with the continuity rows.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_face_dof_map()
  {
    TimerOutput::Scope timer(computing_timer, "build_face_dof_map");
    const auto         owned_system_fe =
      system_index_set(dof_handler_.locally_owned_dofs());

    face_dof_map.clear();
    trace_continuity_pairs.clear();
    interior_faces.clear();
    boundary_faces.clear();

    std::vector<types::global_dof_index> ldofs(fe_->n_dofs_per_cell());
    std::vector<types::global_dof_index> nb_dofs(fe_->n_dofs_per_cell());

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!(cell->is_locally_owned()))
          continue;

        get_system_dof_indices(cell, ldofs);

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            const auto key = canonical_face_key(cell, f);
            // if (cell->face(f)->at_boundary() || is_junction_face(cell->id(),
            // f))
            //   {
            //     std::cout << "Rank " << this_mpi_process
            //               << " inserting junction key " << key.first << " "
            //               << key.second << " owned=" <<
            //               cell->is_locally_owned()
            //               << " ghost=" << cell->is_ghost() << std::endl;
            //   }
            const auto [a_here, u_here] = face_trace_dofs(ldofs, f);

            const bool this_side_is_canonical = (key.first == cell->id());

            if (this_side_is_canonical)
              {
                face_dof_map[key] = FaceTraceDof{a_here, u_here};
              }
            else
              {
                // The canonical side is the neighbour, owned by this rank or a
                // ghost.  Either way its DoF indices are readable.
                const auto        &nb   = cell->neighbor(f);
                const unsigned int nb_f = cell->neighbor_of_neighbor(f);
                get_system_dof_indices(nb, nb_dofs);
                const auto [a_canon, u_canon] = face_trace_dofs(nb_dofs, nb_f);

                face_dof_map[key] = FaceTraceDof{a_canon, u_canon};

                // This cell's own pair is the duplicate side.  Junction and
                // boundary half-faces are never duplicated: each is its own
                // canonical owner, so they never reach this branch.
                if (!cell->face(f)->at_boundary() &&
                    !is_junction_face(cell->id(), f))
                  trace_continuity_pairs.push_back(
                    TraceContinuityPair{a_here, u_here, a_canon, u_canon});
              }
            if (cell->face(f)->at_boundary() &&
                !is_junction_face(cell->id(), f))
              {
                const auto &trace = face_dof_map.at(key);
                boundary_faces.push_back(
                  {cell, f, trace.a_hat_dof, trace.u_hat_dof});
              }
          }
      }

    std::set<FaceKey> processed;
    for (const auto &cell : dof_handler_.active_cell_iterators())
      if (cell->is_locally_owned())
        for (const auto f : cell->face_indices())
          {
            if (cell->face(f)->at_boundary() || is_junction_face(cell->id(), f))
              continue;
            const auto key = canonical_face_key(cell, f);
            if (!processed.insert(key).second)
              continue;
            const auto &trace = face_dof_map.at(key);
            if (!owned_system_fe.is_element(trace.a_hat_dof))
              continue;
            interior_faces.push_back({cell,
                                      cell->neighbor(f),
                                      f,
                                      cell->neighbor_of_neighbor(f),
                                      trace});
          }

    const types::global_dof_index n_pairs =
      Utilities::MPI::sum<types::global_dof_index>(
        trace_continuity_pairs.size(), mpi_communicator_);
    if (verbosity > 0)
      pcout << "Face DOF map: FE dofs = " << dof_handler_.n_dofs()
            << "  continuity pairs = " << n_pairs << std::endl;
  }

  // ==========================================================================
  // build_rcr_dof_map
  //
  // One pressure per terminal with C > 0, owned by the terminal cell's rank.
  // System blocks interleave each rank's FE DoFs and local capacitor pressures;
  // FE evaluation vectors keep the separate native DoFHandler numbering.
  // ==========================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_rcr_dof_map()
  {
    TimerOutput::Scope timer(computing_timer, "build_rcr_dof_map");

    rcr_pc_dof.clear();
    std::vector<types::boundary_id> capacitors;
    if (outlet_type != "Reflection" && outlet_type != "Dirichlet")
      for (const auto &[bid, rcr] : rcr_map)
        if (bid != 0 && rcr.C > 0.)
          capacitors.push_back(bid);
    n_rcr_dofs = capacitors.size();

    // Every capacitor belongs to the unique cell carrying its terminal face.
    std::map<types::boundary_id, unsigned int> local_owners;
    for (const auto &cell : dof_handler_.active_cell_iterators())
      if (cell->is_locally_owned())
        for (const auto f : cell->face_indices())
          if (cell->face(f)->at_boundary())
            local_owners.emplace(cell->face(f)->boundary_id(),
                                 this_mpi_process);
    std::vector<unsigned int> owners(capacitors.size(), n_mpi_processes);
    for (unsigned int i = 0; i < capacitors.size(); ++i)
      if (local_owners.count(capacitors[i]))
        owners[i] = this_mpi_process;
    if (!owners.empty())
      Utilities::MPI::min(owners, mpi_communicator_, owners);

    const auto fe_counts =
      Utilities::MPI::all_gather(mpi_communicator_,
                                 dof_handler_.n_locally_owned_dofs());
    fe_rank_offsets_.assign(n_mpi_processes + 1, 0);
    capacitor_rank_offsets_.assign(n_mpi_processes + 1, 0);
    for (const auto owner : owners)
      {
        AssertThrow(owner < n_mpi_processes,
                    ExcMessage("Capacitor terminal has no owning cell."));
        ++capacitor_rank_offsets_[owner + 1];
      }
    for (unsigned int r = 0; r < n_mpi_processes; ++r)
      {
        fe_rank_offsets_[r + 1] = fe_rank_offsets_[r] + fe_counts[r];
        capacitor_rank_offsets_[r + 1] += capacitor_rank_offsets_[r];
      }
    n_total_dofs = dof_handler_.n_dofs() + n_rcr_dofs;
    // Native FE blocks must remain rank ordered for this offset map.
    const auto &native_owned = dof_handler_.locally_owned_dofs();
    AssertThrow(native_owned.is_contiguous() &&
                  (native_owned.n_elements() == 0 ||
                   native_owned.nth_index_in_set(0) ==
                     fe_rank_offsets_[this_mpi_process]),
                ExcMessage("Expected rank-ordered contiguous FE numbering."));

    rcr_dofs_owned = IndexSet(n_total_dofs);
    auto next_pc   = fe_rank_offsets_;
    for (unsigned int r = 0; r < n_mpi_processes; ++r)
      next_pc[r] = fe_rank_offsets_[r + 1] + capacitor_rank_offsets_[r];
    for (unsigned int i = 0; i < capacitors.size(); ++i)
      {
        const auto index = next_pc[owners[i]]++;
        rcr_pc_dof.emplace(capacitors[i], index);
        if (owners[i] == this_mpi_process)
          rcr_dofs_owned.add_index(index);
      }
    rcr_dofs_owned.compress();

    Assert(Utilities::MPI::sum<types::global_dof_index>(
             rcr_dofs_owned.n_elements(), mpi_communicator_) == n_rcr_dofs,
           ExcMessage(
             "Every capacitor DOF must be owned by exactly one rank."));
    if (verbosity > 0)
      pcout << "RCR capacitor DOFs: n_rcr = " << n_rcr_dofs
            << "  n_total = " << n_total_dofs << std::endl;
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::setup_terminal_connections()
  {
    terminal_links.clear();
    terminal_ports.clear();
    if (terminal_connections_csv_file.empty())
      return;
    AssertThrow(
      outlet_type == "RCR" && !viscoelastic_wall,
      ExcMessage(
        "Terminal connections require RCR boundaries and elastic walls."));
    rapidcsv::Document csv(terminal_connections_csv_file);
    for (const auto &column : {"boundary0", "boundary1", "resistance"})
      AssertThrow(csv.GetColumnIdx(column) >= 0,
                  ExcMessage("Terminal connections CSV requires column " +
                             std::string(column)));
    std::set<std::pair<types::boundary_id, types::boundary_id>> pairs;
    for (size_t row = 0; row < csv.GetRowCount(); ++row)
      {
        const auto   a = csv.GetCell<types::boundary_id>("boundary0", row);
        const auto   b = csv.GetCell<types::boundary_id>("boundary1", row);
        const double resistance = csv.GetCell<double>("resistance", row);
        AssertThrow(a != b && a != 0 && b != 0 && a != 255 && b != 255 &&
                      rcr_map.count(a) && rcr_map.count(b) &&
                      std::isfinite(resistance) && resistance > 0.,
                    ExcMessage("Invalid terminal connection at CSV row " +
                               std::to_string(row + 2)));
        AssertThrow(pairs.emplace(std::min(a, b), std::max(a, b)).second,
                    ExcMessage("Duplicate undirected terminal connection."));
        for (const auto bid : {a, b})
          AssertThrow(
            rcr_map.at(bid).C > 0. || rcr_map.at(bid).R1 == 0.,
            ExcMessage(
              "A connected terminal without a capacitor requires R1=0."));
        terminal_links[a].emplace_back(b, resistance);
        terminal_links[b].emplace_back(a, resistance);
      }
    std::map<types::boundary_id, TerminalPort> local;
    unsigned int                               repeated = 0;
    for (const auto &face : boundary_faces)
      {
        const auto bid = face.cell->face(face.face_no)->boundary_id();
        if (terminal_links.count(bid))
          if (!local
                 .emplace(bid,
                          TerminalPort{face.a_hat_dof,
                                       face.u_hat_dof,
                                       face.cell->material_id(),
                                       compute_a_d_at_face(face.cell,
                                                           face.face_no),
                                       face.face_no == 1 ? 1 : -1,
                                       face.cell->face(face.face_no)->vertex(0),
                                       face.cell->id()})
                 .second)
            repeated = 1;
      }
    AssertThrow(Utilities::MPI::max(repeated, mpi_communicator_) == 0,
                ExcMessage("Repeated connected terminal boundary ID."));
    for (const auto &ports :
         Utilities::MPI::all_gather(mpi_communicator_, local))
      for (const auto &[bid, port] : ports)
        AssertThrow(
          terminal_ports.emplace(bid, port).second,
          ExcMessage(
            "Connected terminal must have exactly one incident cell."));
    AssertThrow(
      terminal_ports.size() == terminal_links.size(),
      ExcMessage(
        "Every connected terminal must be a degree-one mesh endpoint."));
  }

  template <int dim, int spacedim>
  types::global_dof_index
  BloodFlowSystem<dim, spacedim>::terminal_pressure_dof(
    types::boundary_id bid) const
  {
    const auto it = rcr_pc_dof.find(bid);
    return it == rcr_pc_dof.end() ? terminal_ports.at(bid).area : it->second;
  }

  template <int dim, int spacedim>
  double
  BloodFlowSystem<dim, spacedim>::terminal_pressure_derivative(
    types::boundary_id bid,
    const VectorType  &y) const
  {
    if (rcr_pc_dof.count(bid))
      return 1.;
    const auto &port = terminal_ports.at(bid);
    return compute_pressure_derivative(std::max(y(port.area), 1e-10),
                                       port.vessel,
                                       port.reference_area);
  }

  template <int dim, int spacedim>
  double
  BloodFlowSystem<dim, spacedim>::terminal_pressure(types::boundary_id bid,
                                                    const VectorType  &y,
                                                    double             t) const
  {
    const auto it = rcr_pc_dof.find(bid);
    if (it != rcr_pc_dof.end())
      return y(it->second);
    const auto &port = terminal_ports.at(bid);
    return compute_physical_pressure(std::max(y(port.area), 1e-10),
                                     port.vessel,
                                     port.reference_area,
                                     t,
                                     port.point,
                                     port.cell_id);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_terminal_connection_sparsity(
    DynamicSparsityPattern &dsp)
  {
    for (const auto &face : boundary_faces)
      {
        const auto bid = face.cell->face(face.face_no)->boundary_id();
        if (!terminal_links.count(bid))
          continue;
        const auto row = terminal_pressure_dof(bid);
        for (const auto &[other, resistance] : terminal_links.at(bid))
          dsp.add(row, terminal_pressure_dof(other));
      }
  }

  // ============================================================================
  // update_ghosted_vectors
  //
  // The single point where the solver communicates during assembly: copy the
  // locally owned `y` into the two ghosted read vectors.  Everything that
  // follows reads from these and writes only owned rows, so no assembly routine
  // needs a reduction of its own.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::update_ghosted_vectors(
    const VectorType &y) const
  {
    TimerOutput::Scope timer(computing_timer, "update_ghosted_vectors");


    y_relevant                = y;
    const auto indices        = locally_owned_fe_dofs.get_index_vector();
    auto       system_indices = indices;
    for (auto &index : system_indices)
      index = system_dof_index(index);
    std::vector<double> values(indices.size());
    y.extract_subvector_to(system_indices, values);
    y_fe_owned.set(indices, values);
    y_fe_owned.compress(VectorOperation::insert);
    y_fe_relevant = y_fe_owned;
  }

  // ============================================================================
  // get_face_trace
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::get_face_trace(
    const VectorType                                               &y,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const unsigned int                                              face_no,
    double                                                         &A_hat,
    double                                                         &U_hat) const
  {
    const auto key = canonical_face_key(cell, face_no);

    const auto it = face_dof_map.find(key);
    Assert(it != face_dof_map.end(),
           ExcMessage("Face not found in face_dof_map."));

    A_hat = y(it->second.a_hat_dof);
    U_hat = y(it->second.u_hat_dof);
  }

  // ============================================================================
  // Sparsity
  //
  // Every builder below mirrors, one for one, the ownership guards used by the
  // matching assembly routine: a rank inserts a row into the pattern exactly
  // when it will later write that row.  Keeping the two in step is what allows
  // jacobian_matrix.reinit() to be given locally_owned_dofs_ as its row map.
  // ============================================================================

  // Cell rows: volume terms and the face flux against this cell's own trace.
  // The neighbour columns are kept so the pattern also covers the trace rows'
  // dependence on the two incident cells, assembled in build_trace_sparsity().
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_cell_sparsity(
    DynamicSparsityPattern &dsp)
  {
    std::vector<types::global_dof_index> ldofs(fe_->n_dofs_per_cell());
    std::vector<types::global_dof_index> nb_dofs(fe_->n_dofs_per_cell());

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        get_system_dof_indices(cell, ldofs);

        for (const auto i : ldofs)
          for (const auto j : ldofs)
            dsp.add(i, j);

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (cell->face(f)->at_boundary())
              continue;

            get_system_dof_indices(cell->neighbor(f), nb_dofs);
            for (const auto i : ldofs)
              for (const auto j : nb_dofs)
                dsp.add(i, j);
          }
      }
  }

  // Trace rows of interior and boundary faces: coupled to their own pair and to
  // the cell DoFs on both sides of the face.
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_trace_sparsity(
    DynamicSparsityPattern &dsp)
  {
    std::vector<types::global_dof_index> cell_dofs(fe_->n_dofs_per_cell());
    std::vector<types::global_dof_index> nb_dofs(fe_->n_dofs_per_cell());

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        get_system_dof_indices(cell, cell_dofs);

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            const auto it = face_dof_map.find(canonical_face_key(cell, f));
            if (it == face_dof_map.end())
              continue;

            const auto a_hat = it->second.a_hat_dof;
            const auto u_hat = it->second.u_hat_dof;

            // Rows belonging to a canonical side on another rank are that
            // rank's business.
            if (!locally_owned_dofs_.is_element(a_hat))
              continue;

            dsp.add(a_hat, a_hat);
            dsp.add(a_hat, u_hat);
            dsp.add(u_hat, a_hat);
            dsp.add(u_hat, u_hat);

            for (const auto ci : cell_dofs)
              {
                dsp.add(a_hat, ci);
                dsp.add(u_hat, ci);
              }

            if (!cell->face(f)->at_boundary())
              {
                get_system_dof_indices(cell->neighbor(f), nb_dofs);
                for (const auto ni : nb_dofs)
                  {
                    dsp.add(a_hat, ni);
                    dsp.add(u_hat, ni);
                  }
              }
          }
      }
  }

  // Junction rows: fully coupled among the 2K trace unknowns, plus the cell
  // DoFs feeding each vessel's Riemann invariant.
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_junction_sparsity(
    DynamicSparsityPattern &dsp)
  {
    std::vector<types::global_dof_index> cell_dofs(fe_->n_dofs_per_cell());

    for (const auto &J : junctions)
      {
        const unsigned int K = J.n_vessels();

        std::vector<types::global_dof_index> a_row(K), u_row(K);
        std::vector<types::global_dof_index> ldofs(fe_->n_dofs_per_cell());

        for (unsigned int i = 0; i < K; ++i)
          {
            get_system_dof_indices(J.half_faces[i].cell, ldofs);

            const auto [a_hat_dof, u_hat_dof] =
              face_trace_dofs(ldofs, J.half_faces[i].face_no);

            a_row[i] = a_hat_dof;
            u_row[i] = u_hat_dof;
          }
        for (unsigned int i = 0; i < K; ++i)
          for (const auto row : {a_row[i], u_row[i]})
            {
              if (!locally_owned_dofs_.is_element(row))
                continue;

              for (unsigned int j = 0; j < K; ++j)
                {
                  dsp.add(row, a_row[j]);
                  dsp.add(row, u_row[j]);

                  get_system_dof_indices(J.half_faces[j].cell, cell_dofs);
                  for (const auto cj : cell_dofs)
                    dsp.add(row, cj);
                }
            }
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_rcr_sparsity(
    DynamicSparsityPattern &dsp)
  {
    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (!cell->face(f)->at_boundary())
              continue;
            if (is_junction_face(cell->id(), f))
              continue;

            const auto pit = rcr_pc_dof.find(cell->face(f)->boundary_id());
            if (pit == rcr_pc_dof.end())
              continue;

            const auto it = face_dof_map.find(canonical_face_key(cell, f));
            if (it == face_dof_map.end())
              continue;

            const auto pc = pit->second;

            // Both the Pc row and the boundary trace rows are owned here: the
            // terminal face has this cell as its only neighbour.
            dsp.add(pc, pc);
            dsp.add(pc, it->second.a_hat_dof);
            dsp.add(pc, it->second.u_hat_dof);
            dsp.add(it->second.a_hat_dof, pc);
          }
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_trace_continuity_sparsity(
    DynamicSparsityPattern &dsp)
  {
    // The duplicate side is always a locally owned cell's own pair, so these
    // rows are owned by construction.
    for (const auto &p : trace_continuity_pairs)
      {
        dsp.add(p.a_dup, p.a_dup);
        dsp.add(p.a_dup, p.a_canon);
        dsp.add(p.u_dup, p.u_dup);
        dsp.add(p.u_dup, p.u_canon);
      }
  }

  // ============================================================================
  // build_extended_sparsity_pattern
  //
  // The pattern is assembled locally over the relevant rows, exchanged with
  // SparsityTools::distribute_sparsity_pattern() so that each rank learns the
  // off-rank entries written into rows it owns, and then handed to the
  // matrices with locally_owned_dofs_ as the row map.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_extended_sparsity_pattern()
  {
    TimerOutput::Scope timer(computing_timer,
                             "build_extended_sparsity_pattern");
    derivative_jacobian_is_current_ = false;

    DynamicSparsityPattern dsp(n_total_dofs,
                               n_total_dofs,
                               locally_relevant_dofs_);

    build_cell_sparsity(dsp);
    build_trace_sparsity(dsp);
    build_junction_sparsity(dsp);
    build_rcr_sparsity(dsp);
    build_trace_continuity_sparsity(dsp);
    build_terminal_connection_sparsity(dsp);

    SparsityTools::distribute_sparsity_pattern(dsp,
                                               locally_owned_dofs_,
                                               mpi_communicator_,
                                               locally_relevant_dofs_);

    jacobian_matrix.reinit(locally_owned_dofs_,
                           locally_owned_dofs_,
                           dsp,
                           mpi_communicator_);
    state_jacobian_matrix_.reinit(locally_owned_dofs_,
                                  locally_owned_dofs_,
                                  dsp,
                                  mpi_communicator_);
    derivative_jacobian_matrix_.reinit(locally_owned_dofs_,
                                       locally_owned_dofs_,
                                       dsp,
                                       mpi_communicator_);
    linear_system_matrix.reinit(locally_owned_dofs_,
                                locally_owned_dofs_,
                                dsp,
                                mpi_communicator_);
  }

  // ============================================================================
  // setup_system
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::setup_system()
  {
    TimerOutput::Scope timer(computing_timer, "setup_system");

    if (!fe_)
      fe_ = std::make_unique<FESystem<dim, spacedim>>(
        FE_DGQ<dim, spacedim>(fe_degree),
        2, // comps 0,1 : cell A, U (DG)
        FE_DGQ<dim, spacedim>(1),
        2); // comps 2,3 : trace A_hat, U_hat

    dof_handler_.distribute_dofs(*fe_);

    // DoFRenumbering::component_wise() is deliberately not applied.  It would
    // not produce a global [cell | trace] block layout in parallel -- it
    // renumbers within each rank's owned range -- and nothing needs one: the
    // component index sets below take over every role the old integer ranges
    // played.

    // Global network data first: everything downstream (junction detection via
    // the pressure laws, compute_a_d_local() via the arc-length bounds) reads
    // it.
    build_global_vessel_data();

    if (outlet_type == "RCR")
      AssertThrow(!terminal_boundary_ids.empty(),
                  ExcMessage("No terminal boundaries found anywhere in the "
                             "network."));

    detect_junctions();
    build_rcr_dof_map();
    build_face_dof_map();
    setup_terminal_connections();

    // ---- index sets ---------------------------------------------------------
    locally_owned_fe_dofs = dof_handler_.locally_owned_dofs();

    locally_relevant_fe_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler_);

    locally_owned_dofs_.clear();
    locally_owned_dofs_.set_size(n_total_dofs);
    locally_owned_dofs_.add_indices(system_index_set(locally_owned_fe_dofs));
    locally_owned_dofs_.add_indices(rcr_dofs_owned);
    locally_owned_dofs_.compress();
    AssertThrow(locally_owned_dofs_.is_contiguous(),
                ExcMessage("Coupled system ownership must be contiguous."));
    locally_relevant_dofs_.clear();
    locally_relevant_dofs_.set_size(n_total_dofs);
    locally_relevant_dofs_.add_indices(
      system_index_set(locally_relevant_fe_dofs));
    locally_relevant_dofs_.add_indices(rcr_dofs_owned);

    // A rank reads Pc only for terminals whose cell it owns, but make every Pc
    // it can reach through a ghost relevant as well so that diagnostics and
    // output can read them without a special case.
    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (cell->is_artificial())
          continue;

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (!cell->face(f)->at_boundary())
              continue;
            const auto it = rcr_pc_dof.find(cell->face(f)->boundary_id());
            if (it != rcr_pc_dof.end())
              locally_relevant_dofs_.add_index(it->second);
          }
      }
    // Resistive links can join terminals on distant mesh partitions. Import
    // their endpoint pressures along with the ordinary geometric ghost layer.
    for (const auto &face : boundary_faces)
      {
        const auto bid = face.cell->face(face.face_no)->boundary_id();
        if (terminal_links.count(bid))
          for (const auto &[other, resistance] : terminal_links.at(bid))
            locally_relevant_dofs_.add_index(terminal_pressure_dof(other));
      }
    locally_relevant_dofs_.compress();

    // Component index sets: the differential rows of the DAE (cell A, U and the
    // capacitor pressures) versus the algebraic ones (the traces).  Taken from
    // a ComponentMask, so they are right with or without renumbering.
    {
      const ComponentMask cell_mask({true, true, false, false});
      const ComponentMask trace_mask({false, false, true, true});

      const IndexSet cell_fe = DoFTools::extract_dofs(dof_handler_, cell_mask);
      const IndexSet trace_fe =
        DoFTools::extract_dofs(dof_handler_, trace_mask);

      cell_dofs_owned.clear();
      cell_dofs_owned.set_size(n_total_dofs);
      cell_dofs_owned.add_indices(system_index_set(cell_fe));
      cell_dofs_owned.compress();

      trace_dofs_owned.clear();
      trace_dofs_owned.set_size(n_total_dofs);
      trace_dofs_owned.add_indices(system_index_set(trace_fe));
      trace_dofs_owned.compress();

      const std::array<ComponentMask, 4> component_masks = {
        {ComponentMask({true, false, false, false}),
         ComponentMask({false, true, false, false}),
         ComponentMask({false, false, true, false}),
         ComponentMask({false, false, false, true})}};
      for (unsigned int c = 0; c < component_masks.size(); ++c)
        {
          component_dofs_[c].clear();
          component_dofs_[c].set_size(n_total_dofs);
          component_dofs_[c].add_indices(system_index_set(
            DoFTools::extract_dofs(dof_handler_, component_masks[c])));
          component_dofs_[c].compress();
        }

      differential_dofs_.clear();
      differential_dofs_.set_size(n_total_dofs);
      differential_dofs_.add_indices(cell_dofs_owned);
      differential_dofs_.add_indices(rcr_dofs_owned);
      differential_dofs_.compress();

      algebraic_dofs_ = trace_dofs_owned;
    }

    // ---- sparsity + matrices ------------------------------------------------
    build_extended_sparsity_pattern();

    // ---- vectors ------------------------------------------------------------
    solution.reinit(locally_owned_dofs_, mpi_communicator_);
    solution_dot.reinit(locally_owned_dofs_, mpi_communicator_);
    pressure_.reinit(locally_owned_dofs_, mpi_communicator_);
    residual_F.reinit(locally_owned_dofs_, mpi_communicator_);

    y_relevant.reinit(locally_owned_dofs_,
                      locally_relevant_dofs_,
                      mpi_communicator_);
    y_fe_owned.reinit(locally_owned_fe_dofs, mpi_communicator_);
    y_fe_relevant.reinit(locally_owned_fe_dofs,
                         locally_relevant_fe_dofs,
                         mpi_communicator_);
    AssertThrow(y_fe_relevant.size() == dof_handler_.n_dofs(),
                ExcInternalError());
    if (verbosity > 0)
      pcout << "  cell DoFs (differential): "
            << Utilities::MPI::sum<types::global_dof_index>(
                 cell_dofs_owned.n_elements(), mpi_communicator_)
            << "\n  trace DoFs (algebraic)  : "
            << Utilities::MPI::sum<types::global_dof_index>(
                 trace_dofs_owned.n_elements(), mpi_communicator_)
            << "\n  total DoFs              : " << n_total_dofs << std::endl;
  }

  // ============================================================================
  // initialize_terminal_capacitors
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::initialize_terminal_capacitors()
  {
    terminal_Pc_storage.clear();

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        const unsigned int vid = cell->material_id();

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (!cell->face(f)->at_boundary())
              continue;
            if (is_junction_face(cell->id(), f))
              continue;

            const types::boundary_id bid = cell->face(f)->boundary_id();
            if (bid != 0 && is_terminal_boundary(bid))
              terminal_Pc_storage.try_emplace(
                bid,
                vessel_map.at(vid).wall.type == WallLaw::Type::square_root ?
                  vessel_map.at(vid).p_d :
                  compute_pressure_value(compute_a_d_at_face(cell, f),
                                         vid,
                                         compute_a_d_at_face(cell, f)));
          }
      }
  }

  // ============================================================================
  // compute_initial_solution
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::compute_initial_solution(VectorType  &dst,
                                                           const double t)
  {
    TimerOutput::Scope timer(computing_timer, "compute_initial_solution");
    if (initial_state_source == "steady network")
      {
        compute_steady_network_initial_solution(dst, t);
        return;
      }


    dst = 0.0;

    // ---- cell block and this cell's own trace pair --------------------------
    // Both are DoFs of a locally owned cell, hence owned here, so the whole
    // loop writes with insert semantics and no row is touched twice.
    std::vector<types::global_dof_index> ldofs(fe_->n_dofs_per_cell());

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        get_system_dof_indices(cell, ldofs);

        // Nodal interpolation of a_d(x).  For a linearly tapered radius a_d is
        // quadratic in x, so for fe_degree >= 2 the vessel starts exactly at
        // rest (A = a_d, U = 0 gives p = p0 + p_d everywhere).
        for (unsigned int i = 0; i < fe_->n_dofs_per_cell(); ++i)
          {
            const unsigned int comp = fe_->system_to_component_index(i).first;
            dst(ldofs[i]) =
              (comp == 0) ? compute_a_d_at_point(cell, support_point(cell, i)) :
                            0.0;
          }

        // Trace: seed A_hat with the diastolic area at that face and U_hat with
        // zero, which is consistent for every face type (inflow, terminal,
        // interior and junction all start from a flux A_hat*U_hat*bn = 0).
        // Seeding both sides of an interior face from the same rule also starts
        // the continuity residual at zero.
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            const auto [a_here, u_here] = face_trace_dofs(ldofs, f);
            dst(a_here)                 = compute_a_d_at_face(cell, f);
            dst(u_here)                 = 0.0;
          }
      }

    // ---- capacitor block ----------------------------------------------------
    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (!cell->face(f)->at_boundary())
              continue;
            if (is_junction_face(cell->id(), f))
              continue;

            const auto pit = rcr_pc_dof.find(cell->face(f)->boundary_id());
            if (pit == rcr_pc_dof.end())
              continue;

            dst(pit->second) =
              vessel_map.at(cell->material_id()).wall.type ==
                  WallLaw::Type::square_root ?
                vessel_map.at(cell->material_id()).p_d :
                compute_pressure_value(compute_a_d_at_face(cell, f),
                                       cell->material_id(),
                                       compute_a_d_at_face(cell, f));
          }
      }

    dst.compress(VectorOperation::insert);
  }

  // ============================================================================
  // initialize_trace_unknowns
  //
  // Newton on the algebraic rows only, with the cell and capacitor rows pinned
  // by an identity so the same distributed matrix and solver can be reused.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::initialize_trace_unknowns(VectorType  &sol,
                                                            const double t)
  {
    correct_initial_traces(sol, t);
  }

  // ============================================================================
  // build_per_cell_mass_inv
  //
  // For every locally owned cell K:
  //   per_cell_mass     : M_K, used for the M*ydot term of the residual and the
  //                       alpha*M term of the Jacobian
  //   per_cell_mass_inv : M_K^{-1}, initial derivative reconstruction
  //
  // Indexed by cell->active_cell_index(): deal.II numbers the locally stored
  // active cells contiguously from zero, so the vectors are sized to the local
  // cell count and no CellId comparison happens in the residual inner loop.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_per_cell_mass_inv()
  {
    TimerOutput::Scope timer(computing_timer, "build_per_cell_mass_inv");
    derivative_jacobian_is_current_ = false;

    const unsigned int n_local_cells = triangulation_.n_active_cells();
    per_cell_mass.assign(n_local_cells, FullMatrix<double>());
    per_cell_mass_inv.assign(n_local_cells, FullMatrix<double>());

    const QGauss<dim>       quad(fe_degree + 1);
    FEValues<dim, spacedim> fev(*fe_, quad, update_values | update_JxW_values);

    const unsigned int n_dofs = fe_->n_dofs_per_cell();

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        fev.reinit(cell);
        FullMatrix<double> M(n_dofs, n_dofs);

        // FESystem(FE_DGQ, 2) shape functions are component specific: phi_i is
        // nonzero only for its own component, so M(i,j) = 0 whenever
        // component(i) != component(j).  Cross-component pairs are skipped;
        // otherwise M would have zero rows and gauss_jordan() would abort.
        // Only the cell components (0,1) carry a mass term, so the trace rows
        // and columns of M stay exactly zero -- which is what makes alpha*M
        // leave the trace rows algebraic.
        for (unsigned int q = 0; q < fev.n_quadrature_points; ++q)
          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              const unsigned int ci = fe_->system_to_component_index(i).first;
              if (ci >= 2)
                continue;
              for (unsigned int j = 0; j < n_dofs; ++j)
                {
                  const unsigned int cj =
                    fe_->system_to_component_index(j).first;
                  if (ci != cj)
                    continue;
                  M(i, j) +=
                    fev.shape_value(i, q) * fev.shape_value(j, q) * fev.JxW(q);
                }
            }

        per_cell_mass[cell->active_cell_index()] = M;

        // For the initial-state inverse, put 1 on the trace diagonal so the
        // dense inverse exists; that block is never applied.
        FullMatrix<double> M_inv(M);
        for (unsigned int i = 0; i < n_dofs; ++i)
          if (fe_->system_to_component_index(i).first >= 2)
            M_inv(i, i) = 1.0;
        M_inv.gauss_jordan();
        per_cell_mass_inv[cell->active_cell_index()] = std::move(M_inv);
      }
  }

  // ============================================================================
  // open_csv_files
  //
  // One CSV for the selected vessels' arc-length midpoints. The probe
  // cell is chosen by a global argmin over the distance to the midpoint: each
  // rank offers its best locally owned candidate and the rank holding the
  // overall winner evaluates the probe. Rank zero writes gathered snapshots
  // ordered by vessel ID, so neither records nor files are duplicated in MPI.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::open_csv_files()
  {
    // Every rank may open a vessel file, so every rank makes sure the
    // directory exists (create_directories is a no-op when it already does).
    if (!output_directory.empty())
      std::filesystem::create_directories(output_directory);

    close_csv_files();
    mach_warning_active.clear();

    if (!output_terminal_csv_file.empty())
      {
        std::vector<unsigned int> local_terminal_ids;
        for (const auto &cell : dof_handler_.active_cell_iterators())
          if (cell->is_locally_owned())
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
              if (cell->face(f)->at_boundary() &&
                  !is_junction_face(cell->id(), f))
                local_terminal_ids.push_back(cell->face(f)->boundary_id());
        std::map<unsigned int, unsigned int> terminal_counts;
        for (const auto &ids :
             Utilities::MPI::all_gather(mpi_communicator_, local_terminal_ids))
          for (const auto id : ids)
            ++terminal_counts[id];
        for (const auto &entry : terminal_counts)
          AssertThrow(entry.second == 1,
                      ExcMessage(
                        "Terminal boundary ids must be globally unique"));
        std::string error;
        if (this_mpi_process == 0)
          try
            {
              const auto filename = std::filesystem::path(output_directory) /
                                    output_terminal_csv_file;
              if (filename.has_parent_path())
                std::filesystem::create_directories(filename.parent_path());
              csv_terminal.open(filename);
              AssertThrow(csv_terminal.good(),
                          ExcMessage("Cannot open terminal CSV file: " +
                                     output_terminal_csv_file));
              csv_write_row(csv_terminal,
                            "time_s",
                            "pressure_Pa",
                            "flow_m3_s",
                            "boundary_id",
                            "vessel_id",
                            "label");
            }
          catch (const std::exception &exception)
            {
              error = exception.what();
            }
        error = Utilities::MPI::broadcast(mpi_communicator_, error, 0);
        AssertThrow(error.empty(), ExcMessage(error));
      }
    probe_targets.clear();

    collective_csv_action(mpi_communicator_, [&]() {
      if (this_mpi_process == 0 && !output_network_terminals_csv_file.empty())
        {
          std::ofstream terminals(
            prepare_csv_path(output_directory,
                             output_network_terminals_csv_file));
          AssertThrow(terminals,
                      ExcMessage("Cannot open network terminal metadata CSV."));
          csv_write_row(terminals,
                        "boundary_id",
                        "vessel_id",
                        "label",
                        "R1",
                        "R2",
                        "C",
                        "P_out");
          for (const auto &[vid, ids] : vessel_terminals)
            for (const auto id : ids)
              {
                const auto &rcr = rcr_map.at(id);
                csv_write_row(terminals,
                              id,
                              vid,
                              rcr.label,
                              rcr.R1,
                              rcr.R2,
                              rcr.C,
                              rcr.P_out);
              }
        }
    });

    if (output_vessel_csv_file.empty())
      return;
    std::set<unsigned int> selected;
    for (const auto value : vessel_probe_ids)
      {
        AssertThrow(vessel_map.count(value),
                    ExcMessage("Unknown vessel probe ID " +
                               std::to_string(value)));
        selected.insert(value);
      }
    collective_csv_action(mpi_communicator_, [&]() {
      if (this_mpi_process == 0)
        {
          csv_vessel.open(
            prepare_csv_path(output_directory, output_vessel_csv_file));
          AssertThrow(csv_vessel,
                      ExcMessage("Cannot open vessel probe CSV: " +
                                 output_vessel_csv_file));
          csv_write_row(csv_vessel,
                        "time_s",
                        "vessel_id",
                        "pressure_Pa",
                        "flow_m3_s",
                        "area_m2",
                        "velocity_m_s");
        }
    });

    for (const auto &[vid, vpp] : vessel_map)
      {
        (void)vpp;
        if (!selected.empty() && !selected.count(vid))
          continue;

        double     s_mid = 0.0;
        const auto it    = vessel_s_bounds.find(vid);
        if (it != vessel_s_bounds.end())
          s_mid = 0.5 * (it->second.first + it->second.second);

        // Pick the cell whose arc-length interval contains s_mid (distance 0)
        // and probe it at the reference coordinate of s_mid.  Probing at the
        // centre of the nearest cell instead would move the probe with the
        // mesh (L/2, L/4, 3L/8, ... for 1, 2, 4, ... cells per vessel), and
        // runs on different refinements would not be comparable.
        double best_dist = std::numeric_limits<double>::max();
        double best_xi   = 0.5;
        typename DoFHandler<dim, spacedim>::active_cell_iterator best_cell;
        bool                                                     found = false;

        for (const auto &cell : dof_handler_.active_cell_iterators())
          {
            if (!cell->is_locally_owned())
              continue;
            if (cell->material_id() != vid)
              continue;

            const Tensor<1, spacedim> d_hat = compute_directional_vector(cell);
            const double              s0    = cell->vertex(0) * d_hat;
            const double              s1    = cell->vertex(1) * d_hat;
            const double d = std::max(0.0, std::max(s0 - s_mid, s_mid - s1));
            if (d < best_dist)
              {
                best_dist = d;
                best_cell = cell;
                best_xi   = std::clamp((s_mid - s0) / (s1 - s0), 0.0, 1.0);
                found     = true;
              }
          }

        // Ties are broken by rank, so exactly one rank claims the vessel even
        // if two of them report the same distance.
        const auto winner =
          Utilities::MPI::min_max_avg(best_dist, mpi_communicator_);
        const unsigned int winning_rank =
          Utilities::MPI::min(found && best_dist <= winner.min ?
                                this_mpi_process :
                                numbers::invalid_unsigned_int,
                              mpi_communicator_);

        if (winning_rank != this_mpi_process || !found)
          continue;

        probe_targets.emplace_back(vid, best_cell, best_xi);
      }
  }

  // ============================================================================
  // write_csv_row
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::write_csv_row(const double      t,
                                                const VectorType &sol,
                                                const VectorType *ydot)
  {
    AssertThrow(
      !viscoelastic_wall || ydot != nullptr,
      ExcMessage(
        "Viscoelastic pressure output requires the state derivative."));
    if (!output_terminal_csv_file.empty())
      {
        using Row = std::pair<unsigned int, std::vector<std::string>>;
        std::vector<Row> local_rows;
        std::string      error;
        try
          {
            for (const auto &cell : dof_handler_.active_cell_iterators())
              if (cell->is_locally_owned())
                for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell;
                     ++f)
                  if (cell->face(f)->at_boundary() &&
                      !is_junction_face(cell->id(), f))
                    {
                      const auto bid  = cell->face(f)->boundary_id();
                      double     area = 0., velocity = 0.;
                      get_face_trace(sol, cell, f, area, velocity);
                      const double pressure =
                        compute_physical_pressure(area,
                                                  cell->material_id(),
                                                  compute_a_d_at_face(cell, f),
                                                  t,
                                                  cell->face(f)->center(),
                                                  cell->id()) +
                        (ydot ? viscous_pressure(
                                  sol, *ydot, cell, cell->face(f)->center()) :
                                0.);
                      const auto        rcr = rcr_map.find(bid);
                      const std::string label =
                        rcr == rcr_map.end() ?
                          "boundary_" + std::to_string(bid) :
                          rcr->second.label;
                      local_rows.emplace_back(
                        bid,
                        std::vector<std::string>{csv_cell(t),
                                                 csv_cell(pressure),
                                                 csv_cell((f == 0 ? -1. : 1.) *
                                                          area * velocity),
                                                 csv_cell(bid),
                                                 csv_cell(cell->material_id()),
                                                 label});
                    }
          }
        catch (const std::exception &exception)
          {
            error = exception.what();
          }
        for (const auto &message :
             Utilities::MPI::all_gather(mpi_communicator_, error))
          AssertThrow(message.empty(), ExcMessage(message));
        auto parts = Utilities::MPI::gather(mpi_communicator_, local_rows, 0);
        if (this_mpi_process == 0)
          try
            {
              std::vector<Row> rows;
              for (auto &part : parts)
                for (auto &row : part)
                  rows.push_back(std::move(row));
              std::sort(rows.begin(),
                        rows.end(),
                        [](const Row &left, const Row &right) {
                          return left.first < right.first;
                        });
              // Buffer one snapshot, preserving CSV quoting/precision while
              // avoiding a filesystem flush for each individual terminal.
              std::ostringstream snapshot;
              for (const auto &row : rows)
                csv_write_record(snapshot, row.second);
              csv_terminal << snapshot.str() << std::flush;
              AssertThrow(csv_terminal.good(),
                          ExcMessage("Cannot write terminal CSV file: " +
                                     output_terminal_csv_file));
            }
          catch (const std::exception &exception)
            {
              error = exception.what();
            }
        error = Utilities::MPI::broadcast(mpi_communicator_, error, 0);
        AssertThrow(error.empty(), ExcMessage(error));
      }
    if (output_vessel_csv_file.empty())
      return;
    const unsigned int dofs_per_cell = fe_->n_dofs_per_cell();
    using ProbeRow = std::pair<unsigned int, std::vector<std::string>>;
    std::vector<ProbeRow> local_probe_rows;
    collective_csv_action(mpi_communicator_, [&]() {
      for (const auto &[vid, cell, xi] : probe_targets)
        {
          const Point<dim> xi_probe =
            (dim == 1) ? Point<dim>(xi) : Point<dim>();

          // The probe cell is locally owned, so every one of its DoFs is owned
          // here and can be read straight out of the non-ghosted vector.
          std::vector<types::global_dof_index> ldofs(dofs_per_cell);
          get_system_dof_indices(cell, ldofs);

          double A_val = 0.0, U_val = 0.0;
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const double       phi  = fe_->shape_value(i, xi_probe);
              const unsigned int comp = fe_->system_to_component_index(i).first;
              if (comp == 0)
                A_val += sol(ldofs[i]) * phi;
              else if (comp == 1)
                U_val += sol(ldofs[i]) * phi;
            }

          const Point<spacedim> x_probe =
            cell->vertex(0) + xi * (cell->vertex(1) - cell->vertex(0));
          const double P_Pa =
            compute_physical_pressure(A_val,
                                      vid,
                                      compute_a_d_at_point(cell, x_probe),
                                      t,
                                      x_probe,
                                      cell->id()) +
            (ydot ? viscous_pressure(sol, *ydot, cell, x_probe) : 0.);
          local_probe_rows.emplace_back(
            vid,
            std::vector<std::string>{csv_cell(t),
                                     csv_cell(vid),
                                     csv_cell(P_Pa),
                                     csv_cell(A_val * U_val),
                                     csv_cell(A_val),
                                     csv_cell(U_val)});
        }
    });
    const auto parts =
      Utilities::MPI::gather(mpi_communicator_, local_probe_rows, 0);
    collective_csv_action(mpi_communicator_, [&]() {
      if (this_mpi_process == 0 && csv_vessel.is_open())
        {
          std::vector<ProbeRow> rows;
          for (const auto &part : parts)
            rows.insert(rows.end(), part.begin(), part.end());
          std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
            return a.first < b.first;
          });
          for (const auto &row : rows)
            csv_write_record(csv_vessel, row.second);
          csv_vessel.flush();
          AssertThrow(csv_vessel, ExcMessage("Cannot write vessel probe CSV."));
        }
    });
  }

  // ============================================================================
  // close_csv_files
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::close_csv_files()
  {
    flow_regime_log.close();
    if (csv_vessel.is_open())
      csv_vessel.close();
    if (csv_terminal.is_open())
      csv_terminal.close();
  }

  // ============================================================================
  // HLL flux (residual)
  // ============================================================================
  template <int dim, int spacedim>
  std::array<double, 2>
  BloodFlowSystem<dim, spacedim>::hll_flux(
    const double       bn_L,
    const double       bn_R,
    const double       A_L,
    const double       U_L,
    const double       A_R,
    const double       U_R,
    const unsigned int vid_L,
    const unsigned int vid_R,
    const double       ad_L,
    const double       ad_R,
    const double       external_pressure_L,
    const double       external_pressure_R) const
  {
    const double c_L   = compute_wave_speed(A_L, vid_L, ad_L);
    const double c_R   = compute_wave_speed(A_R, vid_R, ad_R);
    const double U_bar = 0.5 * (U_L + U_R);
    const double c_bar = 0.5 * (c_L + c_R);
    const double s_L   = U_bar - c_bar;
    const double s_R   = U_bar + c_bar;
    // const double s_L = std::min(U_L - c_L, U_R - c_R);

    // const double s_R = std::max(U_L + c_L, U_R + c_R);

    const double FAL = scalar_area_flux(bn_L, A_L, U_L);
    const double FUL =
      scalar_momentum_flux(bn_L,
                           U_L,
                           compute_pressure_value(A_L, vid_L, ad_L) +
                             external_pressure_L,
                           density);
    const double FAR = scalar_area_flux(bn_R, A_R, U_R);
    const double FUR =
      scalar_momentum_flux(bn_R,
                           U_R,
                           compute_pressure_value(A_R, vid_R, ad_R) +
                             external_pressure_R,
                           density);

    if (s_L >= 0.0)
      return {{FAL, FUL}};
    if (s_R <= 0.0)
      return {{FAR, FUR}};

    const double inv = 1.0 / (s_R - s_L);
    return {{(s_R * FAL - s_L * FAR + s_R * s_L * (A_R - A_L)) * inv,
             (s_R * FUL - s_L * FUR + s_R * s_L * (U_R - U_L)) * inv}};
  }

  // ============================================================================
  // HLL flux Jacobian (linearised w.r.t. trial perturbations dA_L,
  // dU_L, …)
  // ============================================================================
  template <int dim, int spacedim>
  std::array<double, 2>
  BloodFlowSystem<dim, spacedim>::hll_flux_jac(const double       bn_L,
                                               const double       bn_R,
                                               const double       A_L,
                                               const double       U_L,
                                               const double       A_R,
                                               const double       U_R,
                                               const double       dA_L,
                                               const double       dU_L,
                                               const double       dA_R,
                                               const double       dU_R,
                                               const unsigned int vid_L,
                                               const unsigned int vid_R,
                                               const double       ad_L,
                                               const double       ad_R) const
  {
    const double c_L   = compute_wave_speed(A_L, vid_L, ad_L);
    const double c_R   = compute_wave_speed(A_R, vid_R, ad_R);
    const double U_bar = 0.5 * (U_L + U_R);
    const double c_bar = 0.5 * (c_L + c_R);
    const double s_L   = U_bar - c_bar;
    const double s_R   = U_bar + c_bar;
    // const double s_L = std::min(U_L - c_L, U_R - c_R);

    // const double s_R = std::max(U_L + c_L, U_R + c_R);

    const double c2L_over_AL = c_L * c_L / A_L;
    const double c2R_over_AR = c_R * c_R / A_R;

    const double FAL_j = scalar_area_flux_jac(bn_L, A_L, U_L, dA_L, dU_L);
    const double FUL_j =
      scalar_momentum_flux_jac(bn_L, c2L_over_AL, U_L, dA_L, dU_L);
    const double FAR_j = scalar_area_flux_jac(bn_R, A_R, U_R, dA_R, dU_R);
    const double FUR_j =
      scalar_momentum_flux_jac(bn_R, c2R_over_AR, U_R, dA_R, dU_R);

    if (s_L >= 0.0)
      return {{FAL_j, FUL_j}};
    if (s_R <= 0.0)
      return {{FAR_j, FUR_j}};

    const double          inv    = 1.0 / (s_R - s_L);
    std::array<double, 2> result = {
      {(s_R * FAL_j - s_L * FAR_j + s_R * s_L * (dA_R - dA_L)) * inv,
       (s_R * FUL_j - s_L * FUR_j + s_R * s_L * (dU_R - dU_L)) * inv}};
    if (wall_model != "Square root")
      {
        const double du = .5 * (dU_L + dU_R);
        const double dc =
          .5 * (compute_wave_speed_derivative(A_L, vid_L, ad_L) * dA_L +
                compute_wave_speed_derivative(A_R, vid_R, ad_R) * dA_R);
        const double dsL = du - dc, dsR = du + dc;
        const auto   flux = hll_flux(
          bn_L, bn_R, A_L, U_L, A_R, U_R, vid_L, vid_R, ad_L, ad_R, 0., 0.);
        const std::array<double, 2> left = {
          scalar_area_flux(bn_L, A_L, U_L),
          scalar_momentum_flux(
            bn_L, U_L, compute_pressure_value(A_L, vid_L, ad_L), density)};
        const std::array<double, 2> right = {
          scalar_area_flux(bn_R, A_R, U_R),
          scalar_momentum_flux(
            bn_R, U_R, compute_pressure_value(A_R, vid_R, ad_R), density)};
        const std::array<double, 2> jump = {A_R - A_L, U_R - U_L};
        for (unsigned int i = 0; i < 2; ++i)
          result[i] +=
            inv * (dsR * left[i] - dsL * right[i] +
                   (dsR * s_L + s_R * dsL) * jump[i] - (dsR - dsL) * flux[i]);
      }
    return result;
  }

  // HLL-HDG flux based on Paper "Hybridisable discontinuous
  // Galerkin formulation of compressible flows "

  template <int dim, int spacedim>
  std::array<double, 2>
  BloodFlowSystem<dim, spacedim>::hll_hdg_flux(
    const double bn_L,
    const double /*bn_R*/,
    const double A_L,
    const double U_L, // interior U_e
    const double A_R,
    const double U_R, // trace U_b
    const unsigned int /*vid_L*/,
    const unsigned int vid_R,
    const double /*ad_L*/,
    const double ad_R,
    const double /*external_pressure_L*/,
    const double external_pressure_R) const
  {
    // Stabilization from TRACE state only (eq. 37 of Vila-Perez et
    // al.)
    const double c_b    = compute_wave_speed(A_R, vid_R, ad_R);
    const double s_plus = std::max(0.0, U_R * bn_L + c_b); // s⁺ at trace

    // Physical flux at TRACE (F(U_b)·n)
    const double FA_b = scalar_area_flux(bn_L, A_R, U_R);
    const double FU_b =
      scalar_momentum_flux(bn_L,
                           U_R,
                           compute_pressure_value(A_R, vid_R, ad_R) +
                             external_pressure_R,
                           density);

    // Stabilization term: s⁺(U_e - U_b)
    const double FA = FA_b + s_plus * (A_L - A_R);
    const double FU = FU_b + s_plus * (U_L - U_R);

    return {{FA, FU}};
  }

  template <int dim, int spacedim>
  std::array<double, 2>
  BloodFlowSystem<dim, spacedim>::hll_hdg_flux_jac(
    const double bn_L,
    const double /*bn_R*/,
    const double A_L,
    const double U_L,
    const double A_R,
    const double U_R,
    const double dA_L,
    const double dU_L, // perturbation direction
    const double dA_R,
    const double dU_R,
    const unsigned int /*vid_L*/,
    const unsigned int vid_R,
    const double /*ad_L*/,
    const double ad_R) const
  {
    const double c_b    = compute_wave_speed(A_R, vid_R, ad_R);
    const double dc_b   = compute_wave_speed_derivative(A_R, vid_R, ad_R);
    const double s_plus = std::max(0.0, U_R * bn_L + c_b);

    // dF(U_b)·n / dU_b  (trace perturbation dA_R, dU_R)
    const double c2_over_A_b = c_b * c_b / A_R;
    const double dFA_b       = scalar_area_flux_jac(bn_L, A_R, U_R, dA_R, dU_R);
    const double dFU_b =
      scalar_momentum_flux_jac(bn_L, c2_over_A_b, U_R, dA_R, dU_R);

    if (s_plus <= 0.0)
      {
        // Pure upwind from trace — only trace terms survive
        return {{dFA_b, dFU_b}};
      }

    // d(s⁺)/d(A_R) = dc_b/dA_R * dA_R (if U_R*bn + c_b > 0)
    const double ds_plus_dAR = dc_b * dA_R;
    const double ds_plus_dUR = bn_L * dU_R;
    const double ds_plus     = ds_plus_dAR + ds_plus_dUR;

    const double dFA = dFA_b                     // dF(U_b)/d(U_b) * dU_b
                       + ds_plus * (A_L - A_R)   // d(s⁺)/d(U_b) * (U_e - U_b)
                       + s_plus * (dA_L - dA_R); // s⁺ * d(U_e - U_b)
    const double dFU = dFU_b + ds_plus * (U_L - U_R) + s_plus * (dU_L - dU_R);

    return {{dFA, dFU}};
  }

  // ============================================================================
  // Lax–Friedrichs flux (residual)
  // ============================================================================
  template <int dim, int spacedim>
  std::array<double, 2>
  BloodFlowSystem<dim, spacedim>::lf_flux(
    const double       bn_L,
    const double       bn_R,
    const double       A_L,
    const double       U_L,
    const double       A_R,
    const double       U_R,
    const unsigned int vid_L,
    const unsigned int vid_R,
    const double       ad_L,
    const double       ad_R,
    const double       external_pressure_L,
    const double       external_pressure_R) const
  {
    const double FAL = scalar_area_flux(bn_L, A_L, U_L);
    const double FUL =
      scalar_momentum_flux(bn_L,
                           U_L,
                           compute_pressure_value(A_L, vid_L, ad_L) +
                             external_pressure_L,
                           density);
    const double FAR = scalar_area_flux(bn_R, A_R, U_R);
    const double FUR =
      scalar_momentum_flux(bn_R,
                           U_R,
                           compute_pressure_value(A_R, vid_R, ad_R) +
                             external_pressure_R,
                           density);
    const double alpha =
      theta * compute_LF_penalty(
                A_L, A_R, U_L, U_R, bn_L, bn_R, vid_L, vid_R, ad_L, ad_R);

    return {{0.5 * (FAL + FAR) - 0.5 * alpha * (A_R - A_L),
             0.5 * (FUL + FUR) - 0.5 * alpha * (U_R - U_L)}};
  }

  template <int dim, int spacedim>
  std::array<double, 2>
  BloodFlowSystem<dim, spacedim>::numerical_external_pressure_flux(
    const double       bn_L,
    const double       bn_R,
    const double       A_L,
    const double       U_L,
    const double       A_R,
    const double       U_R,
    const unsigned int vid_L,
    const unsigned int vid_R,
    const double       a_d_L,
    const double       a_d_R,
    const double       external_pressure_L,
    const double       external_pressure_R) const
  {
    const double pressure_flux_L = external_pressure_L / density * bn_L;
    const double pressure_flux_R = external_pressure_R / density * bn_R;

    if (numerical_flux_type == NumericalFluxType::HLL_HDG)
      {
        // HLL-HDG evaluates the physical momentum flux at the trace state;
        // the stabilization is independent of prescribed pressure.
        return {{0.0, pressure_flux_R}};
      }

    if (numerical_flux_type == NumericalFluxType::HLL)
      {
        const double c_L   = compute_wave_speed(A_L, vid_L, a_d_L);
        const double c_R   = compute_wave_speed(A_R, vid_R, a_d_R);
        const double U_bar = 0.5 * (U_L + U_R);
        const double c_bar = 0.5 * (c_L + c_R);
        const double s_L   = U_bar - c_bar;
        const double s_R   = U_bar + c_bar;

        if (s_L >= 0.0)
          return {{0.0, pressure_flux_L}};
        if (s_R <= 0.0)
          return {{0.0, pressure_flux_R}};

        const double inv = 1.0 / (s_R - s_L);
        return {{0.0, (s_R * pressure_flux_L - s_L * pressure_flux_R) * inv}};
      }

    // Lax--Friedrichs stabilization acts only on the state jump and therefore
    // has no prescribed-pressure contribution.
    return {{0.0, 0.5 * (pressure_flux_L + pressure_flux_R)}};
  }

  // ============================================================================
  // Lax–Friedrichs flux Jacobian
  // ============================================================================
  template <int dim, int spacedim>
  std::array<double, 2>
  BloodFlowSystem<dim, spacedim>::lf_flux_jac(const double       bn_L,
                                              const double       bn_R,
                                              const double       A_L,
                                              const double       U_L,
                                              const double       A_R,
                                              const double       U_R,
                                              const double       dA_L,
                                              const double       dU_L,
                                              const double       dA_R,
                                              const double       dU_R,
                                              const unsigned int vid_L,
                                              const unsigned int vid_R,
                                              const double       ad_L,
                                              const double       ad_R) const
  {
    const double c2L = compute_wave_speed(A_L, vid_L, ad_L);
    const double c2R = compute_wave_speed(A_R, vid_R, ad_R);

    const double FAL_j = scalar_area_flux_jac(bn_L, A_L, U_L, dA_L, dU_L);
    const double FUL_j =
      scalar_momentum_flux_jac(bn_L, c2L * c2L / A_L, U_L, dA_L, dU_L);
    const double FAR_j = scalar_area_flux_jac(bn_R, A_R, U_R, dA_R, dU_R);
    const double FUR_j =
      scalar_momentum_flux_jac(bn_R, c2R * c2R / A_R, U_R, dA_R, dU_R);
    const double alpha =
      theta * compute_LF_penalty(
                A_L, A_R, U_L, U_R, bn_L, bn_R, vid_L, vid_R, ad_L, ad_R);

    std::array<double, 2> result = {
      {0.5 * (FAL_j + FAR_j) - 0.5 * alpha * (dA_R - dA_L),
       0.5 * (FUL_j + FUR_j) - 0.5 * alpha * (dU_R - dU_L)}};
    if (wall_model != "Square root")
      {
        const double dcL =
          compute_wave_speed_derivative(A_L, vid_L, ad_L) * dA_L;
        const double dcR =
          compute_wave_speed_derivative(A_R, vid_R, ad_R) * dA_R;
        const std::array<double, 4> waves = {(U_L - c2L) * bn_L,
                                             (U_L + c2L) * bn_L,
                                             (U_R - c2R) * bn_R,
                                             (U_R + c2R) * bn_R};
        const std::array<double, 4> dw    = {(dU_L - dcL) * bn_L,
                                             (dU_L + dcL) * bn_L,
                                             (dU_R - dcR) * bn_R,
                                             (dU_R + dcR) * bn_R};
        unsigned int                k     = 0;
        for (unsigned int i = 1; i < 4; ++i)
          if (std::abs(waves[i]) > std::abs(waves[k]))
            k = i;
        const double da = theta * (waves[k] >= 0. ? 1. : -1.) * dw[k];
        result[0] -= .5 * da * (A_R - A_L);
        result[1] -= .5 * da * (U_R - U_L);
      }
    return result;
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_cell_residuals(
    const double t,
    const VectorType & /*y*/,
    VectorType                     &F,
    const ExternalPressureProvider *provider,
    const bool                      pressure_only)
  {
    TimerOutput::Scope timer(computing_timer, "assemble_cell_residuals");


    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);
    const FEValuesExtractors::Scalar a_hat_extractor(2); // trace area
    const FEValuesExtractors::Scalar u_hat_extractor(3); // trace velocity

    const QGauss<dim>     quad_cell(fe_->tensor_degree() + 1);
    const QGauss<dim - 1> quad_face(fe_->tensor_degree() + 1);

    FEValues<dim, spacedim>     fev(*fe_,
                                quad_cell,
                                update_values | update_gradients |
                                  update_quadrature_points | update_JxW_values);
    FEFaceValues<dim, spacedim> fef(*fe_,
                                    quad_face,
                                    update_values | update_JxW_values |
                                      update_normal_vectors |
                                      update_quadrature_points);

    rhs_function.set_time(t);

    std::vector<double> A_h(fev.n_quadrature_points);
    std::vector<double> U_h(fev.n_quadrature_points);
    std::vector<double> Ah_q(fef.n_quadrature_points);
    std::vector<double> Uh_q(fef.n_quadrature_points);
    std::vector<double> Ahat_q(fef.n_quadrature_points);
    std::vector<double> Uhat_q(fef.n_quadrature_points);

    const unsigned int                   n_dofs = fe_->n_dofs_per_cell();
    std::vector<double>                  local_values(n_dofs);
    Vector<double>                       cell_rhs(n_dofs);
    std::vector<types::global_dof_index> ldofs(n_dofs);

    const double rho = density;

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        const auto         cell_id = cell->id();
        const unsigned int vid     = cell->material_id();
        const double       eta = 2.0 * (friction_profile + 2.0) * numbers::PI *
                           vessel_map.at(vid).dynamic_viscosity / rho;
        fev.reinit(cell);

        get_system_dof_indices(cell, ldofs);
        y_relevant.extract_subvector_to(ldofs, local_values);

        const auto &JxW = fev.get_JxW_values();

        fev[area_extractor].get_function_values_from_local_dof_values(
          local_values, A_h);
        fev[velocity_extractor].get_function_values_from_local_dof_values(
          local_values, U_h);

        cell_rhs = 0;

        // ---- Volume integral
        // --------------------------------------------------
        for (unsigned int q = 0; q < fev.n_quadrature_points; ++q)
          {
            const double A = std::max(A_h[q], 1e-10);
            const double U = U_h[q];
            const double P =
              pressure_only ?
                external_pressure(
                  {t, fev.get_quadrature_points()[q], vid, cell->id()},
                  provider) :
                compute_physical_pressure(
                  A,
                  vid,
                  compute_a_d_at_point(cell, fev.quadrature_point(q)),
                  t,
                  fev.get_quadrature_points()[q],
                  cell_id);
            const Tensor<1, spacedim> b = compute_directional_vector(cell);

            const double rhs_A =
              rhs_function.value(fev.get_quadrature_points()[q], 0);
            const double rhs_U =
              rhs_function.value(fev.get_quadrature_points()[q], 1);

            for (unsigned int i = 0; i < n_dofs; ++i)
              {
                const unsigned int comp =
                  fe_->system_to_component_index(i).first;

                if (comp == 0 && !pressure_only)
                  {
                    cell_rhs(i) +=
                      (rhs_A * fev[area_extractor].value(i, q) +
                       A * U * (b * fev[area_extractor].gradient(i, q))) *
                      JxW[q];
                  }
                else if (comp == 1)
                  {
                    const double phi_u = fev[velocity_extractor].value(i, q);
                    if (pressure_only)
                      cell_rhs(i) +=
                        (P / rho) *
                        (b * fev[velocity_extractor].gradient(i, q)) * JxW[q];
                    else
                      cell_rhs(i) +=
                        (rhs_U * phi_u +
                         (0.5 * U * U + P / rho) *
                           (b * fev[velocity_extractor].gradient(i, q)) -
                         eta * U / A * phi_u) *
                        JxW[q];
                  }
              }
          }

        // ---- Face flux: interior cell value vs. global face trace
        //
        // Interior and trace state sit at the same point, so both pressures
        // use the a_d of the face.

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            fef.reinit(cell, f);
            const double ad_face = compute_a_d_at_face(cell, f);
            const auto  &normals = fef.get_normal_vectors();
            const auto  &JxW     = fef.get_JxW_values();

            // Interior cell values at the face quadrature point
            fef[area_extractor].get_function_values_from_local_dof_values(
              local_values, Ah_q);
            fef[velocity_extractor].get_function_values_from_local_dof_values(
              local_values, Uh_q);

            // Face-trace values: this cell's OWN (A_hat, U_hat) on
            // face f, read with the same extractor idiom as the
            // cell values.  For boundary faces and K>=3 junction
            // half-faces this DOF *is* the unique/per-vessel trace;
            // on an ordinary interior (or 2-way) face it is tied to
            // the canonical side by the continuity rows, so the
            // converged value is identical to the canonical one.
            fef[a_hat_extractor].get_function_values_from_local_dof_values(
              local_values, Ahat_q);
            fef[u_hat_extractor].get_function_values_from_local_dof_values(
              local_values, Uhat_q);

            for (unsigned int q = 0; q < fef.n_quadrature_points; ++q)
              {
                const double A_in  = std::max(Ah_q[q], 1e-10);
                const double U_in  = Uh_q[q];
                const double A_hat = std::max(Ahat_q[q], 1e-10);
                const double U_hat = Uhat_q[q];
                const double bn =
                  compute_tangent_normal_product(cell, normals[q]);
                const Point<spacedim> &point = fef.get_quadrature_points()[q];
                const double           external_p =
                  external_pressure({t, point, vid, cell_id}, provider);

                // numerical_flux designed for left/right states.
                // In current implementation there is no physical
                // neighbour cell on the other side so we are ;

                const auto [FA, FU] =
                  pressure_only ? numerical_external_pressure_flux(bn,
                                                                   bn,
                                                                   A_in,
                                                                   U_in,
                                                                   A_hat,
                                                                   U_hat,
                                                                   vid,
                                                                   vid,
                                                                   ad_face,
                                                                   ad_face,
                                                                   external_p,
                                                                   external_p) :
                                  numerical_flux(bn,
                                                 bn,
                                                 A_in,
                                                 U_in,
                                                 A_hat,
                                                 U_hat,
                                                 vid,
                                                 vid,
                                                 ad_face,
                                                 ad_face,
                                                 external_p,
                                                 external_p);

                for (unsigned int i = 0; i < n_dofs; ++i)
                  {
                    const unsigned int comp =
                      fe_->system_to_component_index(i).first;
                    if (comp >= 2)
                      continue; // trace components carry no cell
                                // flux
                    cell_rhs(i) -=
                      (comp == 0 ? FA : FU) * fef.shape_value(i, q) * JxW[q];
                  }
              }
          }

        // ---- Scatter into global F
        // -------------------------------------------
        // Every DoF of this FESystem is discontinuous, so all of `ldofs`
        // belongs to this cell alone and is owned by this rank: no other rank
        // contributes to these rows.
        F.add(ldofs, cell_rhs);
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_trace_interior_equations(
    const double                    t,
    const VectorType               &y,
    VectorType                     &F,
    const ExternalPressureProvider *provider,
    const bool                      pressure_only)
  {
    TimerOutput::Scope timer(computing_timer,
                             "assemble_trace_interior_equations");

    // FEValues and FEFaceValues read through the ghosted FE-range vector.  It
    // is sized dof_handler_.n_dofs(), which get_function_values() asserts, and
    // it holds every entry a locally owned cell can ask for, including the ones
    // on its ghost neighbours.
    const VectorType &y_cell = y_fe_relevant;

    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);
    // const FEValuesExtractors::Scalar a_hat_extractor(2); // trace area
    // const FEValuesExtractors::Scalar u_hat_extractor(3); // trace velocity

    const QGauss<dim - 1> quad_face(1); // each 1-D face is a single 0-D point
    FEFaceValues<dim, spacedim> fef(*fe_,
                                    quad_face,
                                    update_values | update_normal_vectors |
                                      update_quadrature_points);

    for (const auto &face : interior_faces)
      {
        const auto        &cell  = face.cell;
        const auto        &nb    = face.neighbor;
        const unsigned int f     = face.face_no;
        const unsigned int nb_f  = face.neighbor_face_no;
        const unsigned int vid_L = cell->material_id();
        const unsigned int vid_R = nb->material_id();
        // A non-junction interior face joins two cells of the same vessel
        // at one point, so both states and the trace share its a_d.
        const double ad_face = compute_a_d_at_face(cell, f);
        const double ad_L    = ad_face;
        const double ad_R    = ad_face;
        // Interior cell values at face — use y_cell, not y
        fef.reinit(cell, f);
        const double bn_L =
          compute_interior_tangent_normal_product(cell,
                                                  fef.get_normal_vectors()[0]);

        std::vector<double> A_L_v(1), U_L_v(1);
        fef[area_extractor].get_function_values(y_cell, A_L_v);
        fef[velocity_extractor].get_function_values(y_cell, U_L_v);

        fef.reinit(nb, nb_f);
        const double bn_R =
          compute_interior_tangent_normal_product(nb,
                                                  fef.get_normal_vectors()[0]);

        std::vector<double> A_R_v(1), U_R_v(1);
        fef[area_extractor].get_function_values(y_cell, A_R_v);
        fef[velocity_extractor].get_function_values(y_cell, U_R_v);

        const double           A_L   = std::max(A_L_v[0], 1e-10);
        const double           U_L   = U_L_v[0];
        const double           A_R   = std::max(A_R_v[0], 1e-10);
        const double           U_R   = U_R_v[0];
        const Point<spacedim> &point = fef.get_quadrature_points()[0];
        const double           external_pressure_L =
          external_pressure({t, point, vid_L, cell->id()}, provider);
        const double external_pressure_R =
          external_pressure({t, point, vid_R, nb->id()}, provider);

        // Trace state
        double A_hat = 0.0, U_hat = 0.0;
        A_hat = y(face.trace.a_hat_dof);
        U_hat = y(face.trace.u_hat_dof);
        A_hat = std::max(A_hat, 1e-10);

        // Left HDG type flux residuals (face integrals with
        // interior state from left cell) numerical_flux is
        // designed for left/right states. The trace acts as the
        // exterior state, therefore use opposite orientation on
        // the trace side.
        const auto [FA_L, FU_L] =
          pressure_only ?
            numerical_external_pressure_flux(bn_L,
                                             bn_L,
                                             A_L,
                                             U_L,
                                             A_hat,
                                             U_hat,
                                             vid_L,
                                             vid_R,
                                             ad_L,
                                             ad_face,
                                             external_pressure_L,
                                             external_pressure_R) :
            numerical_flux(bn_L,
                           bn_L,
                           A_L,
                           U_L,
                           A_hat,
                           U_hat,
                           vid_L,
                           vid_R,
                           ad_L,
                           ad_face,
                           external_pressure_L,
                           external_pressure_R);

        // Right HDG type flux residuals (face integrals with
        // interior state from right cell)
        const auto [FA_R, FU_R] =
          pressure_only ?
            numerical_external_pressure_flux(bn_R,
                                             bn_R,
                                             A_R,
                                             U_R,
                                             A_hat,
                                             U_hat,
                                             vid_R,
                                             vid_L,
                                             ad_R,
                                             ad_face,
                                             external_pressure_R,
                                             external_pressure_L) :
            numerical_flux(bn_R,
                           bn_R,
                           A_R,
                           U_R,
                           A_hat,
                           U_hat,
                           vid_R,
                           vid_L,
                           ad_R,
                           ad_face,
                           external_pressure_R,
                           external_pressure_L);

        // Trace dofs
        const FaceTraceDof &td = face.trace;

        // HDG type transmission equations
        // sum of left and right fluxes must be zero
        F(td.a_hat_dof) += FA_L + FA_R;
        F(td.u_hat_dof) += FU_L + FU_R;
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_trace_boundary_equations(
    const double                    t,
    const VectorType               &y,
    VectorType                     &F,
    const ExternalPressureProvider *provider,
    const bool                      pressure_only)
  {
    TimerOutput::Scope timer(computing_timer,
                             "assemble_trace_boundary_equations");


    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);

    std::vector<types::global_dof_index> ldofs(fe_->n_dofs_per_cell());
    std::vector<double>                  local_values(fe_->n_dofs_per_cell());
    std::vector<double>                  A_int_v(1), U_int_v(1);

    const QGauss<dim - 1>       quad_face(1);
    FEFaceValues<dim, spacedim> fef(*fe_,
                                    quad_face,
                                    update_values | update_quadrature_points);

    inflow_function.set_time(t);
    if (outlet_type == "Dirichlet")
      exact_solution.set_time(t);

    for (const auto &face : boundary_faces)
      {
        const auto        &cell = face.cell;
        const unsigned int f    = face.face_no;

        const types::boundary_id bid = cell->face(f)->boundary_id();
        const unsigned int       vid = cell->material_id();

        // Gather the interior state in coupled system numbering.
        fef.reinit(cell, f);
        const Point<spacedim> &point = fef.get_quadrature_points()[0];

        if (pressure_only)
          {
            if (outlet_type == "RCR" && rcr_map.count(bid) &&
                (rcr_map.at(bid).R1 > 0.0 || rcr_map.at(bid).R2 > 0.0))
              {
                Assert(locally_owned_dofs_.is_element(face.a_hat_dof),
                       ExcInternalError());
                F(face.a_hat_dof) +=
                  external_pressure({t, point, vid, cell->id()}, provider);
              }
            continue;
          }

        get_system_dof_indices(cell, ldofs);
        y_relevant.extract_subvector_to(ldofs, local_values);
        fef[area_extractor].get_function_values_from_local_dof_values(
          local_values, A_int_v);
        fef[velocity_extractor].get_function_values_from_local_dof_values(
          local_values, U_int_v);

        const double A_int     = std::max(A_int_v[0], 1e-10);
        const double U_int     = U_int_v[0];
        const double a_d_local = compute_a_d_at_face(cell, f);

        // Current face trace (from trace block of y)
        const double A_hat_cur = y(face.a_hat_dof);
        const double U_hat_cur = y(face.u_hat_dof);
        const double A_hat     = std::max(A_hat_cur, 1e-10);

        double res_A = 0.0, res_U = 0.0;

        if (outlet_type == "Dirichlet")
          {
            res_A = A_hat_cur - exact_solution.value(point, 0);
            res_U = U_hat_cur - exact_solution.value(point, 1);
          }
        else if (bid == 0) // inflow
          {
            const double Q_in = prescribed_inflow(t);
            const double W2_int =
              U_int - compute_invariant(A_int, vid, a_d_local);

            res_A = A_hat_cur * U_hat_cur - Q_in;
            res_U =
              (U_hat_cur - compute_invariant(A_hat, vid, a_d_local)) - W2_int;
          }
        else if (terminal_links.count(bid))
          {
            const auto  &port     = terminal_ports.at(bid);
            const double sign     = port.outward_sign;
            const double Q        = sign * A_hat_cur * U_hat_cur;
            const double pressure = compute_physical_pressure(
              A_hat, vid, a_d_local, t, point, cell->id());
            if (rcr_pc_dof.count(bid))
              res_A = pressure - rcr_map.at(bid).R1 * Q - y(rcr_pc_dof.at(bid));
            else
              {
                double conductance = 0., weighted_pressure = 0.;
                for (const auto &[other, resistance] : terminal_links.at(bid))
                  {
                    conductance += 1. / resistance;
                    weighted_pressure +=
                      terminal_pressure(other, y, t) / resistance;
                  }
                res_A = pressure - (Q + weighted_pressure) / conductance;
              }
            res_U = U_hat_cur +
                    sign * compute_invariant(A_hat, vid, a_d_local) - U_int -
                    sign * compute_invariant(A_int, vid, a_d_local);
          }
        else if (outlet_type == "RCR" && rcr_map.count(bid) &&
                 (rcr_map.at(bid).R1 > 0.0 || rcr_map.at(bid).R2 > 0.0))

          {
            const auto  &rcr = rcr_map.at(bid);
            const double Q   = A_hat_cur * U_hat_cur;
            if (rcr.C > 0.0)
              {
                // const double Pc =
                // terminal_Pc_storage.at(bid); res_A =
                //   compute_pressure_value(A_hat, vid,
                //   a_d_local) - (rcr.R1 * Q + Pc);
                const double Pc = y(rcr_pc_dof.at(bid));
                res_A           = compute_physical_pressure(
                          A_hat, vid, a_d_local, t, point, cell->id()) -
                        (rcr.R1 * Q + Pc);
              }
            else
              { // C=0: the two resistances remain in series.
                res_A = compute_physical_pressure(
                          A_hat, vid, a_d_local, t, point, cell->id()) -
                        ((rcr.R1 + rcr.R2) * Q + rcr.P_out);
              }
            const double W1_int =
              U_int + compute_invariant(A_int, vid, a_d_local);
            res_U =
              (U_hat_cur + compute_invariant(A_hat, vid, a_d_local)) - W1_int;
          }
        else // Reflection outlet
          {
            const double Rt = reflection_coefficient;
            const double W1_int =
              U_int + compute_invariant(A_int, vid, a_d_local);
            const double W2_tgt = -Rt * W1_int;

            res_A =
              (U_hat_cur + compute_invariant(A_hat, vid, a_d_local)) - W1_int;
            res_U =
              (U_hat_cur - compute_invariant(A_hat, vid, a_d_local)) - W2_tgt;
          }

        // A boundary face has exactly one incident cell, so its rows are
        // owned wherever that cell is owned.  Writing with `+=` into a
        // zeroed vector gives the same value as `=` and keeps the whole
        // residual on add semantics for a single compress().
        Assert(locally_owned_dofs_.is_element(face.a_hat_dof),
               ExcInternalError());
        F(face.a_hat_dof) += res_A;
        F(face.u_hat_dof) += res_U;
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_rcr_capacitor_equations(
    const VectorType &y,
    const VectorType &ydot,
    VectorType       &F,
    const double      t)
  {
    if (rcr_pc_dof.empty())
      return;

    for (const auto &face : boundary_faces)
      {
        const auto        &cell = face.cell;
        const unsigned int f    = face.face_no;
        const auto         pit  = rcr_pc_dof.find(cell->face(f)->boundary_id());
        if (pit == rcr_pc_dof.end())
          continue;

        const types::global_dof_index pc_dof = pit->second;
        const auto &rcr = rcr_map.at(cell->face(f)->boundary_id());

        const double A_hat = y(face.a_hat_dof);
        const double U_hat = y(face.u_hat_dof);
        const auto   bid   = cell->face(f)->boundary_id();
        const double sign =
          terminal_links.count(bid) ? terminal_ports.at(bid).outward_sign : 1.;
        const double Q      = sign * A_hat * U_hat;
        const double Pc     = y(pc_dof);
        const double Pc_dot = ydot(pc_dof);

        // C*Ṗc - Q + (Pc - P_out)/R2 = 0
        // Both the terminal cell and its capacitor row belong to this rank.
        double discharge = (Pc - rcr.P_out) / rcr.R2;
        if (terminal_links.count(bid))
          {
            discharge = 0.;
            for (const auto &[other, resistance] : terminal_links.at(bid))
              discharge += (Pc - terminal_pressure(other, y, t)) / resistance;
          }
        F(pc_dof) += rcr.C * Pc_dot - Q + discharge;
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_trace_junction_equations(
    const double                    t,
    const VectorType               &y,
    VectorType                     &F,
    const ExternalPressureProvider *provider,
    const bool                      pressure_only)
  {
    TimerOutput::Scope timer(computing_timer,
                             "assemble_trace_junction_equations");

    if (junctions.empty())
      return;


    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);

    const QGauss<dim - 1> quad_face(1);
    // Scalar DGQ values are reference-face data. Only these values are reused;
    // all material, geometry and pressure evaluations below remain current.
    std::array<std::unique_ptr<FEFaceValues<dim, spacedim>>,
               GeometryInfo<dim>::faces_per_cell>
      face_fe_values;
    for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      {
        face_fe_values[f] =
          std::make_unique<FEFaceValues<dim, spacedim>>(*fe_,
                                                        quad_face,
                                                        update_values);
        face_fe_values[f]->reinit(junctions.front().half_faces.front().cell, f);
      }

    const double rho   = density;
    const double A_min = 1e-10;

    std::vector<types::global_dof_index> ldofs(fe_->n_dofs_per_cell());
    std::vector<double>                  local_values(fe_->n_dofs_per_cell());
    std::vector<double>                  Av(1), Uv(1);
    std::vector<double>                  A_int, U_int, c0, W;
    std::vector<double>                  A_hat, U_hat, c_hat;
    std::vector<int>                     s;
    std::vector<types::global_dof_index> a_idx, u_idx;

    for (const auto &J : junctions)
      {
        const unsigned int K = J.n_vessels();

        // Reuse per-vessel storage across junctions of different sizes.
        A_int.resize(K);
        U_int.resize(K);
        c0.resize(K);
        W.resize(K);
        A_hat.resize(K);
        U_hat.resize(K);
        c_hat.resize(K);
        s.resize(K);
        a_idx.resize(K);
        u_idx.resize(K);

        for (unsigned int i = 0; i < K; ++i)
          {
            const auto        &hf  = J.half_faces[i];
            const unsigned int vid = hf.cell->material_id();

            // Gather the junction state in coupled system numbering.
            auto &fef = *face_fe_values[hf.face_no];
            get_system_dof_indices(hf.cell, ldofs);
            y_relevant.extract_subvector_to(ldofs, local_values);
            fef[area_extractor].get_function_values_from_local_dof_values(
              local_values, Av);
            fef[velocity_extractor].get_function_values_from_local_dof_values(
              local_values, Uv);

            A_int[i] = std::max(Av[0], A_min);
            U_int[i] = Uv[0];
            s[i]     = hf.orientation; // +1 if junction is at face 1
                                       // (right end)
            const double a_d_face = compute_a_d_at_face(hf.cell, hf.face_no);
            c0[i]                 = compute_wave_speed(a_d_face, vid, a_d_face);

            // Outgoing Riemann invariant from cell i toward the
            // junction
            W[i] = U_int[i] + static_cast<double>(s[i]) *
                                compute_invariant(A_int[i], vid, a_d_face);

            // Trace DOF indices and current trace values (from
            // trace block of y)
            // const FaceTraceDof &td =
            //   face_dof_map.at(canonical_face_key(hf.cell, hf.face_no));
            // std::cout << "Rank " << this_mpi_process
            //           << " owned=" << hf.cell->is_locally_owned()
            //           << " ghost=" << hf.cell->is_ghost()
            //           << " artificial=" << hf.cell->is_artificial() <<
            //           std::endl;
            // const auto key = canonical_face_key(hf.cell, hf.face_no);

            // auto it = face_dof_map.find(key);

            // if (it == face_dof_map.end())
            //   {
            //     std::cout << "Rank " << this_mpi_process << " missing face
            //     key
            //     "
            //               << key.first << " " << key.second << std::endl;

            //     std::abort();
            //   }

            // const FaceTraceDof &td = it->second;

            // a_idx[i] = td.a_hat_dof;
            // u_idx[i] = td.u_hat_dof;
            const auto [a_hat_dof, u_hat_dof] =
              face_trace_dofs(ldofs, hf.face_no);

            a_idx[i] = a_hat_dof;
            u_idx[i] = u_hat_dof;

            A_hat[i] = std::max(y(a_idx[i]), A_min);
            U_hat[i] = y(u_idx[i]);
            c_hat[i] = compute_wave_speed(A_hat[i], vid, a_d_face);
          }

        if (pressure_only)
          {
            const double external_pressure_0 =
              external_pressure({t,
                                 J.location,
                                 J.half_faces[0].cell->material_id(),
                                 J.half_faces[0].cell->id()},
                                provider);
            for (unsigned int i = 1; i < K; ++i)
              if (locally_owned_dofs_.is_element(u_idx[i - 1]))
                {
                  const double external_pressure_i =
                    external_pressure({t,
                                       J.location,
                                       J.half_faces[i].cell->material_id(),
                                       J.half_faces[i].cell->id()},
                                      provider);
                  F(u_idx[i - 1]) +=
                    (external_pressure_0 - external_pressure_i) / rho;
                }
            continue;
          }

        // (a) Mass conservation -> row a_idx[0].
        //
        // The 2K rows of a junction are spread over its K half-faces, so a
        // junction cut by a subdomain boundary has its rows split between the
        // two ranks.  Each rank writes only the rows it owns and reads the
        // other vessels' states from its ghost layer, which is complete here
        // because the layer is vertex-adjacent.  No reduction closes these
        // equations.
        if (locally_owned_dofs_.is_element(a_idx[0]))
          {
            double mass_res = 0.0;
            for (unsigned int i = 0; i < K; ++i)
              mass_res += static_cast<double>(s[i]) * A_hat[i] * U_hat[i];
            F(a_idx[0]) += mass_res;
          }

        // (b) Total-head continuity: H_0 − H_i = 0 -> rows
        // u_idx[0..K−2]
        {
          const double a_d0 =
            compute_a_d_at_face(J.half_faces[0].cell, J.half_faces[0].face_no);
          const double H0 =
            0.5 * gamma * U_hat[0] * U_hat[0] +
            compute_physical_pressure(A_hat[0],
                                      J.half_faces[0].cell->material_id(),
                                      a_d0,
                                      t,
                                      J.location,
                                      J.half_faces[0].cell->id()) /
              rho;

          for (unsigned int i = 1; i < K; ++i)
            {
              const double a_di = compute_a_d_at_face(J.half_faces[i].cell,
                                                      J.half_faces[i].face_no);
              const double Hi =
                0.5 * gamma * U_hat[i] * U_hat[i] +
                compute_physical_pressure(A_hat[i],
                                          J.half_faces[i].cell->material_id(),
                                          a_di,
                                          t,
                                          J.location,
                                          J.half_faces[i].cell->id()) /
                  rho;

              if (locally_owned_dofs_.is_element(u_idx[i - 1]))
                F(u_idx[i - 1]) += H0 - Hi;
            }
        }

        // (c) Riemann compatibility: U_hat_i + s_i*4(c_hat_i−c0_i)
        // − W_i = 0
        //     vessel 0 -> row u_idx[K−1]
        //     vessel i>=1 -> row a_idx[i]
        for (unsigned int i = 0; i < K; ++i)
          {
            const double compatibility_res =
              U_hat[i] +
              static_cast<double>(s[i]) *
                compute_invariant(
                  A_hat[i],
                  J.half_faces[i].cell->material_id(),
                  compute_a_d_at_face(J.half_faces[i].cell,
                                      J.half_faces[i].face_no)) -
              W[i];

            const types::global_dof_index row =
              (i == 0) ? u_idx[K - 1] : a_idx[i];

            if (locally_owned_dofs_.is_element(row))
              F(row) += compatibility_res;
          }
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_trace_continuity_equations(
    const VectorType &y,
    VectorType       &F)
  {
    TimerOutput::Scope timer(computing_timer,
                             "assemble_trace_continuity_equations");
    for (const auto &p : trace_continuity_pairs)
      {
        // The duplicate side is always a locally owned cell's own pair, so
        // these rows are owned by construction; the canonical values are read
        // from the ghosted vector and may live on another rank.
        Assert(locally_owned_dofs_.is_element(p.a_dup), ExcInternalError());
        F(p.a_dup) += y(p.a_dup) - y(p.a_canon);
        F(p.u_dup) += y(p.u_dup) - y(p.u_canon);
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_jacobian_trace_continuity_block()
  {
    // Owned by construction: the duplicate side is a locally owned cell's own
    // pair.  The canonical column may belong to another rank, which is fine --
    // only rows are partitioned.
    for (const auto &p : trace_continuity_pairs)
      {
        jacobian_matrix.add(p.a_dup, p.a_dup, 1.0);
        jacobian_matrix.add(p.a_dup, p.a_canon, -1.0);
        jacobian_matrix.add(p.u_dup, p.u_dup, 1.0);
        jacobian_matrix.add(p.u_dup, p.u_canon, -1.0);
      }
  }

  // ============================================================================
  // assemble_residual
  //
  // IDA's F(t, y, ydot) = 0.  Reads go through the ghosted vectors filled by
  // update_ghosted_vectors(); writes go to the locally owned `residual` and are
  // closed by a single compress(add).  Every routine called here uses add
  // semantics, so the two are never mixed in one compress.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_residual(const double      t,
                                                    const VectorType &y,
                                                    const VectorType &ydot,
                                                    VectorType       &residual)
  {
    TimerOutput::Scope timer(computing_timer, "assemble_residual");
    if (verbosity > 1)
      {
        deallog.push("assemble_residual");
        deallog << "t=" << t << std::endl;
      }

    update_ghosted_vectors(y);

    // ---- raw residuals R(y) -------------------------------------------------
    residual_F = 0.0;
    assemble_cell_residuals(t, y_relevant, residual_F);
    assemble_trace_interior_equations(t, y_relevant, residual_F);
    assemble_trace_boundary_equations(t, y_relevant, residual_F);
    assemble_trace_junction_equations(t, y_relevant, residual_F);
    assemble_trace_continuity_equations(y_relevant, residual_F);
    residual_F.compress(VectorOperation::add);

    // ---- residual = M_K * ydot - R, uniformly over the FE range -------------
    // per_cell_mass has an exactly zero block on the trace components (2,3), so
    // this single formula produces the right thing for both kinds of row:
    //   cell rows  (differential):  M ydot - R_cell
    //   trace rows (algebraic)   :  0      - R_trace  =  -R_trace
    // Every FE DoF, trace ones included, belongs to exactly one cell, and that
    // cell is locally owned, so each row is written by exactly one rank once.
    residual = 0.0;

    const unsigned int n_dofs = fe_->n_dofs_per_cell();
    Vector<double>     local_F(n_dofs), local_Mydot(n_dofs), local_res(n_dofs);
    std::vector<types::global_dof_index> ldofs(n_dofs);

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        get_system_dof_indices(cell, ldofs);

        residual_F.extract_subvector_to(ldofs.begin(),
                                        ldofs.end(),
                                        local_F.begin());
        ydot.extract_subvector_to(ldofs.begin(),
                                  ldofs.end(),
                                  local_Mydot.begin());

        per_cell_mass[cell->active_cell_index()].vmult(local_res, local_Mydot);

        for (unsigned int i = 0; i < n_dofs; ++i)
          local_res(i) -= local_F(i);
        residual.add(ldofs, local_res);
      }

    // RCR capacitor rows: differential, written directly in M*ydot - R form.
    assemble_rcr_capacitor_equations(y_relevant, ydot, residual, t);

    residual.compress(VectorOperation::add);
    if (viscoelastic_wall)
      assemble_viscous_wall(y, ydot, &residual, nullptr, false);
    if (verbosity > 1)
      deallog.pop();
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::add_external_pressure_residual(
    const double                    t,
    const VectorType               &y,
    const ExternalPressureProvider &provider,
    VectorType                     &destination) const
  {
    TimerOutput::Scope timer(computing_timer, "add_external_pressure_residual");
    AssertThrow(provider,
                ExcMessage("An external-pressure provider cannot be empty."));
    AssertDimension(destination.size(), n_total_dofs);

    auto *self = const_cast<BloodFlowSystem *>(this);
    self->update_ghosted_vectors(y);
    residual_F = 0.0;
    self->assemble_cell_residuals(t, y_relevant, residual_F, &provider, true);
    self->assemble_trace_interior_equations(
      t, y_relevant, residual_F, &provider, true);
    self->assemble_trace_boundary_equations(
      t, y_relevant, residual_F, &provider, true);
    self->assemble_trace_junction_equations(
      t, y_relevant, residual_F, &provider, true);
    residual_F.compress(VectorOperation::add);

    // The native residual is F = M*ydot - R.  The pressure-only routines
    // assemble their contribution to the raw R vector, so this is the exact
    // additive pressure contribution to F.  Capacitor and trace-continuity
    // rows have no prescribed-pressure term.
    const unsigned int                   n_dofs = fe_->n_dofs_per_cell();
    std::vector<types::global_dof_index> ldofs(n_dofs);
    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;
        get_system_dof_indices(cell, ldofs);
        for (const auto dof : ldofs)
          destination(dof) -= residual_F(dof);
      }
    destination.compress(VectorOperation::add);
  }

  // ============================================================================
  // assemble_jacobian
  //
  // J_IDA = dF/dy + alpha * dF/dydot = -dR/dy + alpha * M.
  //
  // The blocks are added first and compressed once; only then is the matrix
  // scaled and the mass term added, because *= and the ghost exchange of add()
  // both need the matrix in a compressed state.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_jacobian(const double      t,
                                                    const VectorType &y,
                                                    const VectorType &ydot,
                                                    const double      alpha)
  {
    TimerOutput::Scope timer(computing_timer, "assemble_jacobian");
    if (verbosity > 1)
      deallog.push("assemble_jacobian");
    deallog << "t=" << t << std::endl;

    assemble_state_jacobian(t, y, ydot);
    assemble_derivative_jacobian(t, y, ydot);

    jacobian_matrix.copy_from(state_jacobian_matrix_);
    jacobian_matrix.add(alpha, derivative_jacobian_matrix_);
    if (!jacobian_export_filename.empty() && !jacobian_exported &&
        t >= jacobian_export_time)
      {
        export_jacobian_matrix_market(jacobian_export_filename, t, alpha);
        jacobian_exported = true;
      }
    if (verbosity > 1)
      deallog.pop();
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::export_jacobian_matrix_market(
    const std::string &filename,
    const double       t,
    const double       alpha) const
  {
    types::global_dof_index local_nnz = 0;
    {
      std::ofstream part(filename + ".part" +
                         Utilities::int_to_string(this_mpi_process, 4));
      part << std::setprecision(17) << std::scientific;
      for (const auto row : locally_owned_dofs_)
        for (auto it = jacobian_matrix.begin(row);
             it != jacobian_matrix.end(row);
             ++it)
          {
            // Matrix Market is 1-based
            part << row + 1 << ' ' << it->column() + 1 << ' ' << it->value()
                 << '\n';
            ++local_nnz;
          }
    }

    const types::global_dof_index nnz =
      Utilities::MPI::sum(local_nnz, mpi_communicator_);

    if (this_mpi_process == 0)
      {
        std::ofstream header(filename + ".header");
        header << "%%MatrixMarket matrix coordinate real general\n"
               << "% IDA Jacobian J = -dR/dy + alpha*M, t = " << t
               << ", alpha = " << alpha << ", MPI ranks = " << n_mpi_processes
               << '\n'
               << jacobian_matrix.m() << ' ' << jacobian_matrix.n() << ' '
               << nnz << '\n';
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_state_jacobian(
    const double      t,
    const VectorType &y,
    const VectorType &ydot)
  {
    (void)ydot;
    update_ghosted_vectors(y);
    jacobian_matrix = 0.0;
    assemble_jacobian_cell_block(t, y_relevant);
    assemble_jacobian_trace_interior_block(y_relevant);
    assemble_jacobian_trace_boundary_block(t, y_relevant);
    assemble_jacobian_trace_junction_block(y_relevant);
    assemble_jacobian_trace_continuity_block();
    assemble_jacobian_rcr_capacitor_block(y_relevant);
    jacobian_matrix.compress(VectorOperation::add);
    jacobian_matrix *= -1.0;
    jacobian_matrix.compress(VectorOperation::insert);
    if (viscoelastic_wall)
      assemble_viscous_wall(y, ydot, nullptr, &jacobian_matrix, false);
    state_jacobian_matrix_.copy_from(jacobian_matrix);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_derivative_jacobian(
    const double      t,
    const VectorType &y,
    const VectorType &ydot)
  {
    (void)t;
    (void)y;
    (void)ydot;
    // Only the mesh mass matrices and terminal capacitances enter this block.
    // They change during setup or parameter reload, which invalidate the cache.
    if (derivative_jacobian_is_current_ && !viscoelastic_wall)
      return;
    derivative_jacobian_matrix_ = 0.0;

    const unsigned int                   n_dofs = fe_->n_dofs_per_cell();
    std::vector<types::global_dof_index> ldofs(n_dofs);
    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;
        get_system_dof_indices(cell, ldofs);
        const FullMatrix<double> &M_K =
          per_cell_mass[cell->active_cell_index()];
        for (unsigned int i = 0; i < n_dofs; ++i)
          for (unsigned int j = 0; j < n_dofs; ++j)
            derivative_jacobian_matrix_.add(ldofs[i], ldofs[j], M_K(i, j));
      }

    for (const auto &[bid, pc_dof] : rcr_pc_dof)
      if (locally_owned_dofs_.is_element(pc_dof))
        derivative_jacobian_matrix_.add(pc_dof, pc_dof, rcr_map.at(bid).C);

    derivative_jacobian_matrix_.compress(VectorOperation::add);
    if (viscoelastic_wall)
      assemble_viscous_wall(
        y, ydot, nullptr, &derivative_jacobian_matrix_, true);
    derivative_jacobian_is_current_ = true;
  }

  template <int dim, int spacedim>
  const MatrixType &
  BloodFlowSystem<dim, spacedim>::state_jacobian_matrix() const
  {
    return state_jacobian_matrix_;
  }

  template <int dim, int spacedim>
  const MatrixType &
  BloodFlowSystem<dim, spacedim>::derivative_jacobian_matrix() const
  {
    return derivative_jacobian_matrix_;
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_jacobian_cell_block(
    const double t,
    const VectorType & /*y*/)
  {
    TimerOutput::Scope timer(computing_timer, "assemble_jacobian_cell_block");


    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);
    const FEValuesExtractors::Scalar a_hat_extractor(2); // trace area
    const FEValuesExtractors::Scalar u_hat_extractor(3); // trace velocity

    const QGauss<dim>     quad_cell(fe_->tensor_degree() + 1);
    const QGauss<dim - 1> quad_face(fe_->tensor_degree() + 1);

    FEValues<dim, spacedim>     fev(*fe_,
                                quad_cell,
                                update_values | update_gradients |
                                  update_quadrature_points | update_JxW_values);
    FEFaceValues<dim, spacedim> fef(*fe_,
                                    quad_face,
                                    update_values | update_JxW_values |
                                      update_normal_vectors);

    const double rho = density;
    (void)t;
    const unsigned int                   n_dofs = fe_->n_dofs_per_cell();
    std::vector<double>                  local_values(n_dofs);
    std::vector<types::global_dof_index> ldofs(n_dofs);
    std::vector<double>                  A_h(fev.n_quadrature_points),
      U_h(fev.n_quadrature_points);
    std::vector<double> Ah_q(fef.n_quadrature_points),
      Uh_q(fef.n_quadrature_points);
    std::vector<double> Ahat_q(fef.n_quadrature_points),
      Uhat_q(fef.n_quadrature_points);
    FullMatrix<double>     cell_matrix(n_dofs, n_dofs);
    constexpr unsigned int n_trace_columns =
      2 * GeometryInfo<dim>::faces_per_cell;
    std::array<types::global_dof_index, n_trace_columns> trace_columns;
    boost::container::small_vector<double, 64>           trace_values;
    trace_values.resize(n_dofs * n_trace_columns);

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        const unsigned int vid = cell->material_id();

        const double eta = 2.0 * (friction_profile + 2.0) * numbers::PI *
                           vessel_map.at(vid).dynamic_viscosity / rho;

        fev.reinit(cell);

        const auto &JxW = fev.get_JxW_values();
        get_system_dof_indices(cell, ldofs);
        y_relevant.extract_subvector_to(ldofs, local_values);

        fev[area_extractor].get_function_values_from_local_dof_values(
          local_values, A_h);
        fev[velocity_extractor].get_function_values_from_local_dof_values(
          local_values, U_h);

        cell_matrix = 0.0;
        std::fill(trace_values.begin(), trace_values.end(), 0.0);

        // ---- Block (1,1) volume - cell_matrix
        // -------------------------------
        for (unsigned int q = 0; q < fev.n_quadrature_points; ++q)
          {
            const double A    = std::max(A_h[q], 1e-10);
            const double U    = U_h[q];
            const double dpdA = compute_pressure_derivative(
              A, vid, compute_a_d_at_point(cell, fev.quadrature_point(q)));
            const double c2_A = A / rho * dpdA;

            for (unsigned int i = 0; i < n_dofs; ++i)
              {
                const unsigned int ci = fe_->system_to_component_index(i).first;
                if (ci >= 2)
                  continue;
                const Tensor<1, spacedim> grad_phiA =
                  fev[area_extractor].gradient(i, q);
                const Tensor<1, spacedim> grad_phiU =
                  fev[velocity_extractor].gradient(i, q);
                const double phi_U = fev[velocity_extractor].value(i, q);
                const Tensor<1, spacedim> b = compute_directional_vector(cell);

                for (unsigned int j = 0; j < n_dofs; ++j)
                  {
                    if (fe_->system_to_component_index(j).first >= 2)
                      continue;
                    const double trial_A = fev[area_extractor].value(j, q);
                    const double trial_U = fev[velocity_extractor].value(j, q);

                    double contrib = 0.0;
                    if (ci == 0)
                      contrib = (U * trial_A + A * trial_U) * (b * grad_phiA);
                    else
                      contrib =
                        (c2_A / A * trial_A + U * trial_U) * (b * grad_phiU) -
                        eta * (trial_U / A - U * trial_A / (A * A)) * phi_U;

                    cell_matrix(i, j) += contrib * JxW[q];
                  }
              }
          }

        // ---- Block (1,1) face + Block (1,2)
        // --------------------------------- Face cell-cell ->
        // cell_matrix

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            fef.reinit(cell, f);
            const auto  &normals = fef.get_normal_vectors();
            const auto  &JxW_f   = fef.get_JxW_values();
            const double ad_face = compute_a_d_at_face(cell, f);
            fef[area_extractor].get_function_values_from_local_dof_values(
              local_values, Ah_q);
            fef[velocity_extractor].get_function_values_from_local_dof_values(
              local_values, Uh_q);

            // This cell's OWN trace values / DOFs on face f — must
            // match the extractor read in assemble_cell_residuals,
            // otherwise the analytic Jacobian differentiates w.r.t.
            // the wrong column.
            fef[a_hat_extractor].get_function_values_from_local_dof_values(
              local_values, Ahat_q);
            fef[u_hat_extractor].get_function_values_from_local_dof_values(
              local_values, Uhat_q);
            const double A_hat = std::max(Ahat_q[0], 1e-10);
            const double U_hat = Uhat_q[0];

            const auto [loc_a_hat_dof, loc_u_hat_dof] =
              face_trace_dofs(ldofs, f);
            const FaceTraceDof td{loc_a_hat_dof, loc_u_hat_dof};
            trace_columns[2 * f]     = td.a_hat_dof;
            trace_columns[2 * f + 1] = td.u_hat_dof;

            for (unsigned int q = 0; q < fef.n_quadrature_points; ++q)
              {
                const double A_in = std::max(Ah_q[q], 1e-10);
                const double U_in = Uh_q[q];
                const double bn =
                  compute_tangent_normal_product(cell, normals[q]);

                // (1,1) face -> cell_matrix
                for (unsigned int j = 0; j < n_dofs; ++j)
                  {
                    const double dA = fef[area_extractor].value(j, q);
                    const double dU = fef[velocity_extractor].value(j, q);
                    if (dA == 0.0 && dU == 0.0)
                      continue;

                    const auto [dFA, dFU] = numerical_flux_jac(bn,
                                                               bn,
                                                               A_in,
                                                               U_in,
                                                               A_hat,
                                                               U_hat,
                                                               dA,
                                                               dU,
                                                               0.0,
                                                               0.0,
                                                               vid,
                                                               vid,
                                                               ad_face,
                                                               ad_face);

                    for (unsigned int i = 0; i < n_dofs; ++i)
                      {
                        const unsigned int comp =
                          fe_->system_to_component_index(i).first;
                        if (comp >= 2)
                          continue; // trace rows carry no cell flux
                        const double phi = fef.shape_value(i, q);
                        cell_matrix(i, j) -=
                          (comp == 0 ? dFA : dFU) * phi * JxW_f[q];
                      }
                  }

                // (1,2) -> trace_values (trace cols)
                for (const auto trace_col : {td.a_hat_dof, td.u_hat_dof})
                  {
                    const bool         is_a = (trace_col == td.a_hat_dof);
                    const unsigned int local_trace_column =
                      2 * f + (is_a ? 0 : 1);
                    const double dA_hat = is_a ? 1.0 : 0.0;
                    const double dU_hat = is_a ? 0.0 : 1.0;

                    const auto [dFA, dFU] = numerical_flux_jac(bn,
                                                               bn,
                                                               A_in,
                                                               U_in,
                                                               A_hat,
                                                               U_hat,
                                                               0.0,
                                                               0.0,
                                                               dA_hat,
                                                               dU_hat,
                                                               vid,
                                                               vid,
                                                               ad_face,
                                                               ad_face);

                    for (unsigned int i = 0; i < n_dofs; ++i)
                      {
                        const unsigned int comp =
                          fe_->system_to_component_index(i).first;
                        if (comp >= 2)
                          continue; // trace rows carry no cell flux
                        const double phi = fef.shape_value(i, q);
                        trace_values[i * n_trace_columns + local_trace_column] =
                          -(comp == 0 ? dFA : dFU) * phi * JxW_f[q];
                      }
                  }
              }
          }
        // Scatter trace columns before volume + face cell-cell entries.
        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            const double *row_values =
              trace_values.data() + i * n_trace_columns;
            if (std::any_of(row_values,
                            row_values + n_trace_columns,
                            [](const double value) { return value != 0.0; }))
              jacobian_matrix.add(ldofs[i],
                                  n_trace_columns,
                                  trace_columns.data(),
                                  row_values,
                                  true,
                                  false);
          }
        jacobian_matrix.add(ldofs, cell_matrix);
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_jacobian_trace_interior_block(
    const VectorType &y)
  {
    TimerOutput::Scope timer(computing_timer,
                             "assemble_jacobian_trace_interior_block");

    // FEValues and FEFaceValues read through the ghosted FE-range vector.  It
    // is sized dof_handler_.n_dofs(), which get_function_values() asserts, and
    // it holds every entry a locally owned cell can ask for, including the ones
    // on its ghost neighbours.
    const VectorType &y_cell = y_fe_relevant;

    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);

    const QGauss<dim - 1>       quad_face(1);
    FEFaceValues<dim, spacedim> fef(*fe_,
                                    quad_face,
                                    update_values | update_normal_vectors);

    for (const auto &face : interior_faces)
      {
        const auto        &cell  = face.cell;
        const auto        &nb    = face.neighbor;
        const unsigned int f     = face.face_no;
        const unsigned int nb_f  = face.neighbor_face_no;
        const unsigned int vid_L = cell->material_id();
        const unsigned int vid_R = nb->material_id();
        // A non-junction interior face joins two cells of the same vessel
        // at one point, so both states and the trace share its a_d.
        const double ad_face = compute_a_d_at_face(cell, f);
        const double ad_L    = ad_face;
        const double ad_R    = ad_face;
        // ---- Left cell: state and normal
        // --------------------------------
        fef.reinit(cell, f);
        const double bn_L =
          compute_interior_tangent_normal_product(cell,
                                                  fef.get_normal_vectors()[0]);

        std::vector<double> A_Lv(1), U_Lv(1);
        fef[area_extractor].get_function_values(y_cell, A_Lv);
        fef[velocity_extractor].get_function_values(y_cell, U_Lv);
        const double A_L = std::max(A_Lv[0], 1e-10);
        const double U_L = U_Lv[0];

        std::vector<types::global_dof_index> ldofs_L(fe_->n_dofs_per_cell());
        get_system_dof_indices(cell, ldofs_L);

        // ---- Right cell: state and normal
        // --------------------------------
        fef.reinit(nb, nb_f);
        const double bn_R =
          compute_interior_tangent_normal_product(nb,
                                                  fef.get_normal_vectors()[0]);

        std::vector<double> A_Rv(1), U_Rv(1);
        fef[area_extractor].get_function_values(y_cell, A_Rv);
        fef[velocity_extractor].get_function_values(y_cell, U_Rv);
        const double A_R = std::max(A_Rv[0], 1e-10);
        const double U_R = U_Rv[0];

        std::vector<types::global_dof_index> ldofs_R(fe_->n_dofs_per_cell());
        get_system_dof_indices(nb, ldofs_R);

        // ---- Trace state
        // -------------------------------------------------
        double A_hat_cur = 0.0, U_hat_cur = 0.0;
        A_hat_cur          = y(face.trace.a_hat_dof);
        U_hat_cur          = y(face.trace.u_hat_dof);
        const double A_hat = std::max(A_hat_cur, 1e-10);
        const double U_hat = U_hat_cur;

        const FaceTraceDof           &td    = face.trace;
        const types::global_dof_index a_row = td.a_hat_dof;
        const types::global_dof_index u_row = td.u_hat_dof;

        // ================================================================
        // Block (trace, cell_L): dR/dw_L
        // Linearise F_hat(bn_L,bn_L, A_L,U_L, A_hat,U_hat)
        // w.r.t. w_L. Interior = L (trial nonzero), exterior =
        // trace (trial zero).
        // ================================================================
        fef.reinit(cell, f);
        for (unsigned int j = 0; j < fe_->n_dofs_per_cell(); ++j)
          {
            const double phi_A = fef[area_extractor].value(j, 0);
            const double phi_U = fef[velocity_extractor].value(j, 0);

            const auto [dFA, dFU] =
              numerical_flux_jac(bn_L,
                                 bn_L,
                                 A_L,
                                 U_L,
                                 A_hat,
                                 U_hat,
                                 phi_A,
                                 phi_U, // trial on interior (L)
                                 0.0,
                                 0.0, // trace DOF fixed here
                                 vid_L,
                                 vid_R,
                                 ad_L,
                                 ad_face);

            jacobian_matrix.add(a_row, ldofs_L[j], dFA);
            jacobian_matrix.add(u_row, ldofs_L[j], dFU);
          }

        // ================================================================
        // Block (trace, cell_R): dR/dw_R
        // Linearise F_hat(bn_R,bn_R, A_R,U_R, A_hat,U_hat)
        // w.r.t. w_R. Interior = R (trial nonzero), exterior =
        // trace (trial zero).
        // ================================================================
        fef.reinit(nb, nb_f);
        for (unsigned int j = 0; j < fe_->n_dofs_per_cell(); ++j)
          {
            const double phi_A = fef[area_extractor].value(j, 0);
            const double phi_U = fef[velocity_extractor].value(j, 0);

            const auto [dFA, dFU] =
              numerical_flux_jac(bn_R,
                                 bn_R,
                                 A_R,
                                 U_R,
                                 A_hat,
                                 U_hat,
                                 phi_A,
                                 phi_U, // trial on interior (R)
                                 0.0,
                                 0.0, // trace DOF fixed here
                                 vid_R,
                                 vid_L,
                                 ad_R,
                                 ad_face);

            jacobian_matrix.add(a_row, ldofs_R[j], dFA);
            jacobian_matrix.add(u_row, ldofs_R[j], dFU);
          }

        // ================================================================
        // Block (trace, trace): dR/d(A_hat, U_hat)
        // Trace appears as the EXTERIOR state in both fluxes.
        // Contribution from left flux: trial on exterior =
        // (dA_hat, dU_hat). Contribution from right flux: same
        // trial. Sum both contributions.
        // ================================================================
        // Unit perturbations in A_hat and U_hat:
        for (const auto &[dA_hat_trial, dU_hat_trial, col] :
             {std::tuple{1.0, 0.0, td.a_hat_dof},
              std::tuple{0.0, 1.0, td.u_hat_dof}})
          {
            // Left flux: exterior trial = (dA_hat_trial,
            // dU_hat_trial)
            const auto [dFA_L, dFU_L] =
              numerical_flux_jac(bn_L,
                                 bn_L,
                                 A_L,
                                 U_L,
                                 A_hat,
                                 U_hat,
                                 0.0,
                                 0.0, // interior trial = 0
                                 dA_hat_trial,
                                 dU_hat_trial,
                                 vid_L,
                                 vid_R,
                                 ad_L,
                                 ad_face);

            // Right flux: exterior trial = (dA_hat_trial,
            // dU_hat_trial)
            const auto [dFA_R, dFU_R] =
              numerical_flux_jac(bn_R,
                                 bn_R,
                                 A_R,
                                 U_R,
                                 A_hat,
                                 U_hat,
                                 0.0,
                                 0.0, // interior trial = 0
                                 dA_hat_trial,
                                 dU_hat_trial,
                                 vid_R,
                                 vid_L,
                                 ad_R,
                                 ad_face);

            jacobian_matrix.add(a_row, col, dFA_L + dFA_R);
            jacobian_matrix.add(u_row, col, dFU_L + dFU_R);
          }
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_jacobian_trace_boundary_block(
    const double      t,
    const VectorType &y)
  {
    TimerOutput::Scope timer(computing_timer,
                             "assemble_jacobian_trace_boundary_block");


    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);

    std::vector<types::global_dof_index> ldofs(fe_->n_dofs_per_cell());
    std::vector<double>                  local_values(fe_->n_dofs_per_cell());
    std::vector<double>                  A_int_v(1);

    const QGauss<dim - 1>       quad_face(1);
    FEFaceValues<dim, spacedim> fef(*fe_, quad_face, update_values);

    (void)t;

    for (const auto &face : boundary_faces)
      {
        const auto        &cell = face.cell;
        const unsigned int f    = face.face_no;

        const types::boundary_id bid = cell->face(f)->boundary_id();
        const unsigned int       vid = cell->material_id();
        // const auto              &vpp = vessel_map.at(vid);

        fef.reinit(cell, f);
        get_system_dof_indices(cell, ldofs);
        y_relevant.extract_subvector_to(ldofs, local_values);
        fef[area_extractor].get_function_values_from_local_dof_values(
          local_values, A_int_v);

        const double A_int     = std::max(A_int_v[0], 1e-10);
        const double a_d_local = compute_a_d_at_face(cell, f);
        const double dc_int =
          0.25 * compute_invariant_derivative(A_int, vid, a_d_local);

        const double A_hat_cur = y(face.a_hat_dof);
        const double U_hat_cur = y(face.u_hat_dof);
        const double A_hat     = std::max(A_hat_cur, 1e-10);
        const double dc_hat =
          0.25 * compute_invariant_derivative(A_hat, vid, a_d_local);
        const double dPdA_hat =
          compute_pressure_derivative(A_hat, vid, a_d_local);

        const types::global_dof_index a_row = face.a_hat_dof;
        const types::global_dof_index u_row = face.u_hat_dof;

        if (outlet_type == "Dirichlet")
          {
            jacobian_matrix.add(a_row, a_row, 1.0);
            jacobian_matrix.add(u_row, u_row, 1.0);
          }
        else if (bid == 0) // inflow
          {
            // R1 = A_hat * U_hat - Q_in(t)
            // R2 = U_hat - 4(c_hat - c0) - W2_int
            // where W2_int = U_int - 4(c_int - c0)

            // \partialR1/ \partialA_hat, \partialR1/
            // \partialU_hat
            jacobian_matrix.add(a_row, a_row, U_hat_cur);
            jacobian_matrix.add(a_row, u_row, A_hat);

            // \partialR2/\partialA_hat,
            // \partialR2/\partialU_hat
            jacobian_matrix.add(u_row, a_row, -4.0 * dc_hat);
            jacobian_matrix.add(u_row, u_row, 1.0);

            // \partialR2/\partial(A_int, U_int) — block (2,1)
            for (unsigned int j = 0; j < fe_->n_dofs_per_cell(); ++j)
              {
                const double phi_A = fef[area_extractor].value(j, 0);
                const double phi_U = fef[velocity_extractor].value(j, 0);
                // \partialW2_int/\partialw_int = phi_U -
                // 4*dc_int*phi_A
                jacobian_matrix.add(u_row,
                                    ldofs[j],
                                    -(phi_U - 4.0 * dc_int * phi_A));
              }
          }
        else if (terminal_links.count(bid))
          {
            const double sign       = terminal_ports.at(bid).outward_sign;
            double       resistance = rcr_map.at(bid).R1;
            if (!rcr_pc_dof.count(bid))
              {
                double conductance = 0.;
                for (const auto &[other, edge_resistance] :
                     terminal_links.at(bid))
                  conductance += 1. / edge_resistance;
                resistance = 1. / conductance;
                for (const auto &[other, edge_resistance] :
                     terminal_links.at(bid))
                  jacobian_matrix.add(a_row,
                                      terminal_pressure_dof(other),
                                      -resistance / edge_resistance *
                                        terminal_pressure_derivative(other, y));
              }
            jacobian_matrix.add(a_row,
                                a_row,
                                dPdA_hat - resistance * sign * U_hat_cur);
            jacobian_matrix.add(a_row, u_row, -resistance * sign * A_hat_cur);
            jacobian_matrix.add(u_row, a_row, sign * 4. * dc_hat);
            jacobian_matrix.add(u_row, u_row, 1.);
            for (unsigned int j = 0; j < fe_->n_dofs_per_cell(); ++j)
              jacobian_matrix.add(u_row,
                                  ldofs[j],
                                  -fef[velocity_extractor].value(j, 0) -
                                    sign * 4. * dc_int *
                                      fef[area_extractor].value(j, 0));
          }
        else if (outlet_type == "RCR" && rcr_map.count(bid) &&
                 (rcr_map.at(bid).R1 > 0.0 || rcr_map.at(bid).R2 > 0.0))

          {
            const auto &rcr = rcr_map.at(bid);

            if (rcr.C > 0.0)
              {
                // ---- Full RCR
                // -------------------------------------------------------
                // res_A = P(A_hat) - R1*(A_hat*U_hat) - Pc
                // res_U = U_hat + 4(c_hat - c0) - W1_int
                // dres_A/dA_hat = dP/dA_hat - R1*U_hat
                // dres_A/dU_hat = -R1*A_hat
                jacobian_matrix.add(a_row,
                                    a_row,
                                    dPdA_hat - rcr.R1 * U_hat_cur);
                jacobian_matrix.add(a_row, u_row, -rcr.R1 * A_hat);

                // dres_U/dA_hat = 4*dc_hat
                // dres_U/dU_hat = 1
                jacobian_matrix.add(u_row, a_row, 4.0 * dc_hat);
                jacobian_matrix.add(u_row, u_row, 1.0);

                // dres_U/d(A_int, U_int)
                for (unsigned int j = 0; j < fe_->n_dofs_per_cell(); ++j)
                  {
                    const double phi_A = fef[area_extractor].value(j, 0);
                    const double phi_U = fef[velocity_extractor].value(j, 0);
                    jacobian_matrix.add(u_row,
                                        ldofs[j],
                                        -(phi_U + 4.0 * dc_int * phi_A));
                  }
              }
            else
              {
                // C=0: P = (R1 + R2)*Q + P_out; no capacitor unknown.
                const double resistance = rcr.R1 + rcr.R2;
                jacobian_matrix.add(a_row,
                                    a_row,
                                    dPdA_hat - resistance * U_hat_cur);
                jacobian_matrix.add(a_row, u_row, -resistance * A_hat);

                // dres_U/dA_hat = 4*dc_hat
                // dres_U/dU_hat = 1
                // (identical to RCR — res_U has no R
                // dependence)
                jacobian_matrix.add(u_row, a_row, 4.0 * dc_hat);
                jacobian_matrix.add(u_row, u_row, 1.0);

                // dres_U/d(A_int, U_int)
                for (unsigned int j = 0; j < fe_->n_dofs_per_cell(); ++j)
                  {
                    const double phi_A = fef[area_extractor].value(j, 0);
                    const double phi_U = fef[velocity_extractor].value(j, 0);
                    jacobian_matrix.add(u_row,
                                        ldofs[j],
                                        -(phi_U + 4.0 * dc_int * phi_A));
                  }
              }
          }
        else // Reflection
          {
            const double Rt = reflection_coefficient;

            // R1 = U_hat + 4(c_hat - c0) - W1_int
            // R2 = U_hat - 4(c_hat - c0) - (-Rt * W1_int)

            // \partialR1/\partialA_hat,
            // \partialR1/\partialU_hat
            jacobian_matrix.add(a_row, a_row, 4.0 * dc_hat);
            jacobian_matrix.add(a_row, u_row, 1.0);

            // \partialR2/\partialA_hat,
            // \partialR2/\partialU_hat
            jacobian_matrix.add(u_row, a_row, -4.0 * dc_hat);
            jacobian_matrix.add(u_row, u_row, 1.0);

            // \partialR1 and \partialR2 /\partial(A_int,U_int)
            // via W1_int
            for (unsigned int j = 0; j < fe_->n_dofs_per_cell(); ++j)
              {
                const double phi_A = fef[area_extractor].value(j, 0);
                const double phi_U = fef[velocity_extractor].value(j, 0);
                const double dW1   = phi_U + 4.0 * dc_int * phi_A;
                jacobian_matrix.add(a_row, ldofs[j], -dW1);
                jacobian_matrix.add(u_row, ldofs[j], Rt * dW1);
              }
          }
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_jacobian_rcr_capacitor_block(
    const VectorType &y)
  {
    if (rcr_pc_dof.empty())
      return;

    for (const auto &face : boundary_faces)
      {
        const auto        &cell = face.cell;
        const unsigned int f    = face.face_no;
        const auto         pit  = rcr_pc_dof.find(cell->face(f)->boundary_id());
        if (pit == rcr_pc_dof.end())
          continue;

        const types::global_dof_index pc_dof = pit->second;
        const auto &rcr = rcr_map.at(cell->face(f)->boundary_id());

        const double A_hat = y(face.a_hat_dof);
        const double U_hat = y(face.u_hat_dof);

        // R_pc = A_hat*U_hat - (Pc - P_out)/R2
        const auto   bid = cell->face(f)->boundary_id();
        const double sign =
          terminal_links.count(bid) ? terminal_ports.at(bid).outward_sign : 1.;
        jacobian_matrix.add(pc_dof, face.a_hat_dof, sign * U_hat);
        jacobian_matrix.add(pc_dof, face.u_hat_dof, sign * A_hat);
        if (terminal_links.count(bid))
          {
            double conductance = 0.;
            for (const auto &[other, resistance] : terminal_links.at(bid))
              {
                conductance += 1. / resistance;
                jacobian_matrix.add(pc_dof,
                                    terminal_pressure_dof(other),
                                    terminal_pressure_derivative(other, y) /
                                      resistance);
              }
            jacobian_matrix.add(pc_dof, pc_dof, -conductance);
          }
        else
          jacobian_matrix.add(pc_dof, pc_dof, -1.0 / rcr.R2);

        // res_A = P(Â) - R1*Q - Pc  ->  d(res_A)/dPc = -1
        jacobian_matrix.add(face.a_hat_dof, pc_dof, -1.0);
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_jacobian_trace_junction_block(
    const VectorType &y)
  {
    TimerOutput::Scope timer(computing_timer,
                             "assemble_jacobian_trace_junction_block");

    if (junctions.empty())
      return;


    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);

    const QGauss<dim - 1> quad_face(1);
    // Scalar DGQ values are reference-face data. Only these values are reused;
    // all material, geometry and pressure evaluations below remain current.
    std::array<std::unique_ptr<FEFaceValues<dim, spacedim>>,
               GeometryInfo<dim>::faces_per_cell>
      face_fe_values;
    for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      {
        face_fe_values[f] =
          std::make_unique<FEFaceValues<dim, spacedim>>(*fe_,
                                                        quad_face,
                                                        update_values);
        face_fe_values[f]->reinit(junctions.front().half_faces.front().cell, f);
      }

    const double rho   = density;
    const double A_min = 1e-10;

    std::vector<double> local_values(fe_->n_dofs_per_cell());
    std::vector<double> Av(1);
    std::vector<double> A_int, A_hat, U_hat, dP_hat, dc_hat_v, dc_int_v;
    std::vector<int>    orient;
    std::vector<types::global_dof_index>              a_row, u_row;
    std::vector<std::vector<types::global_dof_index>> cell_dofs;

    for (const auto &J : junctions)
      {
        const unsigned int K = J.n_vessels();

        A_int.resize(K);
        A_hat.resize(K);
        U_hat.resize(K);
        dP_hat.resize(K);
        dc_hat_v.resize(K);
        dc_int_v.resize(K);
        orient.resize(K);
        a_row.resize(K);
        u_row.resize(K);
        cell_dofs.resize(K);

        for (unsigned int i = 0; i < K; ++i)
          {
            const auto        &hf  = J.half_faces[i];
            const unsigned int vid = hf.cell->material_id();

            cell_dofs[i].resize(fe_->n_dofs_per_cell());
            get_system_dof_indices(hf.cell, cell_dofs[i]);
            y_relevant.extract_subvector_to(cell_dofs[i], local_values);
            auto &fef = *face_fe_values[hf.face_no];
            fef[area_extractor].get_function_values_from_local_dof_values(
              local_values, Av);

            A_int[i]  = std::max(Av[0], A_min);
            orient[i] = hf.orientation;

            const double a_d_face = compute_a_d_at_face(hf.cell, hf.face_no);
            dc_int_v[i] =
              0.25 * compute_invariant_derivative(A_int[i], vid, a_d_face);


            // c0[i]       =
            // compute_wave_speed(vessel_map.at(vid).a_d, vid);
            // dc_int_v[i] = compute_wave_speed_derivative(A_int[i],
            // vid);

            // const auto  key = canonical_face_key(hf.cell, hf.face_no);
            // const auto &td  = face_dof_map.at(key);
            // a_row[i]        = td.a_hat_dof;
            // u_row[i]        = td.u_hat_dof;

            // A_hat[i] = std::max(y(a_row[i]), A_min);
            // U_hat[i] = y(u_row[i]);

            const auto [a_hat_dof, u_hat_dof] =
              face_trace_dofs(cell_dofs[i], hf.face_no);

            a_row[i] = a_hat_dof;
            u_row[i] = u_hat_dof;

            A_hat[i] = std::max(y(a_row[i]), A_min);
            U_hat[i] = y(u_row[i]);

            dc_hat_v[i] =
              0.25 * compute_invariant_derivative(A_hat[i], vid, a_d_face);
            dP_hat[i] = compute_pressure_derivative(A_hat[i], vid, a_d_face);
          }

        // Row a_row[0]: mass conservation \sum s_i A_hat_i U_hat_i = 0.
        // Each row group is guarded exactly as in the residual, so the two stay
        // in step and the sparsity built by build_junction_sparsity() matches.
        if (locally_owned_dofs_.is_element(a_row[0]))
          for (unsigned int i = 0; i < K; ++i)
            {
              const double s = static_cast<double>(orient[i]);
              jacobian_matrix.add(a_row[0], a_row[i], s * U_hat[i]);
              jacobian_matrix.add(a_row[0], u_row[i], s * A_hat[i]);
            }

        // Rows u_row[0..K-2]: H_0 − H_i = 0
        for (unsigned int i = 1; i < K; ++i)
          {
            if (!locally_owned_dofs_.is_element(u_row[i - 1]))
              continue;

            // \partialH_0/\partialA_hat_0 ,
            // \partialH_0/\partialU_hat_0
            jacobian_matrix.add(u_row[i - 1], a_row[0], dP_hat[0] / rho);
            jacobian_matrix.add(u_row[i - 1], u_row[0], gamma * U_hat[0]);
            //  \partial(-H_i)/\partialA_hat_i,
            //  \partial(-H_i)/\partialU_hat_i
            jacobian_matrix.add(u_row[i - 1], a_row[i], -dP_hat[i] / rho);
            jacobian_matrix.add(u_row[i - 1], u_row[i], -gamma * U_hat[i]);
          }

        // Compat row for vessel 0 -> u_row[K-1]
        // Compat row for vessel i>=1 -> a_row[i]
        for (unsigned int i = 0; i < K; ++i)
          {
            const double                  s = static_cast<double>(orient[i]);
            const types::global_dof_index rr =
              (i == 0) ? u_row[K - 1] : a_row[i];

            if (!locally_owned_dofs_.is_element(rr))
              continue;

            // \partial/\partialA_hat_i, \partial/\partialU_hat_i of
            // (U_hat_i + s*4(c_hat_i - c0_i) - W_i)
            jacobian_matrix.add(rr, a_row[i], s * 4.0 * dc_hat_v[i]);
            jacobian_matrix.add(rr, u_row[i], 1.0);

            // \partial(-W_i)/\partial(A_int_i, U_int_i)
            const unsigned int vid = J.half_faces[i].cell->material_id();
            auto              &fef = *face_fe_values[J.half_faces[i].face_no];

            for (unsigned int j = 0; j < fe_->n_dofs_per_cell(); ++j)
              {
                const double phi_A = fef[area_extractor].value(j, 0);
                const double phi_U = fef[velocity_extractor].value(j, 0);
                // \partialW_i/\partialw = phi_U +
                // s*4*dc_int_v[i]*phi_A
                const double dW = phi_U + s * 4.0 * dc_int_v[i] * phi_A;
                jacobian_matrix.add(rr, cell_dofs[i][j], -dW);
              }
            (void)vid;
          }
      }
  }

  // ============================================================================
  // compute_pressure
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::compute_pressure(const VectorType &y,
                                                   VectorType       &p) const
  {
    compute_pressure(y, p, time);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::compute_pressure(const VectorType &y,
                                                   VectorType       &p,
                                                   const double      t) const
  {
    AssertThrow(!viscoelastic_wall,
                ExcMessage(
                  "Viscoelastic compute_pressure requires y and ydot."));
    compute_pressure_impl(y, p, t, nullptr);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::compute_pressure(
    const VectorType               &y,
    VectorType                     &p,
    const double                    t,
    const ExternalPressureProvider &provider) const
  {
    AssertThrow(provider,
                ExcMessage("An external-pressure provider cannot be empty."));
    AssertThrow(!viscoelastic_wall,
                ExcMessage(
                  "Viscoelastic compute_pressure requires y and ydot."));
    compute_pressure_impl(y, p, t, &provider);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::compute_pressure(const VectorType &y,
                                                   const VectorType &ydot,
                                                   VectorType       &p,
                                                   const double      t) const
  {
    compute_pressure_impl(y, p, t, nullptr);
    if (!viscoelastic_wall)
      return;
    std::vector<types::global_dof_index> indices(fe_->n_dofs_per_cell());
    for (const auto &cell : dof_handler_.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          get_system_dof_indices(cell, indices);
          for (unsigned int i = 0; i < indices.size(); ++i)
            if (fe_->system_to_component_index(i).first == 0)
              p(indices[i]) =
                p(indices[i]) +
                viscous_pressure(y, ydot, cell, support_point(cell, i));
        }
    p.compress(VectorOperation::insert);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::compute_pressure_impl(
    const VectorType               &y,
    VectorType                     &p,
    const double                    t,
    const ExternalPressureProvider *provider) const
  {
    TimerOutput::Scope timer(computing_timer, "compute_pressure");

    p = 0.0;

    std::vector<types::global_dof_index> ldofs(fe_->n_dofs_per_cell());

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        get_system_dof_indices(cell, ldofs);

        for (unsigned int i = 0; i < fe_->n_dofs_per_cell(); ++i)
          if (fe_->system_to_component_index(i).first == 0)
            {
              const double A             = y(ldofs[i]);
              const double base_pressure = compute_pressure_value(
                A,
                cell->material_id(),
                compute_a_d_at_point(cell, support_point(cell, i)));
              p(ldofs[i]) =
                base_pressure +
                external_pressure(
                  {t, cell->center(), cell->material_id(), cell->id()},
                  provider);
            }
      }

    p.compress(VectorOperation::insert);
  }

  template <int dim, int spacedim>
  double
  BloodFlowSystem<dim, spacedim>::pressure(const double       area,
                                           const unsigned int vessel_id) const
  {
    return compute_pressure_value(area, vessel_id);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::set_external_pressure_provider(
    ExternalPressureProvider provider)
  {
    external_pressure_provider = std::move(provider);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::clear_external_pressure_provider()
  {
    external_pressure_provider = {};
  }

  template <int dim, int spacedim>
  double
  BloodFlowSystem<dim, spacedim>::external_pressure(
    const PressureEvaluationPoint &evaluation) const
  {
    if (!external_pressure_provider)
      return 0.0;
    return external_pressure_provider(evaluation);
  }

  template <int dim, int spacedim>
  double
  BloodFlowSystem<dim, spacedim>::pressure(const double           area,
                                           const unsigned int     vessel_id,
                                           const double           t,
                                           const Point<spacedim> &point) const
  {
    return compute_physical_pressure(
      area, vessel_id, vessel_map.at(vessel_id).a_d, t, point);
  }

  template <int dim, int spacedim>
  double
  BloodFlowSystem<dim, spacedim>::pressure_derivative(
    const double       area,
    const unsigned int vessel_id) const
  {
    return compute_pressure_derivative(area, vessel_id);
  }

  template <int dim, int spacedim>
  double
  BloodFlowSystem<dim, spacedim>::wave_speed(const double       area,
                                             const unsigned int vessel_id) const
  {
    return compute_wave_speed(area, vessel_id);
  }

  template <int dim, int spacedim>
  const typename BloodFlowSystem<dim, spacedim>::VesselProperties &
  BloodFlowSystem<dim, spacedim>::vessel_properties(
    const unsigned int vessel_id) const
  {
    return vessel_map.at(vessel_id);
  }


  // ============================================================================
  // output_results
  //
  // DataOut needs a ghosted vector over the FE range: it evaluates on every
  // locally owned cell, whose DoFs are all owned, but the ghosted form is what
  // the interface expects and it costs one exchange per field.  Each rank
  // writes its own .vtu and rank 0 records the .pvtu/.pvd, which is the
  // standard deal.II parallel output path -- no gathering to one rank.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::output_results(const VectorType &y,
                                                 const VectorType &pressure_vec,
                                                 const unsigned int cycle) const
  {
    TimerOutput::Scope timer(computing_timer, "output_results");

    const std::string dir =
      output_directory + (output_directory.empty() ? "" : "/");

    // Restrict each field to the FE range and give it ghost entries.
    auto to_fe_ghosted = [this](const VectorType &src) {
      VectorType owned(locally_owned_fe_dofs, mpi_communicator_);
      for (const auto i : locally_owned_fe_dofs)
        owned(i) = src(system_dof_index(i));
      owned.compress(VectorOperation::insert);

      VectorType ghosted(locally_owned_fe_dofs,
                         locally_relevant_fe_dofs,
                         mpi_communicator_);
      ghosted = owned;
      return ghosted;
    };

    const VectorType sol_fe = to_fe_ghosted(y);
    const VectorType p_fe   = to_fe_ghosted(pressure_vec);

    DataOut<dim, spacedim> data_out;
    data_out.attach_dof_handler(dof_handler_);

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interp(4, DataComponentInterpretation::component_is_scalar);

    std::vector<std::string> names = {"area", "velocity", "A_hat", "U_hat"};
    data_out.add_data_vector(sol_fe,
                             names,
                             DataOut<dim, spacedim>::type_dof_data,
                             interp);

    names = {"pressure", "unused", "unused_p2", "unused_p3"};
    data_out.add_data_vector(p_fe,
                             names,
                             DataOut<dim, spacedim>::type_dof_data,
                             interp);

    // Which rank owns which cell, so the partition is visible in the output.
    Vector<double> subdomain(triangulation_.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation_.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    Vector<double> vessel_ids(triangulation_.n_active_cells());
    for (const auto &cell : triangulation_.active_cell_iterators())
      if (cell->is_locally_owned())
        vessel_ids(cell->active_cell_index()) = cell->material_id();
    data_out.add_data_vector(vessel_ids, "vessel_id");

    data_out.build_patches();

    // Writes <name>-<cycle>.<rank>.vtu on each rank plus the .pvtu record, and
    // appends the time step to the .pvd on rank 0.
    static std::vector<std::pair<double, std::string>> pvd_records;
    const std::string pvtu = data_out.write_vtu_with_pvtu_record(
      dir, output_filename, cycle, mpi_communicator_, 5);

    if (this_mpi_process == 0)
      {
        pvd_records.emplace_back(time, pvtu);
        std::ofstream pvd(dir + output_filename + ".pvd");
        DataOutBase::write_pvd_record(pvd, pvd_records);
        pcout << "  Wrote <" << pvtu << ">" << std::endl;
      }
  }

  // ============================================================================
  // compute_errors
  //
  // integrate_difference() fills one entry per locally owned cell and
  // compute_global_error() does the reduction across ranks, so the printed
  // numbers are the global ones.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::compute_errors(const unsigned int k)
  {
    TimerOutput::Scope timer(computing_timer, "compute_errors");

    // Existing parameter files describe only the physical (A,U) fields.  The
    // HDG state now also contains trace fields, so such an exact solution is
    // not dimensionally compatible with the four-component finite element.
    // Keep those runs usable and reserve error integration for a matching
    // manufactured solution.
    if (exact_solution.n_components != fe_->n_components())
      {
        pcout << "Skipping error computation: exact solution has "
              << exact_solution.n_components << " components, while the finite"
              << " element has " << fe_->n_components() << "." << std::endl;
        return;
      }

    // The FE has 4 components (A, U, A_hat, U_hat), so the masks select out of
    // 4 and `exact_solution` must be a 4-component Function; components 2,3 are
    // masked out.  Meaningful only for manufactured-solution verification runs.
    const ComponentSelectFunction<spacedim> area_mask(0, 1.0, 4);
    const ComponentSelectFunction<spacedim> vel_mask(1, 1.0, 4);

    Vector<float> diff(triangulation_.n_active_cells());

    VectorType cell_sol(locally_owned_fe_dofs, mpi_communicator_);
    for (const auto i : locally_owned_fe_dofs)
      cell_sol(i) = solution(system_dof_index(i));
    cell_sol.compress(VectorOperation::insert);

    VectorType cell_sol_ghosted(locally_owned_fe_dofs,
                                locally_relevant_fe_dofs,
                                mpi_communicator_);
    cell_sol_ghosted = cell_sol;

    exact_solution.set_time(time);

    auto l2_error = [&](const ComponentSelectFunction<spacedim> &mask) {
      VectorTools::integrate_difference(dof_handler_,
                                        cell_sol_ghosted,
                                        exact_solution,
                                        diff,
                                        QGauss<dim>(fe_degree + 3),
                                        VectorTools::L2_norm,
                                        &mask);
      return VectorTools::compute_global_error(triangulation_,
                                               diff,
                                               VectorTools::L2_norm);
    };
    auto h1_error = [&](const ComponentSelectFunction<spacedim> &mask) {
      VectorTools::integrate_difference(dof_handler_,
                                        cell_sol_ghosted,
                                        exact_solution,
                                        diff,
                                        QGauss<dim>(fe_degree + 3),
                                        VectorTools::H1_seminorm,
                                        &mask);
      return VectorTools::compute_global_error(triangulation_,
                                               diff,
                                               VectorTools::H1_seminorm);
    };

    const double AL2 = l2_error(area_mask);
    const double AH1 = h1_error(area_mask);
    const double UL2 = l2_error(vel_mask);
    const double UH1 = h1_error(vel_mask);

    static double prev_AL2 = 0, prev_AH1 = 0, prev_UL2 = 0, prev_UH1 = 0;

    auto rate = [&](double prev, double cur) -> double {
      return (k == 0 || prev == 0.0) ? 0.0 :
                                       std::log(prev / cur) / std::log(2.0);
    };

    pcout << std::scientific << std::setprecision(3) << "=== Errors t=" << time
          << " cycle " << k + 1 << " ===\n"
          << " A  L2=" << AL2 << " rate=" << rate(prev_AL2, AL2) << "\n"
          << " A  H1=" << AH1 << " rate=" << rate(prev_AH1, AH1) << "\n"
          << " U  L2=" << UL2 << " rate=" << rate(prev_UL2, UL2) << "\n"
          << " U  H1=" << UH1 << " rate=" << rate(prev_UH1, UH1) << "\n"
          << " DoFs cell="
          << Utilities::MPI::sum<types::global_dof_index>(
               cell_dofs_owned.n_elements(), mpi_communicator_)
          << " trace="
          << Utilities::MPI::sum<types::global_dof_index>(
               trace_dofs_owned.n_elements(), mpi_communicator_)
          << " total=" << n_total_dofs << "\n"
          << std::string(60, '=') << "\n";

    prev_AL2 = AL2;
    prev_AH1 = AH1;
    prev_UL2 = UL2;
    prev_UH1 = UH1;
  }

  // ============================================================================
  // check_mass_conservation  ->  mass_conservation.csv
  //
  // Compares dV/dt (finite difference between consecutive outputs of the
  // integrated cross-sectional area) with the net flux through the inlet and
  // the terminals.  The reductions are collective; only rank 0 writes.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::check_mass_conservation(const VectorType &y,
                                                          const double t) const
  {
    TimerOutput::Scope timer(computing_timer, "check_mass_conservation");

    // Force y_fe_relevant to reflect *this* y, not whatever trial iterate IDA
    // last evaluated a residual/Jacobian at.  Without this, the volume
    // integral below can read a stale snapshot, producing exact-zero dV/dt on
    // some steps and an overshooting "catch-up" jump on the next.
    update_ghosted_vectors(y);

    const FEValuesExtractors::Scalar area_extr(0);
    const QGauss<dim>                quad(fe_degree + 2);
    FEValues<dim, spacedim> fev(*fe_, quad, update_values | update_JxW_values);

    std::vector<double> A_q(quad.size());
    double              local_volume = 0.0;
    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;
        fev.reinit(cell);
        fev[area_extr].get_function_values(y_fe_relevant, A_q);
        for (unsigned int q = 0; q < quad.size(); ++q)
          local_volume += A_q[q] * fev.JxW(q);
      }
    const double total_volume =
      Utilities::MPI::sum(local_volume, mpi_communicator_);

    double local_net_flux = 0.0;
    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (!cell->face(f)->at_boundary())
              continue;
            if (is_junction_face(cell->id(), f))
              continue;

            double A_hat = 0.0, U_hat = 0.0;
            get_face_trace(y_relevant, cell, f, A_hat, U_hat);
            const double Q            = A_hat * U_hat;
            const double outward_sign = f == 1 ? 1. : -1.;
            local_net_flux -= outward_sign * Q;
          }
      }
    const double net_flux =
      Utilities::MPI::sum(local_net_flux, mpi_communicator_);

    if (!mass_initialized)
      {
        mass_prev_volume = total_volume;
        mass_prev_t      = t;
        mass_initialized = true;
      }

    const double dt = t - mass_prev_t;
    const double dVdt =
      (dt > 0.0) ? (total_volume - mass_prev_volume) / dt : 0.0;
    const double imbalance = dVdt - net_flux;

    // Normalise against a slowly varying flow scale (running peak |net_flux|),
    // not the instantaneous value: net_flux crosses zero every cardiac cycle,
    // and dividing by it there would blow the ratio up for a tiny imbalance.
    mass_flow_scale = std::max(mass_flow_scale, std::abs(net_flux));
    const double rel_imbalance =
      std::abs(imbalance) / std::max(1e-12, mass_flow_scale);

    if (this_mpi_process == 0 && mass_log.is_open())
      {
        csv_write_row(
          mass_log, t, total_volume, dVdt, net_flux, imbalance, rel_imbalance);
      }

    // Two independent gates: relative-to-peak-flow AND an absolute floor, so
    // a genuine problem cannot hide behind either one.
    constexpr double rel_imbalance_warn_tol = 0.20;
    constexpr double abs_imbalance_warn_tol = 1e-5;
    if (rel_imbalance > rel_imbalance_warn_tol &&
        std::abs(imbalance) > abs_imbalance_warn_tol)
      pcout << "  [mass check WARNING] t=" << t << " dV/dt=" << dVdt
            << " net_flux=" << net_flux << " imbalance=" << imbalance << " ("
            << rel_imbalance * 100.0 << "% of peak flow)" << std::endl;

    mass_prev_volume = total_volume;
    mass_prev_t      = t;
  }

  // ============================================================================
  // open_output_files
  //
  // Opens the per-vessel probes and mass_conservation.csv, and resets the
  // running state of the mass check and of the BDF-order histogram.  Call once
  // per simulation (per refinement cycle), before ida.solve_dae().
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::open_output_files()
  {
    open_csv_files();

    mass_initialized = false;
    mass_prev_volume = 0.0;
    mass_prev_t      = 0.0;
    mass_flow_scale  = 0.0;

    if (mass_log.is_open())
      mass_log.close();
    collective_csv_action(mpi_communicator_, [&]() {
      if (this_mpi_process == 0 && !output_mass_conservation_csv_file.empty())
        {
          mass_log.open(prepare_csv_path(output_directory,
                                         output_mass_conservation_csv_file),
                        std::ios::trunc);
          AssertThrow(mass_log,
                      ExcMessage("Cannot open " +
                                 output_mass_conservation_csv_file));
          csv_write_row(mass_log,
                        "t",
                        "total_volume",
                        "dVdt",
                        "net_flux",
                        "imbalance",
                        "rel_imbalance");
        }
    });

    bdf_steps.fill(0);
    bdf_prev_nst             = 0;
    ida_jacobian_evaluations = ida_linear_solves = ida_linear_iterations = 0;
    accepted_h_min                 = std::numeric_limits<double>::infinity();
    accepted_h_max                 = 0.;
    accepted_step_extrema_complete = true;
    previous_console_statistics    = IDAOutputRecord{};
    statistics_header_written      = false;
    statistics_start               = std::chrono::steady_clock::now();

    ida_history.clear();
    prev_nni  = 0;
    prev_ncfn = 0;
    prev_netf = 0;
    prev_nre  = 0;
    prev_nls  = 0;

    if (ida_history_log.is_open())
      ida_history_log.close();
    collective_csv_action(mpi_communicator_, [&]() {
      if (this_mpi_process == 0 && !output_ida_step_history_csv_file.empty())
        {
          ida_history_log.open(
            prepare_csv_path(output_directory,
                             output_ida_step_history_csv_file),
            std::ios::trunc);
          AssertThrow(ida_history_log,
                      ExcMessage("Cannot open " +
                                 output_ida_step_history_csv_file));
          ida_history_log
            << "output_time,ida_time,steps_interval,steps_cumulative,"
               "nonlin_iters_interval,nonlin_iters_cumulative,"
               "nonlin_iters_per_step,nonlin_fails_interval,"
               "err_test_fails_interval,residual_evals_interval,"
               "lin_setups_interval,last_step_size,last_order,"
               "nonlin_fails_cumulative,err_test_fails_cumulative,residual_evals_cumulative,"
               "lin_setups_cumulative,jacobian_evals_cumulative,linear_solves_cumulative,"
               "linear_iters_cumulative,min_step_size,max_step_size,step_extrema_available\n";
        }
    });

    is_prev_nst = is_prev_nni = is_prev_ncfn = is_prev_netf = is_prev_nls = 0;
    is_t_accepted = ida_parameters.initial_time;

    if (ida_internal_log.is_open())
      ida_internal_log.close();
    collective_csv_action(mpi_communicator_, [&]() {
      if (this_mpi_process == 0 && !output_ida_internal_steps_csv_file.empty())
        {
          ida_internal_log.open(
            prepare_csv_path(output_directory,
                             output_ida_internal_steps_csv_file),
            std::ios::trunc);
          AssertThrow(ida_internal_log,
                      ExcMessage("Cannot open IDA internal steps CSV."));
          ida_internal_log
            << "step,t,h,order,nonlin_iters,nonlin_iters_cumulative,"
               "nonlin_conv_fails,err_test_fails,jacobian_setups\n";
        }
    });
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::poll_ida_internal_step(
    const SUNDIALS::IDA<VectorType> &ida,
    const bool                       resync_time,
    const bool                       require_consecutive_steps)
  {
    void *mem = internal_ida_access::get_ida_mem(ida);
    if (mem == nullptr)
      return;

    long int nst = 0;
    IDAGetNumSteps(mem, &nst);
    if (nst == is_prev_nst)
      {
        if (resync_time)
          AssertThrow(IDAGetCurrentTime(mem, &is_t_accepted) == 0,
                      ExcMessage("Cannot resynchronize accepted IDA time."));
        return; // no new accepted step
      }
    if (nst != is_prev_nst + 1)
      {
        AssertThrow(
          !require_consecutive_steps,
          ExcMessage(
            "Accepted-step observation gap: poll IDA from every residual and output callback."));
        accepted_step_extrema_complete = false;
      }

    long int nni = 0, ncfn = 0, netf = 0, nls = 0;
    int      klast = 0;
    double   hlast = 0.0;
    IDAGetNumNonlinSolvIters(mem, &nni);
    IDAGetNumNonlinSolvConvFails(mem, &ncfn);
    IDAGetNumErrTestFails(mem, &netf);
    IDAGetNumLinSolvSetups(mem, &nls);
    IDAGetLastOrder(mem, &klast);
    IDAGetLastStep(mem, &hlast);
    const double accepted_h = std::abs(hlast);
    AssertThrow(std::isfinite(accepted_h) && accepted_h > 0.,
                ExcInternalError());
    accepted_h_min = std::min(accepted_h_min, accepted_h);
    accepted_h_max = std::max(accepted_h_max, accepted_h);
    if (klast >= 1 && klast <= 5)
      bdf_steps[klast] += nst - std::max(is_prev_nst, bdf_prev_nst);

    // Inside the residual, IDA's tn is the *trial* time, so accumulate t += h.
    // In output_step / after solve_dae, tn is exact -> resync.
    is_t_accepted += hlast * static_cast<double>(nst - is_prev_nst);
    if (resync_time)
      IDAGetCurrentTime(mem, &is_t_accepted);

    if (this_mpi_process == 0 && ida_internal_log.is_open())
      ida_internal_log << nst << ',' << std::scientific << std::setprecision(12)
                       << is_t_accepted << ',' << std::setprecision(6) << hlast
                       << ',' << klast << ',' << (nni - is_prev_nni) << ','
                       << nni << ',' << (ncfn - is_prev_ncfn) << ','
                       << (netf - is_prev_netf) << ',' << (nls - is_prev_nls)
                       << '\n';

    is_prev_nst  = nst;
    is_prev_nni  = nni;
    is_prev_ncfn = ncfn;
    is_prev_netf = netf;
    is_prev_nls  = nls;
  }

  // ============================================================================
  // write_output_step
  //
  // One row in every HDG_IDA_Vessel_<vid>.csv owned by this rank and one row
  // in mass_conservation.csv.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::write_output_step(const double      t,
                                                    const VectorType &sol,
                                                    const VectorType *ydot)
  {
    write_csv_row(t, sol, ydot);
    check_mass_conservation(sol, t);
  }

  // ============================================================================
  // record_ida_step
  //
  // Called at the top of output_step, i.e. once per output time requested
  // through the output period.  deal.II's solve_dae() lets IDA take as many
  // internal (adaptive) steps as it needs to pass the requested time, then
  // interpolates the solution back to it.  The difference of IDA's cumulative
  // counters between two consecutive calls is therefore exactly the work IDA
  // did to get from the previous output time to this one:
  //
  //   steps_interval        : internal BDF steps accepted in the interval
  //   nonlin_iters_interval : Newton iterations over all of those steps
  //                           (including iterations of rejected attempts)
  //
  // The record is kept in ida_history and streamed to ida_step_history.csv.
  // Per-residual polling attributes the runner's BDF histogram to each
  // accepted step. Output-only library clients retain grouped attribution to
  // the last accepted order; either path counts each step once.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::record_ida_step(
    const SUNDIALS::IDA<VectorType> &ida)
  {
    void *mem = internal_ida_access::get_ida_mem(ida);
    if (mem == nullptr)
      return;
    double t = 0.0;
    IDAGetCurrentTime(mem, &t);
    record_ida_step(ida, t);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::record_ida_step(
    const SUNDIALS::IDA<VectorType> &ida,
    const double                     t)
  {
    void *mem = internal_ida_access::get_ida_mem(ida);
    if (mem == nullptr)
      return;

    IDAOutputRecord rec = current_ida_statistics(ida, t);
    const auto nst = rec.steps_cumulative, nni = rec.nonlin_iters_cumulative,
               ncfn            = rec.nonlin_fails_cumulative,
               netf            = rec.err_test_fails_cumulative,
               nre             = rec.residual_evals_cumulative,
               nls             = rec.lin_setups_cumulative;
    const long int delta_steps = nst - bdf_prev_nst;
    // Preserve grouped output-only histories for library clients that do not
    // poll each residual; the runner already attributed every accepted step.
    if (nst > is_prev_nst && rec.last_order >= 1 && rec.last_order <= 5)
      bdf_steps[rec.last_order] += nst - std::max(is_prev_nst, bdf_prev_nst);
    bdf_prev_nst = nst;

    // ---- per-output-time record ---------------------------------------------

    rec.steps_interval          = delta_steps;
    rec.steps_cumulative        = nst;
    rec.nonlin_iters_interval   = nni - prev_nni;
    rec.nonlin_iters_cumulative = nni;
    rec.nonlin_fails_interval   = ncfn - prev_ncfn;
    rec.err_test_fails_interval = netf - prev_netf;
    rec.residual_evals_interval = nre - prev_nre;
    rec.lin_setups_interval     = nls - prev_nls;
    ida_history.push_back(rec);

    prev_nni  = nni;
    prev_ncfn = ncfn;
    prev_netf = netf;
    prev_nre  = nre;
    prev_nls  = nls;

    if (this_mpi_process == 0 && ida_history_log.is_open())
      {
        const double iters_per_step =
          rec.steps_interval > 0 ?
            static_cast<double>(rec.nonlin_iters_interval) /
              static_cast<double>(rec.steps_interval) :
            0.0;

        ida_history_log << std::scientific << std::setprecision(10)
                        << rec.output_time << ',' << rec.ida_time << ','
                        << rec.steps_interval << ',' << rec.steps_cumulative
                        << ',' << rec.nonlin_iters_interval << ','
                        << rec.nonlin_iters_cumulative << ','
                        << std::setprecision(4) << iters_per_step << ','
                        << rec.nonlin_fails_interval << ','
                        << rec.err_test_fails_interval << ','
                        << rec.residual_evals_interval << ','
                        << rec.lin_setups_interval << ','
                        << std::setprecision(6) << rec.last_step_size << ','
                        << rec.last_order << ',' << rec.nonlin_fails_cumulative
                        << ',' << rec.err_test_fails_cumulative << ','
                        << rec.residual_evals_cumulative << ','
                        << rec.lin_setups_cumulative << ','
                        << rec.jacobian_evals_cumulative << ','
                        << rec.linear_solves_cumulative << ','
                        << rec.linear_iters_cumulative << ','
                        << rec.minimum_step_size << ',' << rec.maximum_step_size
                        << ',' << rec.step_extrema_available << '\n';
        ida_history_log.flush();
      }
  }

  template <int dim, int spacedim>
  typename BloodFlowSystem<dim, spacedim>::IDAOutputRecord
  BloodFlowSystem<dim, spacedim>::current_ida_statistics(
    const SUNDIALS::IDA<VectorType> &ida,
    const double                     output_time) const
  {
    void *memory = internal_ida_access::get_ida_mem(ida);
    AssertThrow(memory != nullptr,
                ExcMessage("IDA statistics requested before setup."));
    IDAOutputRecord record;
    record.output_time = output_time;
    AssertThrow(
      IDAGetNumSteps(memory, &record.steps_cumulative) == 0 &&
        IDAGetNumNonlinSolvIters(memory, &record.nonlin_iters_cumulative) ==
          0 &&
        IDAGetNumNonlinSolvConvFails(memory, &record.nonlin_fails_cumulative) ==
          0 &&
        IDAGetNumErrTestFails(memory, &record.err_test_fails_cumulative) == 0 &&
        IDAGetNumResEvals(memory, &record.residual_evals_cumulative) == 0 &&
        IDAGetNumLinSolvSetups(memory, &record.lin_setups_cumulative) == 0 &&
        IDAGetCurrentTime(memory, &record.ida_time) == 0,
      ExcMessage("Cannot read native IDA statistics."));
    if (record.steps_cumulative > 0)
      {
        AssertThrow(IDAGetLastStep(memory, &record.last_step_size) == 0 &&
                      IDAGetLastOrder(memory, &record.last_order) == 0,
                    ExcInternalError());
        record.step_extrema_available = accepted_step_extrema_complete &&
                                        is_prev_nst == record.steps_cumulative;
        if (record.step_extrema_available)
          {
            record.minimum_step_size = accepted_h_min;
            record.maximum_step_size = accepted_h_max;
          }
      }
    record.jacobian_evals_cumulative = ida_jacobian_evaluations;
    record.linear_solves_cumulative  = ida_linear_solves;
    record.linear_iters_cumulative   = ida_linear_iterations;
    return record;
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::print_current_statistics(
    const SUNDIALS::IDA<VectorType> &ida,
    const double                     output_time)
  {
    if (!print_statistics || this_mpi_process != 0)
      return;
    const auto  current  = current_ida_statistics(ida, output_time);
    const auto &previous = previous_console_statistics;
    const auto  counts   = [](const long int now, const long int before) {
      return std::to_string(now - before) + "/" + std::to_string(now);
    };
    const auto step = [&](const double value) {
      if (current.steps_cumulative == 0 || value == 0.)
        return std::string("-");
      std::ostringstream text;
      text << std::scientific << std::setprecision(3) << value;
      return text.str();
    };
    std::ostringstream table;
    if (!statistics_header_written)
      {
        table
          << "IDA current statistics: d/t = since previous output / cumulative; h extrema = accepted steps\n"
          << "|        t_output |           t_IDA |     h_last |      h_min |      h_max | q |   steps d/t |  Newton d/t |       J d/t |       R d/t |      LS d/t |  Krylov d/t |     err d/t |   Nfail d/t |     wall_s |\n";
        statistics_header_written = true;
      }
    table << std::scientific << std::setprecision(9) << "| " << std::setw(15)
          << current.output_time << " | " << std::setw(15) << current.ida_time
          << " | " << std::setw(10) << step(current.last_step_size) << " | "
          << std::setw(10) << step(current.minimum_step_size) << " | "
          << std::setw(10) << step(current.maximum_step_size) << " | "
          << std::setw(1) << current.last_order;
    for (const auto &value :
         {counts(current.steps_cumulative, previous.steps_cumulative),
          counts(current.nonlin_iters_cumulative,
                 previous.nonlin_iters_cumulative),
          counts(current.jacobian_evals_cumulative,
                 previous.jacobian_evals_cumulative),
          counts(current.residual_evals_cumulative,
                 previous.residual_evals_cumulative),
          counts(current.linear_solves_cumulative,
                 previous.linear_solves_cumulative),
          counts(current.linear_iters_cumulative,
                 previous.linear_iters_cumulative),
          counts(current.err_test_fails_cumulative,
                 previous.err_test_fails_cumulative),
          counts(current.nonlin_fails_cumulative,
                 previous.nonlin_fails_cumulative)})
      table << " | " << std::setw(11) << value;
    table << std::setprecision(3) << " | " << std::setw(10)
          << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                           statistics_start)
               .count()
          << " |\n";
    pcout << table.str() << std::flush;
    previous_console_statistics = current;
  }

  // ============================================================================
  // write_ida_statistics  ->  ida_statistics.csv
  //
  // ida_mem is still alive after solve_dae() returns (it is freed in the IDA
  // destructor or the next reset), so the cumulative counters can be read
  // here.  One row per simulation, appended, so successive runs (e.g. a sweep
  // over the maximum BDF order) accumulate in the same file.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::write_ida_statistics(
    const SUNDIALS::IDA<VectorType> &ida,
    const double                     solve_wall_time) const
  {
    void *mem = internal_ida_access::get_ida_mem(ida);

    long int nst = 0, netf = 0, nni = 0, ncfn = 0;
    if (mem != nullptr)
      {
        IDAGetNumSteps(mem, &nst);                // accepted steps
        IDAGetNumErrTestFails(mem, &netf);        // error-test failures
        IDAGetNumNonlinSolvIters(mem, &nni);      // nonlinear iterations
        IDAGetNumNonlinSolvConvFails(mem, &ncfn); // nonlinear conv. failures
      }

    // Collective: the slowest rank defines the wall time of the solve.
    const double wall_time =
      Utilities::MPI::max(solve_wall_time, mpi_communicator_);

    collective_csv_action(mpi_communicator_, [&]() {
      if (this_mpi_process != 0 || output_ida_statistics_csv_file.empty())
        return;

      // Derived averages over the whole run.
      const std::size_t n_outputs = ida_history.size();
      const double      nonlin_iters_per_step =
        nst > 0 ? static_cast<double>(nni) / static_cast<double>(nst) : 0.0;
      const double steps_per_output =
        n_outputs > 0 ?
          static_cast<double>(nst) / static_cast<double>(n_outputs) :
          0.0;
      const double nonlin_iters_per_output =
        n_outputs > 0 ?
          static_cast<double>(nni) / static_cast<double>(n_outputs) :
          0.0;

      auto csv = csv_append(prepare_csv_path(output_directory,
                                             output_ida_statistics_csv_file),
                            {"max_order",
                             "wall_time",
                             "n_steps",
                             "n_err_test_fails",
                             "n_nonlin_iters",
                             "n_nonlin_conv_fails",
                             "bdf1_steps",
                             "bdf2_steps",
                             "bdf3_steps",
                             "bdf4_steps",
                             "bdf5_steps",
                             "n_outputs",
                             "nonlin_iters_per_step",
                             "steps_per_output",
                             "nonlin_iters_per_output"});
      csv_write_row(csv,
                    ida_parameters.maximum_order,
                    wall_time,
                    nst,
                    netf,
                    nni,
                    ncfn,
                    bdf_steps[1],
                    bdf_steps[2],
                    bdf_steps[3],
                    bdf_steps[4],
                    bdf_steps[5],
                    n_outputs,
                    nonlin_iters_per_step,
                    steps_per_output,
                    nonlin_iters_per_output);
    });
  }

  // ============================================================================
  // finalize_output
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::finalize_output(
    const SUNDIALS::IDA<VectorType> &ida,
    const double                     solve_wall_time)
  {
    close_csv_files();
    if (mass_log.is_open())
      mass_log.close();
    if (ida_history_log.is_open())
      ida_history_log.close();
    poll_ida_internal_step(ida, /*resync_time=*/true); // catch the last step
    if (ida_internal_log.is_open())
      ida_internal_log.close();
    write_ida_statistics(ida, solve_wall_time);
  }

  // ============================================================================
  // write_scaling_timings  ->  scaling_timings.csv
  //
  // Accumulated wall-clock time per TimerOutput section, tagged with the
  // process count.  Run the executable with mpirun -np 1, 2, 4, ...; each
  // invocation appends one row per section to the same CSV.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::write_scaling_timings() const
  {
    const auto summary = computing_timer.get_summary_data(
      TimerOutput::OutputData::total_wall_time);

    collective_csv_action(mpi_communicator_, [&]() {
      if (this_mpi_process != 0 || output_scaling_timings_csv_file.empty())
        return;

      auto csv = csv_append(prepare_csv_path(output_directory,
                                             output_scaling_timings_csv_file),
                            {"n_mpi_processes", "section", "wall_time"});
      for (const auto &entry : summary)
        csv_write_row(csv, n_mpi_processes, entry.first, entry.second);
    });
  }

  // Explicit instantiation
  template class BloodFlowSystem<1, 3>;

} // namespace MetricFlowX
