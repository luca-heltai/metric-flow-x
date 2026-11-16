// SPDX-License-Identifier: MIT
// A resolved small-area tube must retain its reference pressure and steady
// flow.
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <metric_flow_x/blood_flow_system.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "tests.h"

using namespace MetricFlowX;
using Flow = BloodFlowSystem<1, 3>;

constexpr double reference_pressure = 1017.;
constexpr double velocity           = .02;

void
parameters(const std::string &filename,
           const std::string &floor,
           const std::string &flux = "HLL")
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::ofstream file(filename);
      file << "subsection Metric Flow Parameters\n"
           << " set Density (rho) = 1060\n"
           << " set Viscosity coefficient (mu) = 0\n"
           << " set Profile constant for friction term (xi) = 0\nend\n"
           << "subsection MetricFlowSystem<1, 3>\n"
           << " set Wall model = Square root\n"
           << " set Outlet boundary condition type = RCR\n"
           << " set Finite element degree = 1\n"
           << " set Verbosity (console depth) = 0\n"
           << " set Numerical flux type = " << flux << '\n';
      if (!floor.empty())
        file << " set Minimum positive area = " << floor << '\n';
      file << "end\n";
    }
  MPI_Barrier(MPI_COMM_WORLD);
}

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
      cell->face(face)->set_boundary_id(face + 1);
  mesh.refine_global(1);
  GridTools::partition_triangulation_zorder(
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), mesh, false);
  flow.triangulation().create_triangulation(
    TriangulationDescription::Utilities::create_description_from_triangulation(
      mesh, MPI_COMM_WORLD));
}

Flow::NetworkPhysics
physics(const double area)
{
  Flow::NetworkPhysics input;
  auto                &v = input.vessels[0];
  v.a0 = v.a_d            = area;
  v.r_d                   = std::sqrt(area / numbers::PI);
  v.E                     = 5.e5;
  v.h_wall                = 1.e-6;
  v.L                     = .1;
  v.p_d                   = 1000.;
  v.p0                    = 17.;
  const double resistance = 100. / area;
  input.terminals         = {
    {1, {0., resistance, 0., reference_pressure + 100. * velocity, "source"}},
    {2, {0., resistance, 0., reference_pressure - 100. * velocity, "drain"}}};
  return input;
}

struct Evaluation
{
  std::vector<double>                  residual, jacobian;
  std::vector<types::global_dof_index> owned;
};

Evaluation
evaluate(const double       area,
         const std::string &floor,
         const std::string &flux,
         const bool         check_steady,
         const bool         check_derivative)
{
  parameters("minimum_area.prm", floor, flux);
  ParameterAcceptor::clear();
  Flow flow(MPI_COMM_WORLD);
  flow.initialize_params("minimum_area.prm", "");
  const double expected_floor = floor.empty() ? 1.e-10 : std::stod(floor);
  AssertThrow(flow.get_minimum_positive_area() == expected_floor,
              ExcMessage("The numerical area parameter was not parsed"));
  make_mesh(flow);
  flow.setup(physics(area));
  AssertThrow(flow.get_minimum_positive_area() == expected_floor,
              ExcMessage("In-memory physics replaced a numerical parameter"));
  auto y = flow.make_state(), rate = flow.make_state();
  for (const auto component :
       {Flow::Component::area, Flow::Component::area_trace})
    for (const auto i : flow.component_dofs(component))
      if (flow.locally_owned_dofs().is_element(i))
        y[i] = area;
  for (const auto component :
       {Flow::Component::velocity, Flow::Component::velocity_trace})
    for (const auto i : flow.component_dofs(component))
      if (flow.locally_owned_dofs().is_element(i))
        y[i] = velocity;
  y.compress(VectorOperation::insert);
  auto residual = flow.make_state();
  flow.assemble_residual(.37, y, rate, residual);
  if (check_steady)
    {
      AssertThrow(residual.linfty_norm() < 1.e-8,
                  ExcMessage(
                    "The resolved steady tube violates pressure balance"));
      auto pressure = flow.make_state();
      flow.compute_pressure(y, pressure);
      for (const auto i : flow.component_dofs(Flow::Component::area))
        if (flow.locally_owned_dofs().is_element(i))
          AssertThrow(std::abs(pressure[i] - reference_pressure) < 1.e-10,
                      ExcMessage("Reference tube pressure must equal p0 + pd"));
      const double beta = 4. * std::sqrt(numbers::PI) / 3. * 5.e5 * 1.e-6;
      const double expected_c =
        std::sqrt(beta / (2. * 1060. * std::sqrt(area)));
      AssertThrow(std::abs(flow.wave_speed(area, 0) / expected_c - 1.) < 1.e-12,
                  ExcMessage(
                    "Reference wave speed uses an unresolved area floor"));
    }
  if (area < 1.e-10 && expected_floor == 1.e-10)
    {
      // The old default remains observable. Lowering the numerical floor,
      // rather than changing a radius or a reservoir, resolves this tube.
      AssertThrow(residual.linfty_norm() > 1.,
                  ExcMessage("The legacy default unexpectedly changed"));
    }

  for (const auto i : flow.component_dofs(Flow::Component::area))
    if (flow.locally_owned_dofs().is_element(i))
      y[i] *= 1.2 + .03 * std::sin(i + 1.);
  for (const auto i : flow.component_dofs(Flow::Component::area_trace))
    if (flow.locally_owned_dofs().is_element(i))
      y[i] *= 1.1;
  y.compress(VectorOperation::insert);
  flow.set_external_pressure_provider(
    [](const auto &point) { return 3. + 7. * point.point[0] + point.time; });
  flow.assemble_residual(.37, y, rate, residual);
  flow.assemble_state_jacobian(.37, y, rate);
  const auto &matrix = flow.state_jacobian_matrix();
  if (check_derivative)
    {
      auto direction = flow.make_state();
      for (const auto component :
           {Flow::Component::area, Flow::Component::area_trace})
        for (const auto i : flow.component_dofs(component))
          if (flow.locally_owned_dofs().is_element(i))
            direction[i] = .04 * area * (1. + .1 * std::sin(i + 1.));
      for (const auto component :
           {Flow::Component::velocity, Flow::Component::velocity_trace})
        for (const auto i : flow.component_dofs(component))
          if (flow.locally_owned_dofs().is_element(i))
            direction[i] = .01 * std::cos(i + 1.);
      direction.compress(VectorOperation::insert);
      auto product = flow.make_state();
      matrix.vmult(product, direction);
      constexpr double epsilon = 1.e-5;
      auto             plus = y, minus = y;
      plus.add(epsilon, direction);
      minus.add(-epsilon, direction);
      auto fp = flow.make_state(), fm = flow.make_state();
      flow.assemble_residual(.37, plus, rate, fp);
      flow.assemble_residual(.37, minus, rate, fm);
      fp -= fm;
      fp *= .5 / epsilon;
      fp -= product;
      for (const auto component : {Flow::Component::area,
                                   Flow::Component::velocity,
                                   Flow::Component::area_trace,
                                   Flow::Component::velocity_trace})
        {
          double error = 0., scale = 0.;
          for (const auto i : flow.component_dofs(component))
            if (flow.locally_owned_dofs().is_element(i))
              {
                error += fp[i] * fp[i];
                scale += product[i] * product[i];
              }
          error = std::sqrt(Utilities::MPI::sum(error, MPI_COMM_WORLD));
          scale = std::sqrt(Utilities::MPI::sum(scale, MPI_COMM_WORLD));
          std::ostringstream detail;
          detail << std::scientific << std::setprecision(16)
                 << "Resolved-area Jacobian FD: flux=" << flux
                 << ", component=" << static_cast<unsigned int>(component)
                 << ", area=" << area << ", floor=" << expected_floor
                 << ", error=" << error << ", scale=" << scale
                 << ", relative=" << error / scale;
          AssertThrow(scale > 0. && error / scale < 2.e-6,
                      ExcMessage(detail.str()));
        }
    }
  Evaluation result;
  const auto n = flow.dof_handler().n_dofs();
  result.residual.resize(n);
  result.jacobian.resize(n * n);
  for (const auto row : flow.locally_owned_dofs())
    {
      result.owned.push_back(row);
      result.residual[row] = residual[row];
    }
  auto basis = flow.make_state(), image = flow.make_state();
  for (types::global_dof_index column = 0; column < n; ++column)
    {
      basis = 0.;
      if (flow.locally_owned_dofs().is_element(column))
        basis[column] = 1.;
      basis.compress(VectorOperation::insert);
      matrix.vmult(image, basis);
      for (const auto row : result.owned)
        result.jacobian[row * n + column] = image[row];
    }
  return result;
}

// Hold areas fixed on either side of the numerical floor. This checks only
// velocity columns, not derivatives with respect to a clamped area.
void
rcr_velocity_columns(const double area, const bool capacitive)
{
  parameters("minimum_area_velocity.prm", "1e-10");
  ParameterAcceptor::clear();
  Flow flow(MPI_COMM_WORLD);
  flow.initialize_params("minimum_area_velocity.prm", "");
  make_mesh(flow);
  auto input = physics(area);
  for (auto &[id, rcr] : input.terminals)
    {
      (void)id;
      const double total = rcr.R2;
      rcr.R1             = .3 * total;
      rcr.R2             = .7 * total;
      rcr.C              = capacitive ? 1e-9 : 0.;
    }
  flow.setup(input);
  auto y = flow.make_state(), rate = flow.make_state();
  rate = y = 0.;
  for (const auto component : {Flow::Component::area,
                               Flow::Component::area_trace,
                               Flow::Component::velocity,
                               Flow::Component::velocity_trace})
    for (const auto i : flow.component_dofs(component))
      if (flow.locally_owned_dofs().is_element(i))
        y[i] = component == Flow::Component::area ||
                   component == Flow::Component::area_trace ?
                 area :
                 velocity;
  y.compress(VectorOperation::insert);
  AssertThrow(flow.has_admissible_wall_states(y),
              ExcMessage(
                "Positive below-floor RCR states must remain admissible"));
  flow.assemble_state_jacobian(.37, y, rate);
  for (const types::boundary_id bid : {1u, 2u})
    {
      types::global_dof_index arow  = numbers::invalid_dof_index,
                              ucol  = numbers::invalid_dof_index,
                              pcrow = numbers::invalid_dof_index;
      unsigned int count            = 0;
      for (const auto &cell : flow.dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const auto f : cell->face_indices())
            if (cell->face(f)->at_boundary() &&
                cell->face(f)->boundary_id() == bid)
              {
                ++count;
                std::vector<types::global_dof_index> indices(
                  cell->get_fe().n_dofs_per_cell());
                cell->get_dof_indices(indices);
                for (unsigned int j = 0; j < indices.size(); ++j)
                  {
                    const auto c =
                      cell->get_fe().system_to_component_index(j).first;
                    if (c == 4)
                      pcrow = indices[j];
                    else if ((c == 2 || c == 3) &&
                             cell->get_fe().shape_value(j, Point<1>(f)) > .5)
                      (c == 2 ? arow : ucol) = indices[j];
                  }
              }
      count = Utilities::MPI::sum(count, MPI_COMM_WORLD);
      arow  = Utilities::MPI::min(arow, MPI_COMM_WORLD);
      ucol  = Utilities::MPI::min(ucol, MPI_COMM_WORLD);
      pcrow = Utilities::MPI::min(pcrow, MPI_COMM_WORLD);
      AssertThrow(count == 1 && arow != numbers::invalid_dof_index &&
                    ucol != numbers::invalid_dof_index &&
                    (!capacitive || pcrow != numbers::invalid_dof_index),
                  ExcInternalError());
      auto direction = flow.make_state(), product = flow.make_state();
      direction = 0.;
      if (flow.locally_owned_dofs().is_element(ucol))
        direction[ucol] = 1.;
      direction.compress(VectorOperation::insert);
      flow.state_jacobian_matrix().vmult(product, direction);
      const auto entry = [&](const VectorType             &v,
                             const types::global_dof_index row) {
        return Utilities::MPI::sum(flow.locally_owned_dofs().is_element(row) ?
                                     static_cast<double>(v[row]) :
                                     0.,
                                   MPI_COMM_WORLD);
      };
      const double sign       = bid == 1 ? -1. : 1.;
      const auto  &rcr        = input.terminals.at(bid);
      const double resistance = capacitive ? rcr.R1 : rcr.R1 + rcr.R2;
      // Full F=M*ydot-R: the pressure row has +R_eff*sign*A_raw*dU.
      const double exact = resistance * sign * area;
      AssertThrow(std::isfinite(entry(product, arow)) &&
                    std::abs(entry(product, arow) - exact) <
                      1e-12 * std::abs(exact),
                  ExcMessage("RCR velocity column must use raw trace area"));
      if (capacitive)
        AssertThrow(
          std::abs(entry(product, pcrow) + sign * area) < 1e-12 * area,
          ExcMessage("Capacitor velocity column must use raw trace area"));
      for (const double h : {1e-3, 1e-4, 1e-5})
        {
          auto plus = y, minus = y;
          if (flow.locally_owned_dofs().is_element(ucol))
            {
              plus[ucol]  = y[ucol] + h;
              minus[ucol] = y[ucol] - h;
            }
          plus.compress(VectorOperation::insert);
          minus.compress(VectorOperation::insert);
          auto fp = flow.make_state(), fm = flow.make_state();
          flow.assemble_residual(.37, plus, rate, fp);
          flow.assemble_residual(.37, minus, rate, fm);
          fp -= fm;
          fp *= .5 / h;
          AssertThrow(
            std::abs(entry(fp, arow) - exact) < 2e-6 * std::abs(exact),
            ExcMessage(
              "RCR pressure-row finite difference disagrees with raw flow"));
          if (capacitive)
            AssertThrow(
              std::abs(entry(fp, pcrow) + sign * area) < 2e-6 * area,
              ExcMessage(
                "Capacitor-row finite difference disagrees with raw flow"));
          for (const auto component : {Flow::Component::area,
                                       Flow::Component::velocity,
                                       Flow::Component::area_trace,
                                       Flow::Component::velocity_trace,
                                       Flow::Component::capacitor_pressure})
            {
              double error = 0., scale = 0.;
              for (const auto i : flow.component_dofs(component))
                if (flow.locally_owned_dofs().is_element(i))
                  {
                    const double difference = fp[i] - product[i];
                    error += difference * difference;
                    scale += product[i] * product[i];
                  }
              error = std::sqrt(Utilities::MPI::sum(error, MPI_COMM_WORLD));
              scale = std::sqrt(Utilities::MPI::sum(scale, MPI_COMM_WORLD));
              AssertThrow(std::isfinite(error) && std::isfinite(scale) &&
                            error <= 2e-6 * std::max(scale, 1e-300),
                          ExcMessage(
                            "Full residual velocity-column FD failed"));
            }
        }
    }
}

void
identical(const Evaluation &a, const Evaluation &b)
{
  const bool same =
    a.owned == b.owned && a.residual == b.residual && a.jacobian == b.jacobian;
  AssertThrow(Utilities::MPI::min(same ? 1u : 0u, MPI_COMM_WORLD) == 1,
              ExcMessage(
                "An inactive or default floor changed the residual/Jacobian"));
}

void
rejected(const std::string &value,
         const bool         in_memory,
         const bool         different_ranks)
{
  parameters("minimum_invalid0.prm", value);
  parameters("minimum_invalid1.prm", "2e-14");
  ParameterAcceptor::clear();
  Flow flow(MPI_COMM_WORLD);
  if (in_memory)
    make_mesh(flow);
  const auto cells  = flow.triangulation().n_global_active_cells();
  bool       caught = false, parsed = false;
  try
    {
      const bool other = different_ranks &&
                         Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0;
      flow.initialize_params(other ? "minimum_invalid1.prm" :
                                     "minimum_invalid0.prm",
                             "");
      parsed = true;
      if (in_memory)
        flow.setup(physics(4.e-11));
      else
        flow.setup();
    }
  catch (const std::exception &error)
    {
      caught =
        !parsed || std::string(error.what()).find("Minimum positive area") !=
                     std::string::npos;
    }
  AssertThrow(Utilities::MPI::min(caught ? 1u : 0u, MPI_COMM_WORLD) == 1,
              ExcMessage("Invalid area floors must be rejected on every rank"));
  AssertThrow(flow.triangulation().n_global_active_cells() == cells &&
                flow.dof_handler().n_dofs() == 0,
              ExcMessage("Invalid numerical controls changed mesh/DoFs"));
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  for (const std::string flux : {"HLL", "LAX_FRIEDRICHS"})
    {
      // HLL supplies the independent full-residual FD oracle. Both fluxes
      // retain the full-matrix checks for inactive/default floors below.
      const bool check_derivative = flux == "HLL";
      const auto resolved =
        evaluate(4.e-11, "1e-14", flux, true, check_derivative);
      identical(resolved, evaluate(4.e-11, "1e-15", flux, true, false));
      identical(evaluate(4.e-11, "", flux, false, false),
                evaluate(4.e-11, "1e-10", flux, false, false));
      identical(evaluate(1.e-4, "", flux, true, check_derivative),
                evaluate(1.e-4, "1e-10", flux, true, false));
    }
  for (const double area : {4.e-11, 1.e-4})
    for (const bool capacitive : {false, true})
      rcr_velocity_columns(area, capacitive);
  for (const bool in_memory : {false, true})
    {
      for (const std::string value : {"0", "nan", "inf", "-inf"})
        rejected(value, in_memory, false);
      if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
        {
          rejected("1e-14", in_memory, true);
          rejected("0", in_memory, true);
        }
    }
  ParameterAcceptor::clear();
  Flow flow(MPI_COMM_WORLD);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::ofstream file("minimum_file_setup.prm");
      file << "include " << PRM_DIR "constant.prm" << '\n'
           << "subsection MetricFlowSystem<1, 3>\n"
           << " set Minimum positive area = 1e-14\n"
           << " set Verbosity (console depth) = 0\nend\n";
    }
  MPI_Barrier(MPI_COMM_WORLD);
  flow.initialize_params("minimum_file_setup.prm", "");
  flow.setup();
  AssertThrow(flow.get_minimum_positive_area() == 1.e-14 &&
                flow.dof_handler().n_dofs() > 0,
              ExcMessage(
                "File setup did not preserve the configured area floor"));
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      deallog.depth_file(10);
      deallog << "Minimum positive area: PASSED" << std::endl;
    }
}
