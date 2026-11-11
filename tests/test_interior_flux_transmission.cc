// SPDX-License-Identifier: MIT
// Independent stationary-flux, weak-gradient and conservation oracles.
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <metric_flow_x/blood_flow_system.h>

#include "pressure_jacobian_checks.h"
#include "tests.h"

using namespace MetricFlowX;
using Flow = BloodFlowSystem<1, 3>;

void
test()
{
  const auto comm = MPI_COMM_WORLD;
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    {
      std::ofstream parameters("interior_flux.prm");
      parameters << "subsection Metric Flow Parameters\n"
                 << " set Density (rho) = 1060\n"
                 << " set Viscosity coefficient (mu) = 0\n"
                 << " set Profile constant for friction term (xi) = 0\nend\n"
                 << "subsection MetricFlowSystem<1, 3>\n"
                 << " set Finite element degree = 2\n"
                 << " set Number of global refinement = 0\n"
                 << " set Wall model = Square root\n"
                 << " set Outlet boundary condition type = RCR\n"
                 << " set Interior transmission = flux\nend\n";
    }
  MPI_Barrier(comm);
  Flow flow(comm);
  AssertThrow(flow.get_interior_transmission() ==
                Flow::InteriorTransmission::state,
              ExcMessage("State transmission must remain the default."));
  flow.initialize_params("interior_flux.prm", "");
  AssertThrow(flow.get_interior_transmission() ==
                Flow::InteriorTransmission::flux,
              ExcMessage("Interior transmission parameter was not parsed."));
  Triangulation<1, 3>      prepared;
  std::vector<Point<3>>    vertices{{0., 0., 0.}, {1., 0., 0.}, {2., 0., 0.}};
  std::vector<CellData<1>> cells(2);
  for (unsigned int i = 0; i < 2; ++i)
    {
      cells[i].vertices[0] = i;
      cells[i].vertices[1] = i + 1;
      cells[i].material_id = 0;
    }
  prepared.create_triangulation(vertices, cells, SubCellData());
  for (const auto &cell : prepared.active_cell_iterators())
    for (const auto f : cell->face_indices())
      if (cell->face(f)->at_boundary())
        cell->face(f)->set_boundary_id(f + 1);
  GridTools::partition_triangulation_zorder(
    Utilities::MPI::n_mpi_processes(comm), prepared, false);
  flow.triangulation().create_triangulation(
    TriangulationDescription::Utilities::create_description_from_triangulation(
      prepared, comm));

  constexpr double rho = 1060., ad = 1.e-6, young = 4.e5, ratio = .1;
  constexpr double al = ad, ar = 1.04 * ad, ul = .5, q = al * ul;
  const double     ur = q / ar;
  // Analytical square-root law, independent of the native flux/pressure
  // helpers.
  constexpr double stiffness = 4. * young * ratio / 3.;
  constexpr double pl = 2000., reference_pressure = 1000.;
  const double     pr =
    pl - stiffness * (std::sqrt(ar / ad) - 1.) + .5 * rho * (ul * ul - ur * ur);
  const double         c = std::sqrt(stiffness / (2. * rho));
  Flow::NetworkPhysics physics;
  auto                &vessel = physics.vessels[0];
  vessel.a0 = vessel.a_d = ad;
  vessel.r_in = vessel.r_out = vessel.r_d = std::sqrt(ad / numbers::PI);
  vessel.E                                = young;
  vessel.h_wall                           = ratio * vessel.r_d;
  vessel.p_d                              = reference_pressure;
  vessel.L                                = 2.;
  vessel.dynamic_viscosity                = 0.;
  constexpr double resistance             = 1.e8;
  const double     pressure_l             = reference_pressure + pl;
  const double     pressure_r =
    reference_pressure + stiffness * (std::sqrt(ar / ad) - 1.) + pr;
  physics.terminals = {
    {1, {0., resistance, 0., pressure_l + resistance * q, "source"}},
    {2, {0., resistance, 0., pressure_r - resistance * q, "drain"}}};
  flow.setup(physics);

  std::map<CellId, unsigned int>                  side;
  std::map<types::global_dof_index, unsigned int> components;
  for (const auto &cell : flow.dof_handler().active_cell_iterators())
    if (!cell->is_artificial())
      {
        side[cell->id()] = cell->center()[0] < 1. ? 0 : 1;
        if (cell->is_locally_owned())
          {
            std::vector<types::global_dof_index> indices(
              cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(indices);
            for (unsigned int i = 0; i < indices.size(); ++i)
              components[indices[i]] =
                cell->get_fe().system_to_component_index(i).first;
          }
      }
  const auto state = [&](std::array<double, 2> areas,
                         std::array<double, 2> velocities) {
    auto y = flow.make_state();
    for (const auto &cell : flow.dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          std::vector<types::global_dof_index> indices(
            cell->get_fe().n_dofs_per_cell());
          cell->get_dof_indices(indices);
          for (unsigned int i = 0; i < indices.size(); ++i)
            {
              const auto component =
                cell->get_fe().system_to_component_index(i).first;
              y[indices[i]] = component % 2 == 0 ?
                                areas[side.at(cell->id())] :
                                velocities[side.at(cell->id())];
            }
        }
    y.compress(VectorOperation::insert);
    return y;
  };
  const Flow::ExternalPressureProvider step_pressure = [&](const auto &point) {
    return side.at(point.cell_id) == 0 ? pl : pr;
  };
  auto                                   rate = flow.make_state();
  std::array<types::global_dof_index, 4> trace_indices;
  trace_indices.fill(numbers::invalid_dof_index);
  for (const auto &pair : flow.trace_continuity_pairs)
    trace_indices = {{pair.a_canon, pair.u_canon, pair.a_dup, pair.u_dup}};
  for (auto &index : trace_indices)
    index = Utilities::MPI::min(index, comm);
  AssertThrow(trace_indices[0] != numbers::invalid_dof_index,
              ExcMessage("The fixture must have one ordinary interior face."));

  for (const auto flux : {Flow::NumericalFluxType::HLL,
                          Flow::NumericalFluxType::HLL_HDG,
                          Flow::NumericalFluxType::LAX_FRIEDRICHS})
    {
      flow.set_numerical_flux(flux);
      flow.set_external_pressure_provider(step_pressure);
      auto y        = state({{al, ar}}, {{ul, ur}});
      auto residual = flow.make_state();
      flow.assemble_residual(.375, y, rate, residual);
      AssertThrow(residual.linfty_norm() < 2.e-9,
                  ExcMessage(
                    "Nonzero-Q stationary pressure step is not preserved."));

      // The legacy state tie cannot represent these distinct exact side traces.
      flow.set_interior_transmission(Flow::InteriorTransmission::state);
      flow.assemble_residual(.375, y, rate, residual);
      AssertThrow(residual.linfty_norm() > 1.e-4,
                  ExcMessage(
                    "The stationary-step oracle must detect the legacy tie."));
      flow.set_interior_transmission(Flow::InteriorTransmission::flux);

      // Arbitrary interface traces must conserve the actual cell half-fluxes.
      for (const auto index : {trace_indices[0], trace_indices[2]})
        if (flow.locally_owned_dofs().is_element(index))
          y[index] = static_cast<double>(y[index]) *
                     (index == trace_indices[0] ? .95 : 1.08);
      for (const auto index : {trace_indices[1], trace_indices[3]})
        if (flow.locally_owned_dofs().is_element(index))
          y[index] = static_cast<double>(y[index]) +
                     (index == trace_indices[1] ? .13 : -.07);
      y.compress(VectorOperation::insert);
      flow.update_ghosted_vectors(y);
      auto raw = flow.make_state();
      flow.assemble_cell_residuals(.375, flow.y_relevant, raw);
      raw.compress(VectorOperation::add);
      double mass = 0., momentum = 0.;
      for (const auto &[index, component] : components)
        if (component == 0)
          mass += raw[index];
        else if (component == 1)
          momentum += raw[index];
      auto canonical = flow.make_state();
      flow.assemble_trace_interior_equations(.375, flow.y_relevant, canonical);
      canonical.compress(VectorOperation::add);
      for (const auto index : flow.locally_owned_dofs())
        {
          if (index == trace_indices[0])
            mass += canonical[index];
          if (index == trace_indices[1])
            momentum += canonical[index];
        }
      AssertThrow(std::abs(Utilities::MPI::sum(mass, comm)) < 2.e-16 &&
                    std::abs(Utilities::MPI::sum(momentum, comm)) < 2.e-10,
                  ExcMessage(
                    "Cell and canonical interior fluxes are inconsistent."));

      // A uniform pressure offset cancels in all cell/interior rows; only
      // the two fixed external RCR reference pressures retain the offset.
      auto uniform = flow.make_state();
      flow.add_external_pressure_residual(
        .375, y, [](const auto &) { return 73.; }, uniform);
      std::set<types::global_dof_index> boundary_pressure_rows;
      for (const auto &face : flow.boundary_faces)
        boundary_pressure_rows.insert(face.a_hat_dof);
      for (const auto index : flow.locally_owned_dofs())
        AssertThrow(
          std::abs(uniform[index] -
                   (boundary_pressure_rows.count(index) ? -73. : 0.)) < 2.e-10,
          ExcMessage("Uniform prescribed pressure did not cancel."));

      // Linear pressure has the exact weak gradient against quadratic nodal
      // tests: integral(phi_i) = (1/6,2/3,1/6) on each unit cell.
      y = state({{ad, ad}}, {{.5, .5}});
      flow.set_external_pressure_provider(
        [](const auto &point) { return 1700. + 300. * point.point[0]; });
      flow.update_ghosted_vectors(y);
      raw = 0.;
      flow.assemble_cell_residuals(.375, flow.y_relevant, raw);
      raw.compress(VectorOperation::add);
      for (const auto &cell : flow.dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            std::vector<types::global_dof_index> indices(
              cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(indices);
            for (unsigned int i = 0; i < indices.size(); ++i)
              {
                const auto [component, local] =
                  cell->get_fe().system_to_component_index(i);
                if (component < 2)
                  {
                    const double integral = local == 1 ? 2. / 3. : 1. / 6.;
                    const double expected =
                      component == 1 ? -300. / rho * integral : 0.;
                    AssertThrow(
                      std::abs(raw[indices[i]] - expected) < 2.e-10,
                      ExcMessage(
                        "The weak linear pressure gradient is incorrect."));
                  }
              }
          }

      // At a continuous-pressure shared trace the two modes have identical
      // residuals. Their constraint coordinates have an invertible subsonic
      // change of variables (AU,H), not identical Jacobian rows.
      flow.set_external_pressure_provider([](const auto &) { return pl; });
      auto flux_residual  = flow.make_state(),
           state_residual = flow.make_state();
      flow.assemble_residual(.375, y, rate, flux_residual);
      flow.set_interior_transmission(Flow::InteriorTransmission::state);
      flow.assemble_residual(.375, y, rate, state_residual);
      state_residual -= flux_residual;
      AssertThrow(state_residual.linfty_norm() < 2.e-10,
                  ExcMessage("Continuous-pressure transmission roots differ."));
      flow.set_interior_transmission(Flow::InteriorTransmission::flux);
      flow.update_ghosted_vectors(y);
      flow.jacobian_matrix = 0.;
      flow.assemble_jacobian_trace_interior_block(flow.y_relevant);
      flow.assemble_jacobian_trace_continuity_block();
      flow.jacobian_matrix.compress(VectorOperation::add);
      std::array<std::array<double, 4>, 4> block{};
      for (unsigned int i = 0; i < 4; ++i)
        for (unsigned int j = 0; j < 4; ++j)
          {
            double value = 0.;
            if (flow.locally_owned_dofs().is_element(trace_indices[i]))
              value =
                flow.jacobian_matrix.el(trace_indices[i], trace_indices[j]);
            block[i][j] = Utilities::MPI::sum(value, comm);
          }
      double determinant = 1.;
      for (unsigned int k = 0; k < 4; ++k)
        {
          unsigned int pivot = k;
          for (unsigned int i = k + 1; i < 4; ++i)
            if (std::abs(block[i][k]) > std::abs(block[pivot][k]))
              pivot = i;
          if (pivot != k)
            {
              std::swap(block[k], block[pivot]);
              determinant *= -1.;
            }
          const double diagonal = block[k][k];
          determinant *= diagonal;
          AssertThrow(std::abs(diagonal) > 1.e-12,
                      ExcMessage(
                        "Subsonic trace transmission Jacobian is singular."));
          for (unsigned int i = k + 1; i < 4; ++i)
            {
              const double factor = block[i][k] / diagonal;
              for (unsigned int j = k + 1; j < 4; ++j)
                block[i][j] -= factor * block[k][j];
            }
        }
      AssertThrow(std::abs(determinant) > 1.e-6,
                  ExcMessage("The continuous-pressure trace block lost rank."));

      // Each cell/trace column is differentiated independently. Positive and
      // negative supersonic states exercise both outward normals and branches.
      for (const double speed : {-3., -.2, .2, 3.})
        {
          y = state({{.9 * ad, 1.2 * ad}},
                    {{(speed + .13) * c, (speed - .07) * c}});
          for (const auto index : {trace_indices[0], trace_indices[2]})
            if (flow.locally_owned_dofs().is_element(index))
              y[index] = static_cast<double>(y[index]) * (1.04);
          for (const auto index : {trace_indices[1], trace_indices[3]})
            if (flow.locally_owned_dofs().is_element(index))
              y[index] = static_cast<double>(y[index]) + (.11 * c);
          y.compress(VectorOperation::insert);
          flow.set_external_pressure_provider(step_pressure);
          flow.assemble_state_jacobian(.375, y, rate);
          const auto &jacobian = flow.state_jacobian_matrix();
          for (types::global_dof_index column = 0; column < y.size(); ++column)
            {
              auto direction = flow.make_state();
              if (flow.locally_owned_dofs().is_element(column))
                direction[column] = components.at(column) % 2 == 0 ? ad : c;
              direction.compress(VectorOperation::insert);
              auto             plus = y, minus = y;
              constexpr double epsilon = 1.e-6;
              plus.add(epsilon, direction);
              minus.add(-epsilon, direction);
              auto fp = flow.make_state(), fm = flow.make_state(),
                   action = flow.make_state();
              flow.assemble_residual(.375, plus, rate, fp);
              flow.assemble_residual(.375, minus, rate, fm);
              fp -= fm;
              fp *= .5 / epsilon;
              jacobian.vmult(action, direction);
              for (const auto &[row, component] : components)
                {
                  const double scale = component % 2 == 0 ? ad : c;
                  AssertThrow(
                    std::abs(action[row] - fp[row]) <=
                      3.e-6 *
                        std::max(
                          {scale, std::abs(action[row]), std::abs(fp[row])}),
                    ExcMessage([&]() {
                      std::ostringstream message;
                      message
                        << std::setprecision(17)
                        << "Flux transmission state Jacobian disagrees with FD: flux="
                        << static_cast<unsigned int>(flux) << " speed=" << speed
                        << " column=" << column << " row=" << row
                        << " component=" << component
                        << " action=" << action[row] << " FD=" << fp[row]
                        << " scale=" << scale;
                      return message.str();
                    }()));
                }
            }
          check_sparse_pressure_jacobian(flow, y);
          // This is also the native residual probing used by the dense
          // pressure-operator fallback, checked against full residual FD.
          for (unsigned int column = 0; column < 3; ++column)
            {
              const Flow::ExternalPressureProvider basis =
                [&, column](const auto &point) {
                  return column == 2 ?
                           point.point[0] :
                           (side.at(point.cell_id) == column ? 1. : 0.);
                };
              constexpr double epsilon = .01;
              auto             fp = flow.make_state(), fm = flow.make_state(),
                   pressure_action = flow.make_state();
              flow.set_external_pressure_provider([&](const auto &point) {
                return step_pressure(point) + epsilon * basis(point);
              });
              flow.assemble_residual(.375, y, rate, fp);
              flow.set_external_pressure_provider([&](const auto &point) {
                return step_pressure(point) - epsilon * basis(point);
              });
              flow.assemble_residual(.375, y, rate, fm);
              fp -= fm;
              fp *= .5 / epsilon;
              flow.add_external_pressure_residual(.375,
                                                  y,
                                                  basis,
                                                  pressure_action);
              for (const auto &[row, component] : components)
                AssertThrow(
                  std::abs(fp[row] - pressure_action[row]) <=
                    3.e-7 * std::max({1. / rho,
                                      std::abs(fp[row]),
                                      std::abs(pressure_action[row])}),
                  ExcMessage(
                    "Prescribed-pressure residual/visitor fallback differs from FD."));
            }
        }
    }
  deallog.depth_file(10);
  deallog
    << "Interior flux transmission, pressure steps and derivatives: PASSED"
    << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  mpi_initlog();
  test();
}
