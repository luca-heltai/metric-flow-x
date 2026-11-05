// SPDX-License-Identifier: MIT
// A uniform inviscid tube between two pressure reservoirs has an exact steady
// flow. Reverse its cells to exercise both RCR orientations, with and without
// capacitors, then check nonuniform characteristics and their Jacobians.
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <metric_flow_x/blood_flow_system.h>

#include "tests.h"

using namespace MetricFlowX;
using Flow = BloodFlowSystem<1, 3>;

constexpr double reference_area = 1.e-4;
constexpr double pressure       = 1000.;
constexpr double speed          = .02;
constexpr double resistance     = 1.e8;

void
make_mesh(Flow &flow, const bool reverse)
{
  Triangulation<1, 3>      prepared;
  std::vector<Point<3>>    points{{0., 0., 0.}, {.1, 0., 0.}};
  std::vector<CellData<1>> cells(1);
  cells[0].vertices[0] = reverse ? 1 : 0;
  cells[0].vertices[1] = reverse ? 0 : 1;
  prepared.create_triangulation(points, cells, SubCellData());
  for (const auto &cell : prepared.active_cell_iterators())
    for (const auto face : cell->face_indices())
      cell->face(face)->set_boundary_id(
        cell->face(face)->center()[0] < .05 ? 1 : 2);
  prepared.refine_global(1);
  GridTools::partition_triangulation_zorder(
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), prepared, false);
  flow.triangulation().create_triangulation(
    TriangulationDescription::Utilities::create_description_from_triangulation(
      prepared, MPI_COMM_WORLD));
}

void
test()
{
  for (const bool reverse : {false, true})
    for (const bool capacitor : {false, true})
      {
        ParameterAcceptor::clear();
        Flow flow(MPI_COMM_WORLD);
        flow.initialize_params("rcr_pressure_faces.prm", "");
        make_mesh(flow, reverse);
        Flow::NetworkPhysics physics;
        auto                &v = physics.vessels[0];
        v.a0 = v.a_d      = reference_area;
        v.r_d             = std::sqrt(reference_area / numbers::PI);
        v.E               = 5.e5;
        v.h_wall          = .0005;
        v.L               = .1;
        v.p_d             = pressure;
        const double r1   = capacitor ? .2 * resistance : 0.;
        const double r2   = resistance - r1;
        const double c    = capacitor ? 5.e-9 : 0.;
        physics.terminals = {{1, {r1, r2, c, 1200., "pressure-source"}},
                             {2, {r1, r2, c, 800., "pressure-drain"}}};
        flow.setup(physics);
        for (const auto &cell : flow.dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            for (const auto face : cell->face_indices())
              if (cell->face(face)->at_boundary() &&
                  cell->face(face)->boundary_id() == 1)
                AssertThrow(
                  face == (reverse ? 1u : 0u),
                  ExcMessage(
                    "The reversed test mesh must exchange its endpoint orientation"));
        auto y = flow.make_state(), rate = flow.make_state();
        y                      = 0.;
        rate                   = 0.;
        const double direction = reverse ? -1. : 1.;
        for (const auto component :
             {Flow::Component::area, Flow::Component::area_trace})
          for (const auto i : flow.component_dofs(component))
            if (flow.locally_owned_dofs().is_element(i))
              y[i] = reference_area;
        for (const auto component :
             {Flow::Component::velocity, Flow::Component::velocity_trace})
          for (const auto i : flow.component_dofs(component))
            if (flow.locally_owned_dofs().is_element(i))
              y[i] = direction * speed;
        for (const auto &[bid, index] : flow.rcr_pc_dof)
          if (flow.locally_owned_dofs().is_element(index))
            y[index] = physics.terminals.at(bid).P_out +
                       r2 * (bid == 1 ? -1. : 1.) * reference_area * speed;
        y.compress(VectorOperation::insert);
        auto residual = flow.make_state();
        flow.assemble_residual(7.2, y, rate, residual);
        AssertThrow(
          residual.linfty_norm() < 1.e-9,
          ExcMessage(
            "Exact pressure-driven steady tube violates the RCR balance: " +
            Utilities::to_string(residual.linfty_norm())));

        // Deliberately unequal cell and trace states expose the outgoing
        // invariant; a uniform state alone would hide the wrong characteristic
        // at face zero.
        for (const auto i : flow.component_dofs(Flow::Component::area_trace))
          if (flow.locally_owned_dofs().is_element(i))
            y[i] = 1.1 * reference_area;
        for (const auto i :
             flow.component_dofs(Flow::Component::velocity_trace))
          if (flow.locally_owned_dofs().is_element(i))
            y[i] = direction * .03;
        y.compress(VectorOperation::insert);
        flow.update_ghosted_vectors(y);
        residual = 0.;
        flow.assemble_trace_boundary_equations(7.2, flow.y_relevant, residual);
        residual.compress(VectorOperation::add);
        const double beta = 4. * std::sqrt(numbers::PI) / 3. * v.E * v.h_wall;
        const auto   invariant = [&](const double area) {
          return 4. * std::sqrt(beta / (2. * 1060. * reference_area)) *
                 (std::pow(area, .25) - std::pow(reference_area, .25));
        };
        for (const auto &face : flow.boundary_faces)
          {
            const auto   bid     = face.cell->face(face.face_no)->boundary_id();
            const double sign    = bid == 1 ? -1. : 1.;
            const double area    = 1.1 * reference_area;
            const double outward = sign * area * .03;
            const double pc      = capacitor ? physics.terminals.at(bid).P_out +
                                            r2 * sign * reference_area * speed :
                                               physics.terminals.at(bid).P_out;
            const double expected_pressure =
              pressure + beta / reference_area *
                           (std::sqrt(area) - std::sqrt(reference_area));
            const double expected_area =
              expected_pressure - (capacitor ? r1 : resistance) * outward - pc;
            const double expected_velocity =
              direction * (.03 - speed + sign * invariant(area));
            AssertThrow(
              std::abs(residual[face.a_hat_dof] - expected_area) < 1.e-9,
              ExcMessage("RCR pressure must use physical outward flow"));
            AssertThrow(
              std::abs(residual[face.u_hat_dof] - expected_velocity) < 1.e-11,
              ExcMessage(
                "RCR must preserve the physically outgoing characteristic"));
          }

        auto perturbation = flow.make_state();
        for (const auto component :
             {Flow::Component::area, Flow::Component::area_trace})
          for (const auto i : flow.component_dofs(component))
            if (flow.locally_owned_dofs().is_element(i))
              perturbation[i] = 1.e-6 * (1. + .1 * std::sin(i + 1.));
        for (const auto component :
             {Flow::Component::velocity, Flow::Component::velocity_trace})
          for (const auto i : flow.component_dofs(component))
            if (flow.locally_owned_dofs().is_element(i))
              perturbation[i] = .01 * std::cos(i + 1.);
        for (const auto i :
             flow.component_dofs(Flow::Component::capacitor_pressure))
          if (flow.locally_owned_dofs().is_element(i))
            perturbation[i] = 3. * std::cos(i + 1.);
        perturbation.compress(VectorOperation::insert);
        flow.assemble_state_jacobian(7.2, y, rate);
        auto product = flow.make_state();
        flow.state_jacobian_matrix().vmult(product, perturbation);
        auto         plus = y, minus = y;
        const double epsilon = 1.e-5;
        plus.add(epsilon, perturbation);
        minus.add(-epsilon, perturbation);
        auto fp = flow.make_state(), fm = flow.make_state();
        flow.assemble_residual(7.2, plus, rate, fp);
        flow.assemble_residual(7.2, minus, rate, fm);
        fp -= fm;
        fp *= .5 / epsilon;
        fp -= product;
        for (const auto component : {Flow::Component::area_trace,
                                     Flow::Component::velocity_trace,
                                     Flow::Component::capacitor_pressure})
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
            AssertThrow(
              error < 1.e-6 * scale + 1.e-11,
              ExcMessage(
                "RCR state Jacobian disagrees with finite differences"));
          }

        flow.initialize_consistent_state(y, rate, 7.2);
        flow.assemble_residual(7.2, y, rate, residual);
        AssertThrow(
          std::isfinite(residual.l2_norm()) && rate.l2_norm() > 0.,
          ExcMessage(
            "Nonzero reservoir pressures must produce a consistent nontrivial startup"));
      }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::ofstream prm("rcr_pressure_faces.prm");
      prm << "subsection Metric Flow Parameters\n"
          << " set Density (rho) = 1060\n set Viscosity coefficient (mu) = 0\n"
          << " set Profile constant for friction term (xi) = 0\nend\n"
          << "subsection MetricFlowSystem<1, 3>\n"
          << " set Numerical flux type = HLL\n"
          << " set Wall model = Square root\n"
          << " set Outlet boundary condition type = RCR\n"
          << " set Finite element degree = 1\n"
          << " set Verbosity (console depth) = 0\n"
          << " set Initial consistency tolerance = 1e-5\nend\n";
    }
  MPI_Barrier(MPI_COMM_WORLD);
  test();
  deallog.depth_file(10);
  deallog << "RCR pressure sources at both tube orientations: PASSED"
          << std::endl;
}
