// SPDX-License-Identifier: MIT
// A seal preserves the outgoing characteristic while setting outward flow to
// zero. Unequal interior/trace states expose the orientation at both faces.
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <metric_flow_x/blood_flow_system.h>

#include "tests.h"

using namespace MetricFlowX;
using Flow = BloodFlowSystem<1, 3>;

constexpr double area = 1.e-4;

void
mesh(Flow &flow, const bool reverse, const bool inflow = false)
{
  Triangulation<1, 3>      prepared;
  std::vector<Point<3>>    points{{0., 0., 0.}, {.1, 0., 0.}};
  std::vector<CellData<1>> cells(1);
  cells[0].vertices[0] = reverse ? 1 : 0;
  cells[0].vertices[1] = reverse ? 0 : 1;
  prepared.create_triangulation(points, cells, SubCellData());
  for (const auto &cell : prepared.active_cell_iterators())
    for (const auto face : cell->face_indices())
      cell->face(face)->set_boundary_id(inflow ? face :
                                        cell->face(face)->center()[0] < .05 ?
                                                 1 :
                                                 2);
  GridTools::partition_triangulation_zorder(
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), prepared, false);
  flow.triangulation().create_triangulation(
    TriangulationDescription::Utilities::create_description_from_triangulation(
      prepared, MPI_COMM_WORLD));
}

Flow::NetworkPhysics
physics(const bool capacitor = false)
{
  Flow::NetworkPhysics input;
  auto                &v = input.vessels[0];
  v.a0 = v.a_d = area;
  v.r_d        = std::sqrt(area / numbers::PI);
  v.E          = 5.e5;
  v.h_wall     = .0005;
  v.L          = .1;
  v.p_d        = 1000.;
  input.sealed_terminals.insert(1);
  input.terminals.emplace(
    2,
    Flow::RCRPhysics{1.e7, 9.e7, capacitor ? 5.e-9 : 0., 1200., "reservoir"});
  return input;
}

void
check(const bool reverse, const bool capacitor)
{
  ParameterAcceptor::clear();
  Flow flow(MPI_COMM_WORLD);
  flow.initialize_params("sealed_terminals.prm", "");
  mesh(flow, reverse);
  const auto input = physics(capacitor);
  const auto received =
    Utilities::unpack<Flow::NetworkPhysics>(Utilities::pack(input));
  AssertThrow(received.sealed_terminals == input.sealed_terminals,
              ExcMessage("Serialization lost the sealed endpoints"));
  flow.setup(received);
  AssertThrow(flow.rcr_pc_dof.size() == (capacitor ? 1u : 0u) &&
                !flow.rcr_map.count(1) && !flow.terminal_Pc_storage.count(1),
              ExcMessage("A seal introduced an RCR state or reservoir"));
  auto y = flow.make_state(), rate = flow.make_state();
  for (const auto component :
       {Flow::Component::area, Flow::Component::area_trace})
    for (const auto i : flow.component_dofs(component))
      if (flow.locally_owned_dofs().is_element(i))
        y[i] = component == Flow::Component::area ? area : 1.1 * area;
  for (const auto component :
       {Flow::Component::velocity, Flow::Component::velocity_trace})
    for (const auto i : flow.component_dofs(component))
      if (flow.locally_owned_dofs().is_element(i))
        y[i] = component == Flow::Component::velocity ? .02 : .03;
  for (const auto i : flow.component_dofs(Flow::Component::capacitor_pressure))
    if (flow.locally_owned_dofs().is_element(i))
      y[i] = 1100.;
  y.compress(VectorOperation::insert);
  flow.update_ghosted_vectors(y);
  auto residual = flow.make_state();
  flow.assemble_trace_boundary_equations(.17, flow.y_relevant, residual);
  residual.compress(VectorOperation::add);
  const auto  &v         = input.vessels.at(0);
  const double beta      = 4. * std::sqrt(numbers::PI) / 3. * v.E * v.h_wall;
  const auto   invariant = [&](const double a) {
    return 4. * std::sqrt(beta / (2. * 1060. * area)) *
           (std::pow(a, .25) - std::pow(area, .25));
  };
  for (const auto &face : flow.boundary_faces)
    if (face.cell->face(face.face_no)->boundary_id() == 1)
      {
        AssertThrow(face.face_no == (reverse ? 1u : 0u), ExcInternalError());
        const double sign = reverse ? 1. : -1.;
        AssertThrow(
          std::abs(residual[face.a_hat_dof] - sign * 1.1 * area * .03) <
              1.e-15 &&
            std::abs(residual[face.u_hat_dof] -
                     (.03 - .02 + sign * invariant(1.1 * area))) < 1.e-11,
          ExcMessage(
            "The sealed boundary used the wrong outgoing characteristic"));
      }

  auto direction = flow.make_state();
  for (const auto component :
       {Flow::Component::area, Flow::Component::area_trace})
    for (const auto i : flow.component_dofs(component))
      if (flow.locally_owned_dofs().is_element(i))
        direction[i] = 1.e-6 * (1. + .1 * std::sin(i + 1.));
  for (const auto component :
       {Flow::Component::velocity, Flow::Component::velocity_trace})
    for (const auto i : flow.component_dofs(component))
      if (flow.locally_owned_dofs().is_element(i))
        direction[i] = .01 * std::cos(i + 1.);
  for (const auto i : flow.component_dofs(Flow::Component::capacitor_pressure))
    if (flow.locally_owned_dofs().is_element(i))
      direction[i] = 3.;
  direction.compress(VectorOperation::insert);
  const Flow::ExternalPressureProvider pressure = [](const auto &point) {
    return 37. + 13. * point.point[0];
  };
  flow.set_external_pressure_provider(pressure);
  flow.assemble_state_jacobian(.17, y, rate);
  auto action = flow.make_state();
  flow.state_jacobian_matrix().vmult(action, direction);
  auto         plus = y, minus = y;
  const double epsilon = 1.e-5;
  plus.add(epsilon, direction);
  minus.add(-epsilon, direction);
  auto fp = flow.make_state(), fm = flow.make_state();
  flow.assemble_residual(.17, plus, rate, fp);
  flow.assemble_residual(.17, minus, rate, fm);
  fp -= fm;
  fp *= .5 / epsilon;
  fp -= action;
  for (const auto component : {Flow::Component::area_trace,
                               Flow::Component::velocity_trace,
                               Flow::Component::capacitor_pressure})
    {
      double error = 0., scale = 0.;
      for (const auto i : flow.component_dofs(component))
        if (flow.locally_owned_dofs().is_element(i))
          {
            error += fp[i] * fp[i];
            scale += action[i] * action[i];
          }
      error = std::sqrt(Utilities::MPI::sum(error, MPI_COMM_WORLD));
      scale = std::sqrt(Utilities::MPI::sum(scale, MPI_COMM_WORLD));
      AssertThrow(error < 1.e-6 * scale + 1.e-10,
                  ExcMessage(
                    "Sealed/RCR Jacobian disagrees with finite differences"));
    }
  residual = 0.;
  flow.add_external_pressure_residual(.17, y, pressure, residual);
  for (const auto &face : flow.boundary_faces)
    if (face.cell->face(face.face_no)->boundary_id() == 1)
      AssertThrow(residual[face.a_hat_dof] == 0. &&
                    residual[face.u_hat_dof] == 0.,
                  ExcMessage("External pressure created flow through a seal"));
  flow.clear_external_pressure_provider();
  flow.initialize_consistent_state(y, rate, .17);
  AssertThrow(rate.l2_norm() > 0.,
              ExcMessage("Nonzero reservoir forcing did not reach the state"));
  for (const auto &face : flow.boundary_faces)
    if (face.cell->face(face.face_no)->boundary_id() == 1)
      AssertThrow(std::abs(y[face.a_hat_dof] * y[face.u_hat_dof]) < 1.e-12,
                  ExcMessage("The consistent startup leaks through a seal"));
}

void
check_inflow_and_invalid_inputs()
{
  for (unsigned int failure = 0; failure < 7; ++failure)
    {
      ParameterAcceptor::clear();
      Flow flow(MPI_COMM_WORLD);
      flow.initialize_params("sealed_terminals.prm", "");
      mesh(flow, false, failure == 0);
      auto input = physics();
      if (failure == 0)
        input.terminals.clear();
      else if (failure == 1)
        input.terminals.emplace(1,
                                Flow::RCRPhysics{0., 1.e8, 0., 0., "overlap"});
      else if (failure == 2)
        input.sealed_terminals.insert(0);
      else if (failure == 3)
        input.sealed_terminals.insert(255);
      else if (failure == 4)
        input.sealed_terminals.insert(3);
      else if (failure == 5)
        input.sealed_terminals.clear();
      else if (failure == 6)
        input.terminal_connections.push_back({1, 2, 1.e8});
      bool caught = false;
      try
        {
          flow.setup(input);
        }
      catch (const ExceptionBase &)
        {
          caught = true;
        }
      const unsigned int size = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
      AssertThrow(
        Utilities::MPI::sum(caught ? 1u : 0u, MPI_COMM_WORLD) ==
          (failure == 0 ? 0u : size),
        ExcMessage(
          "Invalid sealed terminal input was not rejected collectively"));
      if (failure == 0)
        {
          auto y = flow.make_state(), rate = flow.make_state();
          flow.initialize_consistent_state(y, rate, .17);
          AssertThrow(
            rate.l2_norm() > 0.,
            ExcMessage(
              "Time-dependent nonzero inflow must fill the sealed tube"));
          for (const auto &face : flow.boundary_faces)
            if (face.cell->face(face.face_no)->boundary_id() == 1)
              AssertThrow(
                std::abs(y[face.a_hat_dof] * y[face.u_hat_dof]) < 1.e-12,
                ExcMessage("Inflow-driven startup leaks through its seal"));
        }
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  mpi_initlog();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::ofstream prm("sealed_terminals.prm");
      prm << "subsection Functions\n set Inflow function = 2e-9*(1+t)\nend\n"
          << "subsection Metric Flow Parameters\n"
          << " set Density (rho) = 1060\n set Viscosity coefficient (mu) = 0\n"
          << " set Profile constant for friction term (xi) = 0\nend\n"
          << "subsection MetricFlowSystem<1, 3>\n"
          << " set Numerical flux type = HLL\n"
          << " set Wall model = Square root\n"
          << " set Outlet boundary condition type = RCR\n"
          << " set Finite element degree = 1\n"
          << " set Verbosity (console depth) = 0\n"
          << " set Initial consistency tolerance = 1e-10\nend\n";
    }
  MPI_Barrier(MPI_COMM_WORLD);
  for (const bool reverse : {false, true})
    for (const bool capacitor : {false, true})
      check(reverse, capacitor);
  check_inflow_and_invalid_inputs();
  deallog.depth_file(10);
  deallog << "Explicit sealed endpoints at both tube orientations: PASSED"
          << std::endl;
}
