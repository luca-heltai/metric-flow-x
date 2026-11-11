// SPDX-License-Identifier: MIT
// Conservation under reversal is independent of the HLL implementation.
#include <metric_flow_x/blood_flow_system.h>

#include "tests.h"

using namespace dealii;
using namespace MetricFlowX;

void
test()
{
  BloodFlowSystem<1, 3> flow;
  flow.initialize_params(PRM_DIR "constant.prm");
  flow.setup();
  const double area  = flow.vessel_properties(0).a_d;
  const double c     = flow.compute_wave_speed(area, 0, area);
  const auto   close = [](double actual, double expected) {
    AssertThrow(std::abs(actual - expected) <=
                  2.e-8 *
                    std::max({std::abs(actual), std::abs(expected), 1.e-12}),
                ExcMessage("HLL normal conservation or derivative failed."));
  };
  for (const double speed : {-3., -.2, .2, 3.})
    for (const double n : {-1., 1.})
      {
        const double u = speed * c;
        const auto   uniform =
          flow.hll_flux(n, n, area, u, area, u, 0, 0, area, area, 400., 400.);
        close(uniform[0], n * area * u);
        close(uniform[1],
              n * (.5 * u * u +
                   (flow.compute_pressure_value(area, 0, area) + 400.) /
                     flow.par["rho"]));
        const double al = .8 * area, ar = 1.2 * area;
        const double ul = u + .1 * c, ur = u - .1 * c;
        const auto   forward =
          flow.hll_flux(n, n, al, ul, ar, ur, 0, 0, area, area, 400., 700.);
        const auto reverse =
          flow.hll_flux(-n, -n, ar, ur, al, ul, 0, 0, area, area, 700., 400.);
        for (unsigned int i = 0; i < 2; ++i)
          close(forward[i], -reverse[i]);
        const auto pressure = flow.numerical_external_pressure_flux(
          n, n, al, ul, ar, ur, 0, 0, area, area, 400., 700.);
        const auto pressure_reverse = flow.numerical_external_pressure_flux(
          -n, -n, ar, ur, al, ul, 0, 0, area, area, 700., 400.);
        close(pressure[1], -pressure_reverse[1]);

        // Check each state direction without crossing a sonic branch.
        // Constant nonzero pressure has no state derivative on this half-face.
        for (unsigned int j = 0; j < 4; ++j)
          {
            std::array<double, 4> state     = {al, ul, ar, ur};
            std::array<double, 4> direction = {};
            direction[j]                    = j % 2 == 0 ? area : c;
            constexpr double h              = 1.e-5;
            auto             plus = state, minus = state;
            plus[j] += h * direction[j];
            minus[j] -= h * direction[j];
            const auto fp  = flow.hll_flux(n,
                                          n,
                                          plus[0],
                                          plus[1],
                                          plus[2],
                                          plus[3],
                                          0,
                                          0,
                                          area,
                                          area,
                                          400.,
                                          400.);
            const auto fm  = flow.hll_flux(n,
                                          n,
                                          minus[0],
                                          minus[1],
                                          minus[2],
                                          minus[3],
                                          0,
                                          0,
                                          area,
                                          area,
                                          400.,
                                          400.);
            const auto jac = flow.hll_flux_jac(n,
                                               n,
                                               al,
                                               ul,
                                               ar,
                                               ur,
                                               direction[0],
                                               direction[1],
                                               direction[2],
                                               direction[3],
                                               0,
                                               0,
                                               area,
                                               area);
            for (unsigned int i = 0; i < 2; ++i)
              close(jac[i], (fp[i] - fm[i]) / (2 * h));
          }
      }

  // Discontinuous prescribed pressure exercises the state-dependent HLL weights
  // in the assembled trace rows, including opposite half-face normals.
  std::map<CellId, double> pressures;
  for (const auto &cell : flow.dof_handler().active_cell_iterators())
    if (!cell->is_artificial())
      pressures[cell->id()] = 400. + 70. * cell->global_active_cell_index();
  const BloodFlowSystem<1, 3>::ExternalPressureProvider provider =
    [&](const auto &point) { return pressures.at(point.cell_id); };
  for (const double speed : {-3., -.2, .2, 3.})
    {
      auto state = flow.make_state();
      flow.compute_initial_solution(state, 0.);
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
                if (component < 4)
                  state[indices[i]] =
                    component % 2 == 0 ?
                      (component == 0 ? .8 : 1.2) * area :
                      (speed + (component == 1 ? .1 : -.1)) * c;
              }
          }
      state.compress(VectorOperation::insert);
      flow.jacobian_matrix = 0.;
      flow.add_external_pressure_state_jacobian(0.,
                                                state,
                                                provider,
                                                flow.jacobian_matrix);
      for (unsigned int component = 0; component < 4; ++component)
        {
          auto       direction = flow.make_state();
          const auto selected =
            static_cast<BloodFlowSystem<1, 3>::Component>(component);
          for (const auto index : flow.component_dofs(selected))
            if (flow.locally_owned_dofs().is_element(index))
              direction[index] = component % 2 == 0 ? area : c;
          direction.compress(VectorOperation::insert);
          auto             plus = state, minus = state;
          constexpr double h = 1.e-5;
          plus.add(h, direction);
          minus.add(-h, direction);
          auto fp = flow.make_state(), fm = flow.make_state();
          flow.add_external_pressure_residual(0., plus, provider, fp);
          flow.add_external_pressure_residual(0., minus, provider, fm);
          fp -= fm;
          fp *= 1. / (2 * h);
          auto derivative = flow.make_state();
          flow.jacobian_matrix.vmult(derivative, direction);
          const double scale = std::max(fp.l2_norm(), 1.);
          derivative -= fp;
          AssertThrow(derivative.l2_norm() < 2.e-8 * scale,
                      ExcMessage(
                        "HLL prescribed-pressure state weights failed."));
        }
    }

  deallog.depth_file(10);
  deallog
    << "HLL uniform consistency, normal reversal and state derivatives: PASSED"
    << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  mpi_initlog();
  test();
}
