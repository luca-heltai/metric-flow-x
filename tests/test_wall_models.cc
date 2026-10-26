// SPDX-License-Identifier: MIT
// Constitutive/invariant derivatives, complete implicit Jacobians at nonuniform
// states and nonzero area rates, mixed junctions, pressure output and bad
// input.
#include <metric_flow_x/blood_flow_system.h>

#include "tests.h"
using namespace MetricFlowX;

void
check_law(WallLaw wall)
{
  const double ref = 4.5239e-4, thickness = .0012, rho = 1060.;
  wall.km = 2.;
  const double expected_nu =
    wall.type == WallLaw::Type::artery ? 1. / ref : 2. / (ref * std::sqrt(ref));
  AssertThrow(std::abs(wall.viscosity(ref, ref) / expected_nu - 1.) < 1.e-14,
              ExcInternalError());
  const double expected_pressure =
    wall.reference_pressure +
    (wall.type == WallLaw::Type::artery ?
       thickness * std::sqrt(numbers::PI * ref) / ref * wall.ec *
         wall.epsilonr * std::log1p(std::exp(-wall.epsilon0 / wall.epsilonr)) :
       0.);
  AssertThrow(std::abs(wall.elastic(ref, ref, thickness).pressure -
                       expected_pressure) < 1.e-10,
              ExcInternalError());
  for (const double ratio : {.3, .8, 1., 1.2})
    {
      const double a = ref * ratio, h = a * 1.e-5;
      const auto   v = wall.elastic(a, ref, thickness);
      const auto   p = wall.elastic(a + h, ref, thickness),
                 m   = wall.elastic(a - h, ref, thickness);
      AssertThrow(std::abs((p.pressure - m.pressure) / (2 * h) - v.first) <
                    1.e-7 * std::abs(v.first),
                  ExcInternalError());
      AssertThrow(std::abs((p.first - m.first) / (2 * h) - v.second) <
                    1.e-7 * std::abs(v.second),
                  ExcInternalError());
      const double derivative = (wall.invariant(a + h, ref, thickness, rho) -
                                 wall.invariant(a - h, ref, thickness, rho)) /
                                (2 * h);
      AssertThrow(std::abs(derivative -
                           wall.wave_speed(a, ref, thickness, rho) / a) <
                    1.e-7 * std::abs(derivative),
                  ExcInternalError());
    }
}
void
test()
{
  using NumericalFluxType = BloodFlowSystem<1, 3>::NumericalFluxType;
  const auto check_system = [](const std::string &prm) {
    ParameterAcceptor::clear();
    BloodFlowSystem<1, 3> p;
    p.initialize_params(prm);
    p.create_triangulation();
    p.setup_system();
    p.initialize_terminal_capacitors();
    p.build_per_cell_mass_inv();
    p.compute_initial_solution(p.solution, 0.);
    p.initialize_trace_unknowns(p.solution, 0.);
    for (const auto &cell : p.dof_handler_.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          std::vector<types::global_dof_index> indices(
            p.fe_->n_dofs_per_cell());
          cell->get_dof_indices(indices);
          for (unsigned int k = 0; k < indices.size(); ++k)
            {
              const auto i = indices[k];
              if (p.fe_->system_to_component_index(k).first % 2 == 0)
                p.solution(i) =
                  p.solution(i) * (1. + .03 * std::cos(double(i)));
              else
                p.solution(i) = .03 + .01 * std::sin(double(i));
            }
        }
    p.solution.compress(VectorOperation::insert);
    VectorType rate(p.locally_owned_dofs_, p.mpi_communicator_);
    for (const auto i : p.locally_owned_dofs_)
      rate(i) = .00002 * std::cos(double(i));
    rate.compress(VectorOperation::insert);
    VectorType pressure(rate);
    p.compute_pressure(p.solution, rate, pressure, 0.);
    for (const auto &cell : p.dof_handler_.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          std::vector<types::global_dof_index> indices(
            p.fe_->n_dofs_per_cell());
          cell->get_dof_indices(indices);
          for (unsigned int k = 0; k < indices.size(); ++k)
            if (p.fe_->system_to_component_index(k).first == 0)
              {
                const auto   i     = indices[k];
                const auto   point = p.support_point(cell, k);
                const double ref   = p.compute_a_d_at_point(cell, point);
                const auto  &vp    = p.vessel_map.at(cell->material_id());
                const double expected =
                  vp.wall.elastic(p.solution(i), ref, vp.h_wall).pressure +
                  vp.wall.viscosity(p.solution(i), ref) * rate(i);
                AssertThrow(std::abs(pressure(i) - expected) < 1.e-8,
                            ExcInternalError());
              }
        }
    for (const auto flux : {NumericalFluxType::HLL,
                            NumericalFluxType::HLL_HDG,
                            NumericalFluxType::LAX_FRIEDRICHS})
      {
        p.set_numerical_flux(flux);
        for (const bool derivative : {false, true})
          {
            if (derivative)
              p.assemble_derivative_jacobian(0., p.solution, rate);
            else
              p.assemble_state_jacobian(0., p.solution, rate);
            const auto &matrix = derivative ? p.derivative_jacobian_matrix() :
                                              p.state_jacobian_matrix();
            VectorType  plus(rate), minus(rate), fp(rate), fm(rate), unit(rate),
              column(rate);
            double worst = 0.;
            for (types::global_dof_index j = 0; j < p.n_total_dofs; ++j)
              {
                plus                   = derivative ? rate : p.solution;
                minus                  = plus;
                const double magnitude = p.locally_owned_dofs_.is_element(j) ?
                                           std::abs(double(plus(j))) :
                                           0.;
                const double h =
                  derivative ?
                    1.e-9 :
                    1.e-6 * std::max(1.e-4,
                                     Utilities::MPI::max(magnitude,
                                                         p.mpi_communicator_));
                if (p.locally_owned_dofs_.is_element(j))
                  {
                    plus(j)  = plus(j) + h;
                    minus(j) = minus(j) - h;
                  }
                plus.compress(VectorOperation::insert);
                minus.compress(VectorOperation::insert);
                p.assemble_residual(0.,
                                    derivative ? p.solution : plus,
                                    derivative ? plus : rate,
                                    fp);
                p.assemble_residual(0.,
                                    derivative ? p.solution : minus,
                                    derivative ? minus : rate,
                                    fm);
                unit = 0.;
                if (p.locally_owned_dofs_.is_element(j))
                  unit(j) = 1.;
                unit.compress(VectorOperation::insert);
                matrix.vmult(column, unit);
                for (const auto i : p.locally_owned_dofs_)
                  {
                    const double fd = (fp(i) - fm(i)) / (2 * h);
                    const double error =
                      std::abs(fd - column(i)) /
                      (1. + std::abs(fd) + std::abs(column(i)));
                    worst = std::max(worst, error);
                  }
              }
            worst = Utilities::MPI::max(worst, p.mpi_communicator_);
            std::cout << prm << " flux=" << int(flux) << " rate=" << derivative
                      << " error=" << worst << std::endl;
            AssertThrow(
              worst < 2.e-5,
              ExcMessage(
                "Wall-model Jacobian finite-difference check failed."));
          }
      }
  };
  WallLaw artery;
  artery.type     = WallLaw::Type::artery;
  artery.ee       = 260000.;
  artery.ec       = 20000000.;
  artery.epsilon0 = .25;
  artery.epsilonr = .05;
  check_law(artery);
  WallLaw vein;
  vein.type      = WallLaw::Type::vein;
  vein.stiffness = 1000.;
  vein.m         = 10.;
  vein.n         = -1.5;
  check_law(vein);
  vein.type = WallLaw::Type::dural_sinus;
  vein.m    = .5;
  vein.n    = 0.;
  check_law(vein);
  check_system(PRM_DIR "wall_arterial.prm");
  check_system(PRM_DIR "wall_venous.prm");
  check_system(PRM_DIR "wall_mixed.prm");
  // These errors must be detected before distributed setup, on every rank.
  const auto rejected = [](const std::string &prm,
                           const std::string &model,
                           const std::string &message,
                           const std::string &field = "",
                           const std::string &value = "") {
    ParameterAcceptor::clear();
    BloodFlowSystem<1, 3> p;
    p.initialize_params(prm);
    p.wall_model = model;
    if (!field.empty())
      {
        if (Utilities::MPI::this_mpi_process(p.mpi_communicator_) == 0)
          {
            std::ifstream in(p.vtk_file_path);
            std::string   text((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
            const auto    start = text.find("SCALARS " + field + " ");
            const auto    data  = text.find("LOOKUP_TABLE default\n", start) +
                              std::string("LOOKUP_TABLE default\n").size();
            const auto end = text.find('\n', data);
            if (value.empty())
              text.erase(start, end - start + 1);
            else
              text.replace(data, end - data, value + " " + value);
            std::ofstream out("invalid_wall.vtk");
            out << text;
          }
        MPI_Barrier(p.mpi_communicator_);
        p.vtk_file_path = "invalid_wall.vtk";
      }
    try
      {
        p.create_triangulation();
      }
    catch (const std::exception &e)
      {
        std::istringstream words(e.what());
        std::string        normalized, word;
        while (words >> word)
          normalized += (normalized.empty() ? "" : " ") + word;
        AssertThrow(normalized.find(message) != std::string::npos,
                    ExcMessage(e.what()));
        MPI_Barrier(p.mpi_communicator_);
        return;
      }
    AssertThrow(false, ExcMessage("Incompatible wall input was accepted."));
  };
  rejected(PRM_DIR "constant.prm", "ADAVN arterial", "adavn_p_ref");
  rejected(PRM_DIR "wall_arterial.prm", "ADAVN mixed", "adavn_wall_type");
  rejected(PRM_DIR "wall_mixed.prm", "ADAVN arterial", "venous cell");
  rejected(PRM_DIR "wall_mixed.prm", "ADAVN venous", "arterial cell");
  rejected(PRM_DIR "wall_venous.prm", "ADAVN venous", "adavn_K_m", "adavn_K_m");
  rejected(
    PRM_DIR "wall_venous.prm", "ADAVN venous", "adavn_K_m", "adavn_K_m", "-1");
  rejected(PRM_DIR "wall_venous.prm",
           "ADAVN venous",
           "venous K and m",
           "adavn_venous_m",
           "-1");
  rejected(PRM_DIR "wall_venous.prm",
           "ADAVN venous",
           "a_d must be finite and positive",
           "a_d",
           "0");
  rejected(PRM_DIR "wall_arterial.prm",
           "ADAVN arterial",
           "arterial moduli",
           "adavn_epsilon_r",
           "0");
  deallog.depth_file(10);
  deallog << "Wall models: PASSED" << std::endl;
}
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test();
}
