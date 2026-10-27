// SPDX-License-Identifier: MIT
#include <metric_flow_x/blood_flow_system.h>

namespace MetricFlowX
{
  using namespace dealii;

  /** Weak gradient of the Kelvin--Voigt pressure nu(A)*A_dot. The viscous
   * stress uses a single central value on ordinary interior faces; junctions
   * and RCR closures use the incident vessel's extrapolated stress. Traces
   * remain algebraic: only cell-area derivatives enter this operator.
   */
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_viscous_wall(const VectorType &y,
                                                        const VectorType &ydot,
                                                        VectorType *residual,
                                                        MatrixType *matrix,
                                                        const bool  derivative)
  {
    update_ghosted_vectors(y);
    VectorType rate;
    rate.reinit(locally_owned_dofs_, locally_relevant_dofs_, mpi_communicator_);
    rate           = ydot;
    const auto add = [&](const types::global_dof_index row,
                         const double                  weight,
                         const auto                   &cell,
                         const Point<spacedim>        &point) {
      if (!locally_owned_dofs_.is_element(row))
        return;
      std::vector<types::global_dof_index> indices(fe_->n_dofs_per_cell());
      get_system_dof_indices(cell, indices);
      Point<dim> xi;
      const auto delta = cell->vertex(1) - cell->vertex(0);
      xi[0] =
        std::clamp(((point - cell->vertex(0)) * delta) / delta.norm_square(),
                   0.,
                   1.);
      double area = 0., area_rate = 0.;
      for (unsigned int j = 0; j < indices.size(); ++j)
        if (fe_->system_to_component_index(j).first == 0)
          {
            const double phi = fe_->shape_value(j, xi);
            area += y_relevant(indices[j]) * phi;
            area_rate += rate(indices[j]) * phi;
          }
      AssertThrow(area > 0. && std::isfinite(area),
                  ExcMessage(
                    "Viscoelastic wall requires a finite positive cell area."));
      const auto  &wall = vessel_map.at(cell->material_id()).wall;
      const double nu = wall.viscosity(area, compute_a_d_at_point(cell, point));
      if (residual)
        (*residual)(row) += weight * nu * area_rate;
      if (matrix)
        for (unsigned int j = 0; j < indices.size(); ++j)
          if (fe_->system_to_component_index(j).first == 0)
            matrix->add(row,
                        indices[j],
                        weight * fe_->shape_value(j, xi) *
                          (derivative ? nu : -.5 * nu / area * area_rate));
    };
    const QGauss<dim>                    quadrature(fe_->tensor_degree() + 1);
    FEValues<dim, spacedim>              values(*fe_,
                                   quadrature,
                                   update_values | update_gradients |
                                     update_quadrature_points |
                                     update_JxW_values);
    const FEValuesExtractors::Scalar     velocity(1);
    std::vector<types::global_dof_index> indices(fe_->n_dofs_per_cell());
    for (const auto &cell : dof_handler_.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          get_system_dof_indices(cell, indices);
          values.reinit(cell);
          const auto tangent = compute_directional_vector(cell);
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            for (unsigned int i = 0; i < indices.size(); ++i)
              if (fe_->system_to_component_index(i).first == 1)
                add(indices[i],
                    -values.JxW(q) *
                      (tangent * values[velocity].gradient(i, q)) / density,
                    cell,
                    values.quadrature_point(q));
          for (unsigned int face = 0; face < 2; ++face)
            {
              Point<dim> xi;
              xi[0]              = face;
              const auto point   = cell->vertex(face);
              const bool central = !cell->face(face)->at_boundary() &&
                                   !is_junction_face(cell->id(), face);
              const double normal = face == 0 ? -1. : 1.;
              for (unsigned int i = 0; i < indices.size(); ++i)
                if (fe_->system_to_component_index(i).first == 1)
                  {
                    const double weight =
                      normal * fe_->shape_value(i, xi) / density;
                    add(indices[i],
                        central ? .5 * weight : weight,
                        cell,
                        point);
                    if (central)
                      add(indices[i], .5 * weight, cell->neighbor(face), point);
                  }
              if (cell->face(face)->at_boundary() &&
                  !is_junction_face(cell->id(), face) && outlet_type == "RCR" &&
                  cell->face(face)->boundary_id() != 0 &&
                  rcr_map.count(cell->face(face)->boundary_id()))
                {
                  const auto trace = face_trace_dofs(indices, face);
                  add(trace.first, -1., cell, point);
                }
            }
        }
    for (const auto &junction : junctions)
      for (unsigned int i = 1; i < junction.n_vessels(); ++i)
        {
          const auto &row_half = junction.half_faces[i - 1];
          get_system_dof_indices(row_half.cell, indices);
          const auto trace = face_trace_dofs(indices, row_half.face_no);
          add(trace.second,
              -1. / density,
              junction.half_faces[0].cell,
              junction.location);
          add(trace.second,
              1. / density,
              junction.half_faces[i].cell,
              junction.location);
        }
    if (residual)
      residual->compress(VectorOperation::add);
    if (matrix)
      matrix->compress(VectorOperation::add);
  }
  template void
  BloodFlowSystem<1, 3>::assemble_viscous_wall(const VectorType &,
                                               const VectorType &,
                                               VectorType *,
                                               MatrixType *,
                                               const bool);
} // namespace MetricFlowX
