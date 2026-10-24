#ifndef metric_flow_x_serial_direct_solver_h
#define metric_flow_x_serial_direct_solver_h

#include <deal.II/base/exceptions.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>

#include <cmath>

namespace MetricFlowX
{
  // Serial adapter for the driver's PETSc-without-MUMPS direct path. The
  // original operator is copied unchanged; UMFPACK owns its pivoted factors.
  class SerialDirectSolver
  {
  public:
    template <class Matrix>
    void
    initialize(const Matrix &source)
    {
      AssertThrow(source.m() == source.n() && source.m() > 0,
                  dealii::ExcMessage(
                    "A serial direct solve needs a nonempty square matrix."));
      matrix.clear();
      dealii::DynamicSparsityPattern pattern(source.m(), source.n());
      for (dealii::types::global_dof_index i = 0; i < source.m(); ++i)
        for (auto entry = source.begin(i); entry != source.end(i); ++entry)
          pattern.add(i, entry->column());
      sparsity.copy_from(pattern);
      matrix.reinit(sparsity);
      for (dealii::types::global_dof_index i = 0; i < source.m(); ++i)
        for (auto entry = source.begin(i); entry != source.end(i); ++entry)
          {
            AssertThrow(std::isfinite(entry->value()),
                        dealii::ExcMessage(
                          "Non-finite serial direct operator."));
            matrix.set(i, entry->column(), entry->value());
          }
      factor.initialize(matrix);
      rhs.reinit(source.m());
      solution.reinit(source.m());
    }

    template <class Vector>
    void
    vmult(Vector &destination, const Vector &source)
    {
      AssertThrow(source.size() == rhs.size() &&
                    destination.size() == rhs.size(),
                  dealii::ExcMessage("Serial direct vector size mismatch."));
      for (dealii::types::global_dof_index i = 0; i < rhs.size(); ++i)
        rhs(i) = source(i);
      factor.vmult(solution, rhs);
      for (dealii::types::global_dof_index i = 0; i < rhs.size(); ++i)
        destination(i) = solution(i);
      destination.compress(dealii::VectorOperation::insert);
    }

  private:
    dealii::SparsityPattern      sparsity;
    dealii::SparseMatrix<double> matrix;
    dealii::SparseDirectUMFPACK  factor;
    dealii::Vector<double>       rhs, solution;
  };
} // namespace MetricFlowX
#endif
