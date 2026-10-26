// The global saddle system is invertible while its rank-local diagonal blocks
// can be singular. Check the original equations against a dense reference.
#include <deal.II/base/mpi.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>

#include <metric_flow_x/blood_flow_system.h>

#include <cfenv>

#include "tests.h"
using namespace dealii;
using namespace MetricFlowX;
void
test()
{
  const MPI_Comm        comm = MPI_COMM_WORLD;
  BloodFlowSystem<1, 3> flow(comm);
  // Two unknowns on four ranks force empty local factors. Check exception
  // flags explicitly so Release and emulated builds detect the same error.
  for (const unsigned int n : {2u, 4u})
    {
      const auto owned =
        Utilities::MPI::create_evenly_distributed_partitioning(comm, n);
      DynamicSparsityPattern pattern(n);
      for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
          pattern.add(i, j);
      MatrixType matrix;
      matrix.reinit(owned, pattern, comm);
      MetricFlowX::VectorType rhs(owned, comm), x(owned, comm),
        residual(owned, comm);
      const double           entries[4][4] = {{1., 0., 1., 0.},
                                              {0., 0., 1., 0.},
                                              {1., 1., 0., 0.},
                                              {0., 0., 0., 1.}};
      FullMatrix<double>     dense(n, n);
      dealii::Vector<double> dense_rhs(n), reference(n);
      for (unsigned int i = 0; i < n; ++i)
        {
          dense_rhs[i] = i + 1.;
          for (unsigned int j = 0; j < n; ++j)
            dense(i, j) = n == 2 ? (i + j == 1 ? 1. : 0.) : entries[i][j];
        }
      dense.gauss_jordan();
      dense.vmult(reference, dense_rhs);
      for (const auto i : owned)
        {
          rhs(i) = dense_rhs[i];
          for (unsigned int j = 0; j < n; ++j)
            matrix.set(i, j, n == 2 ? (i + j == 1 ? 1. : 0.) : entries[i][j]);
        }
      rhs.compress(VectorOperation::insert);
      matrix.compress(VectorOperation::insert);
      BloodFlowSystem<1, 3>::IterativePreconditioner preconditioner;
      std::feclearexcept(FE_ALL_EXCEPT);
      flow.initialize_iterative_preconditioner(preconditioner, matrix);
      x = 0.;
      SolverControl                   control(1000, 1e-10 * rhs.l2_norm());
      LA::SolverGMRES::AdditionalData data;
      data.right_preconditioning = true;
      LA::SolverGMRES solver(control, data);
      solver.solve(matrix, x, rhs, preconditioner);
      AssertThrow(
        Utilities::MPI::max(std::fetestexcept(FE_INVALID | FE_DIVBYZERO),
                            comm) == 0,
        ExcMessage(
          "Preconditioner factorization raised a floating-point exception."));
      matrix.vmult(residual, x);
      residual -= rhs;
      unsigned int invalid = 0;
      for (const auto i : owned)
        if (std::abs(x(i) - reference[i]) > 1e-8)
          invalid = 1;
      AssertThrow(
        Utilities::MPI::max(invalid, comm) == 0 && residual.l2_norm() < 1e-8,
        ExcMessage(
          "The distributed saddle solve must satisfy the original system."));
    }
  deallog.depth_file(10);
  deallog
    << "Distributed saddle solve matches dense reference and original residual: PASSED"
    << std::endl;
}
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  mpi_initlog();
  test();
}
