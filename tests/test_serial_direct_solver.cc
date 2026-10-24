// A nonsingular saddle system has structural zero diagonals. Solve several
// right-hand sides, refactor changed values, and check the untouched operator.
#include <deal.II/base/mpi.h>

#include <deal.II/lac/full_matrix.h>

#include <metric_flow_x/blood_flow_system.h>
#include <metric_flow_x/serial_direct_solver.h>

#include <cmath>
#include <limits>

#include "tests.h"
using namespace dealii;
using namespace MetricFlowX;
void
test()
{
  IndexSet owned(4);
  owned.add_range(0, 4);
  DynamicSparsityPattern pattern(4);
  for (unsigned int i = 0; i < 4; ++i)
    for (unsigned int j = 0; j < 4; ++j)
      pattern.add(i, j);
  MatrixType source;
  source.reinit(owned, pattern, MPI_COMM_SELF);
  const double       entries[4][4] = {{1., 0., 1., 0.},
                                      {0., 0., 1., 0.},
                                      {1., 1., 0., 0.},
                                      {0., 0., 0., 1e-8}};
  FullMatrix<double> original(4, 4);
  for (unsigned int i = 0; i < 4; ++i)
    for (unsigned int j = 0; j < 4; ++j)
      {
        original(i, j) = entries[i][j];
        source.set(i, j, entries[i][j]);
      }
  source.compress(VectorOperation::insert);
#ifdef DEAL_II_WITH_UMFPACK
  SerialDirectSolver direct;
  VectorType         rhs(owned, MPI_COMM_SELF), x(owned, MPI_COMM_SELF),
    defect(owned, MPI_COMM_SELF);
  for (unsigned int update = 0; update < 2; ++update)
    {
      if (update)
        {
          original(0, 0) = 2.;
          source.set(0, 0, 2.);
          source.compress(VectorOperation::insert);
        }
      direct.initialize(source);
      FullMatrix<double> inverse(original);
      inverse.gauss_jordan();
      for (unsigned int right_hand_side = 0; right_hand_side < 2;
           ++right_hand_side)
        {
          dealii::Vector<double> b(4), reference(4);
          for (unsigned int i = 0; i < 4; ++i)
            {
              b[i]   = right_hand_side ? (i == 1 ? 1. :
                                          i == 3 ? -2e-8 :
                                                   0.) :
                                         (i == 3 ? 4e-8 : i + 1.);
              rhs(i) = b[i];
            }
          rhs.compress(VectorOperation::insert);
          inverse.vmult(reference, b);
          direct.vmult(x, rhs);
          source.vmult(defect, x);
          defect -= rhs;
          const double bound = 64. * std::numeric_limits<double>::epsilon();
          for (unsigned int i = 0; i < 4; ++i)
            {
              AssertThrow(
                std::abs(x(i) - reference[i]) <=
                  bound * std::max(1., std::abs(reference[i])),
                ExcMessage(
                  "Serial solve differs from independent dense inverse."));
              for (unsigned int j = 0; j < 4; ++j)
                AssertThrow(
                  source.el(i, j) == original(i, j),
                  ExcMessage(
                    "Serial factorization changed the original operator."));
            }
          AssertThrow(
            defect.l2_norm() <= bound * rhs.l2_norm(),
            ExcMessage("Serial correction does not satisfy original J*x=rhs."));
        }
    }
  bool rejected = false;
  try
    {
      source.set(0, 0, std::numeric_limits<double>::quiet_NaN());
      source.compress(VectorOperation::insert);
      direct.initialize(source);
    }
  catch (const std::exception &)
    {
      // Debug backends may reject the NaN at insertion; Release must still
      // reject it in the serial adapter before numeric factorization.
      rejected = true;
    }
  AssertThrow(rejected,
              ExcMessage(
                "Non-finite operators must be rejected before factorization."));
#else
  bool rejected = false;
  try
    {
      SerialDirectSolver unavailable;
      unavailable.initialize(source);
    }
  catch (const std::exception &)
    {
      rejected = true;
    }
  AssertThrow(rejected,
              ExcMessage("Unavailable UMFPACK must fail explicitly."));
#endif
  deallog.depth_file(10);
  deallog << "Serial direct adapter checks: PASSED" << std::endl;
}
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  mpi_initlog();
  test();
}
