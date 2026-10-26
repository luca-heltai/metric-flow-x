#ifndef metric_flow_x_ida_access_h
#define metric_flow_x_ida_access_h

#include <deal.II/sundials/ida.h>

#include <metric_flow_x/config.h>
#ifdef METRIC_FLOW_X_WITH_PETSC
#  include <deal.II/lac/petsc_vector.h>
#else
#  include <deal.II/lac/trilinos_vector.h>
#endif

namespace MetricFlowX
{
  namespace internal
  {
    // Defined beside the IDA statistics compatibility shim.
#ifdef METRIC_FLOW_X_WITH_PETSC
    void *
    native_ida_memory(
      const dealii::SUNDIALS::IDA<dealii::PETScWrappers::MPI::Vector> &ida);
#else
    void *
    native_ida_memory(
      const dealii::SUNDIALS::IDA<dealii::TrilinosWrappers::MPI::Vector> &ida);
#endif
#if DEAL_II_SUNDIALS_VERSION_GTE(7, 0, 0)
#  ifdef METRIC_FLOW_X_WITH_PETSC
    SUNContext
    native_ida_context(
      const dealii::SUNDIALS::IDA<dealii::PETScWrappers::MPI::Vector> &ida);
#  else
    SUNContext
    native_ida_context(
      const dealii::SUNDIALS::IDA<dealii::TrilinosWrappers::MPI::Vector> &ida);
#  endif
#endif
  } // namespace internal

} // namespace MetricFlowX
#endif
