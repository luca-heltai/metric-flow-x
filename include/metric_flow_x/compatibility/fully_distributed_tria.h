// SPDX-License-Identifier: LGPL-2.1-or-later
#ifndef metric_flow_x_compatibility_fully_distributed_tria_h
#define metric_flow_x_compatibility_fully_distributed_tria_h

#include <deal.II/base/config.h>

#include <deal.II/distributed/fully_distributed_tria.h>

namespace MetricFlowX
{
  namespace Compatibility
  {
    /**
     * Native fully distributed checkpoint I/O with corrected cell offsets.
     *
     * deal.II 9.7.1 (and affected later releases) converts the first-cell
     * index to bytes twice. Override only checkpoint I/O, using the native
     * attached-data serializer and SolutionTransfer, without patching deal.II.
     * The native archive format and same-rank restart restrictions are
     * retained.
     */
    template <int dim, int spacedim = dim>
    class FullyDistributedTriangulation
      : public dealii::parallel::fullydistributed::Triangulation<dim, spacedim>
    {
    public:
      using dealii::parallel::fullydistributed::Triangulation<dim, spacedim>::
        Triangulation;

      void
      save(const std::string &filename) const override;
      void
      load(const std::string &filename) override;
    };
  } // namespace Compatibility
} // namespace MetricFlowX

#endif
