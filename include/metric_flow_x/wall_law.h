// SPDX-License-Identifier: MIT
#ifndef METRIC_FLOW_X_WALL_LAW_H
#define METRIC_FLOW_X_WALL_LAW_H

#include <deal.II/base/exceptions.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace MetricFlowX
{
  /** Arterial (3) and venous (6) laws of Müller et al. (2023),
   * doi:10.3389/fphys.2023.1162391. All coefficients use SI units.
   */
  struct WallLaw
  {
    enum class Type
    {
      square_root = 0,
      artery      = 1,
      vein        = 2,
      dural_sinus = 3
    };
    Type   type = Type::square_root;
    double ee = 0., ec = 0., epsilon0 = 0., epsilonr = 0.;
    double stiffness = 0., m = 0., n = 0., reference_pressure = 0., km = 0.;

    struct Elastic
    {
      double pressure, first, second;
    };

    Elastic
    elastic(const double area,
            const double reference_area,
            const double thickness) const
    {
      using namespace dealii;
      AssertThrow(
        std::isfinite(area) && area > 0. && std::isfinite(reference_area) &&
          reference_area > 0.,
        ExcMessage(
          "Wall law requires finite positive area and reference area."));
      AssertThrow(type != Type::square_root,
                  ExcMessage(
                    "Use the existing vessel law for Square root walls."));
      const double ratio = area / reference_area;
      if (type == Type::artery)
        {
          const double s = std::sqrt(ratio), strain = s - 1.;
          const double chi     = (strain - epsilon0) / epsilonr;
          const double sigmoid = chi >= 0. ?
                                   1. / (1. + std::exp(-chi)) :
                                   std::exp(chi) / (1. + std::exp(chi));
          const double softplus =
            std::max(chi, 0.) + std::log1p(std::exp(-std::abs(chi)));
          const double f   = ee * strain + ec * epsilonr * softplus;
          const double df  = ee + ec * sigmoid;
          const double ddf = ec * sigmoid * (1. - sigmoid) / epsilonr;
          const double factor =
            thickness * std::sqrt(std::acos(-1.) * reference_area);
          return {reference_pressure + factor / area * f,
                  factor / (area * area) * (-f + .5 * s * df),
                  factor / (area * area * area) *
                    (2. * f - 1.25 * s * df + .25 * s * s * ddf)};
        }
      const double am = std::pow(ratio, m), an = std::pow(ratio, n);
      return {reference_pressure + stiffness * (am - an),
              stiffness / area * (m * am - n * an),
              stiffness / (area * area) *
                (m * (m - 1.) * am - n * (n - 1.) * an)};
    }

    double
    viscosity(const double area, const double reference_area) const
    {
      AssertThrow(
        std::isfinite(area) && area > 0. && std::isfinite(reference_area) &&
          reference_area > 0.,
        dealii::ExcMessage(
          "Viscoelastic wall requires finite positive area and reference area."));
      return type == Type::artery ?
               km / (2. * std::sqrt(area * reference_area)) :
               km / (reference_area * std::sqrt(area));
    }

    double
    wave_speed(const double area,
               const double reference_area,
               const double thickness,
               const double density) const
    {
      AssertThrow(std::isfinite(density) && density > 0.,
                  dealii::ExcMessage(
                    "Wall wave speed requires finite positive density."));
      const double modulus =
        area * elastic(area, reference_area, thickness).first;
      AssertThrow(
        std::isfinite(modulus) && modulus > 0.,
        dealii::ExcMessage(
          "ADAVN wall law is non-hyperbolic at this area (A*dP/dA <= 0)."));
      return std::sqrt(modulus / density);
    }

    /** Integral of c(A)/A from the reference state. Eight-point Gauss
     * integration in log(A/Aref), with intervals no wider than 1/4.
     */
    double
    invariant(const double area,
              const double reference_area,
              const double thickness,
              const double density) const
    {
      AssertThrow(
        std::isfinite(area) && area > 0. && std::isfinite(reference_area) &&
          reference_area > 0.,
        dealii::ExcMessage(
          "Wall invariant requires finite positive area and reference area."));
      static constexpr std::array<double, 4> x = {.1834346424956498,
                                                  .5255324099163290,
                                                  .7966664774136267,
                                                  .9602898564975363};
      static constexpr std::array<double, 4> w = {.3626837833783620,
                                                  .3137066458778873,
                                                  .2223810344533745,
                                                  .1012285362903763};
      const double       limit = std::log(area / reference_area);
      const unsigned int intervals =
        std::max(1u,
                 static_cast<unsigned int>(std::ceil(std::abs(limit) / .25)));
      double result = 0.;
      for (unsigned int i = 0; i < intervals; ++i)
        {
          const double half   = limit / (2. * intervals),
                       center = (2. * i + 1.) * half;
          for (unsigned int q = 0; q < x.size(); ++q)
            for (const double sign : {-1., 1.})
              result += half * w[q] *
                        wave_speed(reference_area *
                                     std::exp(center + sign * half * x[q]),
                                   reference_area,
                                   thickness,
                                   density);
        }
      return result;
    }
  };
} // namespace MetricFlowX
#endif
