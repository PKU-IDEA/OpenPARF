/**
 * File              : resource_area_kernel.hpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 07.14.2020
 * Last Modified Date: 07.15.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */
#ifndef OPENPARF_OPS_RESOURCE_AREA_SRC_RESOURCE_AREA_KERNEL_HPP_
#define OPENPARF_OPS_RESOURCE_AREA_SRC_RESOURCE_AREA_KERNEL_HPP_

#include <algorithm>
#include <boost/container/flat_map.hpp>
#include <limits>
#include <numeric>

#include "util/arith.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

/// The enum class of packing rules.
enum PackingRules {
  kUltraScale = 0,
  kXarch,
};

template<typename T, typename V>
static void fixedPoint2FloatingPoint(T *target, V const *src, int32_t size, T scale_factor, int32_t num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int32_t i = 0; i < size; ++i) {
    target[i] = src[i] * scale_factor;
  }
}

// See elfPlace paper Fig 6
template<typename T>
static T smooth_ceil(T val, T threshold) {
  T r = OPENPARF_STD_NAMESPACE::fmod(val, 1.0);
  return val - r + OPENPARF_STD_NAMESPACE::min(r / threshold, 1.0);
}

/// @brief fill the demand map for lut with a Gaussian distribution
// Note that the demand map is a #(num_bins_x, num_bins_y, lut_types) tensor
template<typename T, typename DemandMapType, typename Atomic>
void fillGaussianDemandMapKernel(T const *pos,
        uint8_t const                    *is_luts,
        uint8_t const                    *is_ffs,
        int32_t const                    *ff_ctrlsets,
        int32_t                           num_bins_x,
        int32_t                           num_bins_y,
        int32_t                           num_insts,
        int32_t                           cksr_size,
        int32_t                           ce_size,
        T                                 stddev_x,
        T                                 stddev_y,
        T                                 stddev_trunc,
        int32_t                           num_threads,
        DemandMapType                    *lut_demand_map,
        DemandMapType                    *ff_demand_map,
        Atomic                            atomic_add) {
  int32_t ext_bin_x = OPENPARF_STD_NAMESPACE::max<int32_t>(OPENPARF_STD_NAMESPACE::lround(stddev_trunc - 0.5), 0);
  int32_t ext_bin_y = OPENPARF_STD_NAMESPACE::max<int32_t>(OPENPARF_STD_NAMESPACE::lround(stddev_trunc - 0.5), 0);

  OPENPARF_STD_NAMESPACE::vector<T> demX(2 * ext_bin_x + 1);
  OPENPARF_STD_NAMESPACE::vector<T> demY(2 * ext_bin_y + 1);

  int32_t                           chunk_size = OPENPARF_STD_NAMESPACE::max(int32_t(num_insts / num_threads / 16), 1);
  // FIXME(Jing Mai): do NOT use OpenMP here for deterministic results
  // #pragma omp        parallel for num_threads(num_threads) schedule(static)
  for (int32_t i = 0; i < num_insts; ++i) {
    if (is_luts[i] == 0 && is_ffs[i] == 0) continue;
    int32_t       offset        = i << 1;
    const T       inst_center_x = pos[offset];
    const T       inst_center_y = pos[offset | 1];

    const int32_t bin_x         = int32_t(inst_center_x / stddev_x);
    const int32_t bin_y         = int32_t(inst_center_y / stddev_y);

    // Get the bin box, affected by this instance
    const int32_t bin_xl        = (bin_x > ext_bin_x ? bin_x - ext_bin_x : 0);
    const int32_t bin_yl        = (bin_y > ext_bin_y ? bin_y - ext_bin_y : 0);
    const int32_t bin_xh        = OPENPARF_STD_NAMESPACE::min(bin_x + ext_bin_x + 1, num_bins_x);
    const int32_t bin_yh        = OPENPARF_STD_NAMESPACE::min(bin_y + ext_bin_y + 1, num_bins_y);

    // Compute the probability that this instance will fall into each bin box

    // First we use its reciprocal to scale the probability in each bin to make sure the total proabiblity is 1
    const T       sf            = 1.0 /
                 (arithmetic::ComputeGaussianAUC(inst_center_x, stddev_x, bin_xl * stddev_x, bin_xh * stddev_x) *
                         arithmetic::ComputeGaussianAUC(inst_center_y, stddev_y, bin_yl * stddev_y, bin_yh * stddev_y));
    // Compute demand in x/y directions
    for (int32_t x = bin_xl; x < bin_xh; ++x) {
      demX[x - bin_xl] = arithmetic::ComputeGaussianAUC(inst_center_x, stddev_x, x * stddev_x, (x + 1) * stddev_x);
    }
    for (int32_t y = bin_yl; y < bin_yh; ++y) {
      demY[y - bin_yl] = arithmetic::ComputeGaussianAUC(inst_center_y, stddev_y, y * stddev_y, (y + 1) * stddev_y);
    }
    // Compute the demand (probability) in each affected bin and fill the demand map. This part is dependent on whether
    // the instances is a ff or lut
    if (is_luts[i] > 0) {
      for (int32_t x = bin_xl; x < bin_xh; ++x) {
        for (int32_t y = bin_yl; y < bin_yh; ++y) {
          int32_t idx = (x * num_bins_y + y) * 6 + is_luts[i] - 1;
          /**
           * There are actually only 5 types of luts, namely LUT2 to LUT6 with corresponding `is_luts` values are 2
           * to 6. Their storing index is their `is_luts` values minus one, namely 1 to 5.
           * */
#pragma omp atomic update
          lut_demand_map[idx] += sf * demX[x - bin_xl] * demY[y - bin_yl];
        }
      }
    }
    if (is_ffs[i] > 0) {
      for (int32_t x = bin_xl; x < bin_xh; ++x) {
        for (int32_t y = bin_yl; y < bin_yh; ++y) {
          openparfAssert(ff_ctrlsets[2 * i] != std::numeric_limits<int32_t>::max());
          openparfAssert(ff_ctrlsets[2 * i + 1] != std::numeric_limits<int32_t>::max());
          int32_t idx =
                  (x * num_bins_y + y) * cksr_size * ce_size + ff_ctrlsets[2 * i] * ce_size + ff_ctrlsets[2 * i + 1];
// atomic_add(ff_demand_map[idx], sf * demX[x - bin_xl] * demY[y - bin_yl]);
#pragma omp atomic update
          ff_demand_map[idx] += sf * demX[x - bin_xl] * demY[y - bin_yl];
        }
      }
    }
  }
}


template<typename T>
void        computeInstanceAreaMapKernel(T const *lut_demand_map,
               T const                           *ff_demand_map,
               int32_t                            num_bins_x,
               int32_t                            num_bins_y,
               T                                  stddev_x,
               T                                  stddev_y,
               T                                  stddev_trunc,
               int32_t                            cksr_size,
               int32_t                            ce_size,
               int32_t                            slice_capacity,
               int32_t                            num_threads,
               T                                 *lut_area_map,
               T                                 *ff_area_map,
               PackingRules                       packing_rule) {
         // Get bin extension in x and y direction
  // We compute area based on each window [binX - extBinW, binY - extBinY, binX + extBinX, binY + extBinY]
  int32_t extBinX = OPENPARF_STD_NAMESPACE::max<int32_t>(OPENPARF_STD_NAMESPACE::lround(stddev_trunc - 0.5), 0);
  int32_t extBinY = OPENPARF_STD_NAMESPACE::max<int32_t>(OPENPARF_STD_NAMESPACE::lround(stddev_trunc - 0.5), 0);
  int32_t chunk_size = OPENPARF_STD_NAMESPACE::max(int32_t(num_bins_x * num_bins_y / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
  for (int32_t binIdx = 0; binIdx < num_bins_x * num_bins_y; ++binIdx) {
           // Get the window [binXLo, binYLo, binXHi - 1, binYHi - 1] to perform area computation
    int32_t                             binX    = binIdx / num_bins_y;
    int32_t                             binY    = binIdx % num_bins_y;
    int32_t                             binXLo  = (binX > extBinX ? binX - extBinX : 0);
    int32_t                             binYLo  = (binY > extBinY ? binY - extBinY : 0);
    int32_t                             binXHi  = OPENPARF_STD_NAMESPACE::min(binX + extBinX + 1, num_bins_x);
    int32_t                             binYHi  = OPENPARF_STD_NAMESPACE::min(binY + extBinY + 1, num_bins_y);

    // Aggregate demands in this window
    const T                            *ptr     = lut_demand_map + binIdx * 6;
    OPENPARF_STD_NAMESPACE::array<T, 6> lut_dem = {ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5]};
    OPENPARF_STD_NAMESPACE::vector<OPENPARF_STD_NAMESPACE::vector<T>> ff_dem(cksr_size, std::vector<T>(ce_size, 0));
    for (int32_t cksr = 0; cksr < cksr_size; cksr++) {
             for (int32_t ce = 0; ce < ce_size; ce++) {
               int32_t idx      = binIdx * cksr_size * ce_size;
               ff_dem[cksr][ce] = ff_demand_map[idx + cksr * ce_size + ce];
      }
    }

    for (int32_t x = binXLo; x < binXHi; ++x) {
             for (int32_t y = binYLo; y < binYHi; ++y) {
               if (x != binX && y != binY) {
                 for (int32_t i = 0; i < 6; i++) {
                   lut_dem[i] += lut_demand_map[(x * num_bins_y + y) * 6 + i];
          }
                 for (int32_t cksr = 0; cksr < cksr_size; cksr++) {
                   for (int32_t ce = 0; ce < ce_size; ce++) {
                     int32_t idx = (x * num_bins_y + y) * cksr_size * ce_size;
                     ff_dem[cksr][ce] += ff_demand_map[idx + cksr * ce_size + ce];
            }
          }
        }
      }
    }

    // First compute for lut
    // Compute instance areas based on the window demand distribution
    T  winArea        = (binXHi - binXLo) * (binYHi - binYLo) * stddev_x * stddev_y;
    T  lut_total_dem  = OPENPARF_STD_NAMESPACE::accumulate(lut_dem.begin(), lut_dem.end(), (T) 0.0);
    T  space          = OPENPARF_STD_NAMESPACE::max(winArea - lut_total_dem, (T) 0.0);
    T  lut_total_area = lut_total_dem + space;
    T *lut_area       = lut_area_map + binIdx * 6;
    if (packing_rule == PackingRules::kUltraScale) {
             lut_area[0] = (lut_dem[0] + lut_dem[1] + lut_dem[2] + lut_dem[3] + 2.0 * (lut_dem[4] + lut_dem[5] + space)) /
                           lut_total_area;
             lut_area[1] = (lut_dem[0] + lut_dem[1] + lut_dem[2] + 2.0 * (lut_dem[3] + lut_dem[4] + lut_dem[5] + space)) /
                           lut_total_area;
             lut_area[2] = (lut_dem[0] + lut_dem[1] + 2.0 * (lut_dem[2] + lut_dem[3] + lut_dem[4] + lut_dem[5] + space)) /
                           lut_total_area;
             lut_area[3] = (lut_dem[0] + 2.0 * (lut_dem[1] + lut_dem[2] + lut_dem[3] + lut_dem[4] + lut_dem[5] + space)) /
                           lut_total_area;
             lut_area[4] = 2.0;
             lut_area[5] = 2.0;
    } else {
             openparfAssert(lut_dem[0] == 0);
             openparfAssert(lut_dem[1] == 0);                                            // LUT2
             openparfAssert(lut_dem[2] == 0);                                            // LUT3
             openparfAssert(lut_dem[3] == 0);                                            // LUT4
             lut_area[4] = (lut_dem[4] + 2.0 * (lut_dem[5] + space)) / lut_total_area;   // LUT5
             lut_area[5] = 2.0;                                                          // LUT6
    }

    // See elfplace Eq.34
    for (int32_t cksr = 0; cksr < cksr_size; cksr++) {
             T total_quarter = 0;
             for (int32_t ce = 0; ce < ce_size; ce++) {
               total_quarter += smooth_ceil(ff_dem[cksr][ce] * 0.25, 0.25);
      }
             T sf = slice_capacity / 2 * smooth_ceil(total_quarter * 0.5, 0.5) / total_quarter;
             for (int32_t ce = 0; ce < ce_size; ce++) {
               int32_t idx                            = binIdx * cksr_size * ce_size;
               T       quarter                        = smooth_ceil(ff_dem[cksr][ce] * 0.25, 0.25);
               ff_area_map[idx + cksr * ce_size + ce] = sf * quarter / ff_dem[cksr][ce];
      }
    }
  }
}

template<typename T>
void        collectInstanceAreasKernel(T const *pos,
               T const                         *lut_area_map,
               T const                         *ff_area_map,
               uint8_t const                   *is_luts,
               uint8_t const                   *is_ffs,
               int32_t const                   *ff_ctrlsets,
               int32_t                          num_bins_x,
               int32_t                          num_bins_y,
               T                                stddev_x,
               T                                stddev_y,
               T                                stddev_trunc,
               int32_t                          num_insts,
               int32_t                          cksr_size,
               int32_t                          ce_size,
               T                                unit_area,
               int32_t                          num_threads,
               T                               *inst_areas) {
         // Set instance area in the area array one by one
  T lut_total_area = 0;
  T ff_total_area  = 0;
       #pragma omp parallel for num_threads(num_threads) reduction(+ : lut_total_area, ff_total_area)
  for (int32_t i = 0; i < num_insts; ++i) {
           if (is_luts[i] == 0 && is_ffs[i] == 0) {
             continue;
    }
           int32_t       offset        = i << 1;
           const T       inst_center_x = pos[offset];
           const T       inst_center_y = pos[offset | 1];
           const int32_t bin_x         = int32_t(inst_center_x / stddev_x);
           const int32_t bin_y         = int32_t(inst_center_y / stddev_y);
           int32_t       idx           = bin_x * num_bins_y + bin_y;
           if (is_luts[i]) {
             inst_areas[i] = lut_area_map[idx * 6 + is_luts[i] - 1] * unit_area;
             lut_total_area += inst_areas[i];
    } else if (is_ffs[i]) {
             openparfAssert(ff_ctrlsets[2 * i] != std::numeric_limits<int32_t>::max());
             openparfAssert(ff_ctrlsets[2 * i + 1] != std::numeric_limits<int32_t>::max());
             inst_areas[i] = ff_area_map[idx * cksr_size * ce_size + ff_ctrlsets[2 * i] * ce_size + ff_ctrlsets[2 * i + 1]] *
                             unit_area;
             ff_total_area += inst_areas[i];
    }
  }
       }

template<typename T>
void OPENPARF_NOINLINE fillGaussianDemandMapLauncher(T const *pos,
        uint8_t const                                        *is_luts,
        uint8_t const                                        *is_ffs,
        int32_t const                                        *ff_ctrlsets,
        int32_t                                               num_bins_x,
        int32_t                                               num_bins_y,
        int32_t                                               num_insts,
        int32_t                                               cksr_size,
        int32_t                                               ce_size,
        T                                                     stddev_x,
        T                                                     stddev_y,
        T                                                     stddev_trunc,
        int32_t                                               num_threads,
        T                                                    *lut_demand_map,
        T                                                    *ff_demand_map) {

  // AtomicAdd<int64_t> atomic_add(0x100000);
  auto atomic_add = [](T &a, T b) {
#pragma omp atomic update
    a += b;
  };
  // OPENPARF_STD_NAMESPACE::vector<int64_t> lut_demand_map_fixed(num_bins_x * num_bins_y * 6, 0);
  // OPENPARF_STD_NAMESPACE::vector<int64_t> ff_demand_map_fixed(num_bins_x * num_bins_y * cksr_size * ce_size, 0);
  fillGaussianDemandMapKernel(pos, is_luts, is_ffs, ff_ctrlsets, num_bins_x, num_bins_y, num_insts, cksr_size, ce_size,
          stddev_x, stddev_y, stddev_trunc, num_threads, lut_demand_map, ff_demand_map, atomic_add);
  /*fixedPoint2FloatingPoint(lut_demand_map,
                           lut_demand_map_fixed.data(),
                           lut_demand_map_fixed.size(),
                           (T) 1.0 / atomic_add.scaleFactor(), num_threads);
  fixedPoint2FloatingPoint(ff_demand_map,
                           ff_demand_map_fixed.data(),
                           ff_demand_map_fixed.size(),
                           (T) 1.0 / atomic_add.scaleFactor(), num_threads);
  */
}

template<typename T>
void OPENPARF_NOINLINE computeInstanceAreaMapLauncher(T const *lut_demand_map,
        T const                                               *ff_demand_map,
        int32_t                                                num_bins_x,
        int32_t                                                num_bins_y,
        T                                                      stddev_x,
        T                                                      stddev_y,
        T                                                      stddev_trunc,
        int32_t                                                cksr_size,
        int32_t                                                ce_size,
        int32_t                                                slice_capacity,
        int32_t                                                num_threads,
        T                                                     *lut_area_map,
        T                                                     *ff_area_map,
        PackingRules                                           packing_rule) {
  computeInstanceAreaMapKernel(lut_demand_map, ff_demand_map, num_bins_x, num_bins_y, stddev_x, stddev_y, stddev_trunc,
          cksr_size, ce_size, slice_capacity, num_threads, lut_area_map, ff_area_map, packing_rule);
}


template<typename T>
void OPENPARF_NOINLINE collectInstanceAreasLauncher(T const *pos,
        T const                                             *lut_area_map,
        T const                                             *ff_area_map,
        uint8_t const                                       *is_luts,
        uint8_t const                                       *is_ffs,
        int32_t const                                       *ff_ctrlsets,
        int32_t                                              num_bins_x,
        int32_t                                              num_bins_y,
        int32_t                                              num_insts,
        int32_t                                              cksr_size,
        int32_t                                              ce_size,
        int32_t                                              slice_capacity,
        T                                                    stddev_x,
        T                                                    stddev_y,
        T                                                    stddev_trunc,
        int32_t                                              num_threads,
        T                                                   *inst_areas) {

  collectInstanceAreasKernel(pos, lut_area_map, ff_area_map, is_luts, is_ffs, ff_ctrlsets, num_bins_x, num_bins_y,
          stddev_x, stddev_y, stddev_trunc, num_insts, cksr_size, ce_size, (T) 1 / slice_capacity, num_threads,
          inst_areas);
}

inline void computeControlSetsLauncher(int32_t const *inst_pins,
        int32_t const                                *inst_pins_start,
        int32_t const                                *pin2net_map,
        uint8_t const                                *is_inst_FFs,
        uint8_t const                                *pin_signal_types,
        int32_t                                       num_insts,
        int32_t                                      *ff_ctrlsets,   ///< #insts x 2
        int32_t                                      &num_cksr,
        int32_t                                      &num_ce) {
  boost::container::flat_map<int32_t, int32_t>                                      ce_map;
  boost::container::flat_map<int32_t, boost::container::flat_map<int32_t, int32_t>> cksr_map;
  int32_t                                                                           cksr_id = 0;
  int32_t                                                                           ce_id   = 0;
  std::fill(ff_ctrlsets, ff_ctrlsets + num_insts * 2, std::numeric_limits<int32_t>::max());

  for (int32_t i = 0; i < num_insts; ++i) {
    if (is_inst_FFs[i]) {
      // Find the net ID of CK, CE, and SR pins
      int32_t ck, sr, ce;
      ck = sr = ce = std::numeric_limits<int32_t>::max();
      auto bgn     = inst_pins_start[i];
      auto end     = inst_pins_start[i + 1];
      for (int32_t j = bgn; j < end; ++j) {
        auto pin_id = inst_pins[j];
        switch (toEnum<SignalType>(pin_signal_types[pin_id])) {
          case SignalType::kClock:
            ck = pin2net_map[pin_id];
            break;
          case SignalType::kControlSR:
            sr = pin2net_map[pin_id];
            break;
          case SignalType::kControlCE:
            ce = pin2net_map[pin_id];
            break;
          default:
            break;
        }
      }

      // Set the CK/SR mapping
      auto ck_it = cksr_map.find(ck);
      if (ck_it == cksr_map.end()) {
        ff_ctrlsets[i * 2] = cksr_id;
        cksr_map[ck][sr]   = cksr_id++;
      } else {
        auto &sr_map = ck_it->second;
        auto  sr_it  = sr_map.find(sr);
        if (sr_it == sr_map.end()) {
          ff_ctrlsets[i * 2] = cksr_id;
          sr_map[sr]         = cksr_id++;
        } else {
          ff_ctrlsets[i * 2] = sr_it->second;
        }
      }

      // Set the CE mapping
      auto ce_it = ce_map.find(ce);
      if (ce_it == ce_map.end()) {
        ff_ctrlsets[i * 2 + 1] = ce_id;
        ce_map[ce]             = ce_id++;
      } else {
        ff_ctrlsets[i * 2 + 1] = ce_it->second;
      }
    }
  }

  num_cksr = cksr_id;
  num_ce   = ce_id;
}

OPENPARF_END_NAMESPACE

#endif   // OPENPARF_OPS_RESOURCE_AREA_SRC_RESOURCE_AREA_KERNEL_HPP_
