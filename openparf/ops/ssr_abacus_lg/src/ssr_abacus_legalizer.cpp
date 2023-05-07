/**
 * File              : ssr_abacus_legalizer.cpp
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 12.02.2021
 * Last Modified Date: 12.02.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#include "ssr_abacus_legalizer.h"

// C++ standard libraries' headers
#include <algorithm>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// project headers
#include "container/spiral_accessor.hpp"
#include "container/vector_2d.hpp"

OPENPARF_BEGIN_NAMESPACE

namespace ssr_abacus_legalizer {

using container::SpiralAccessor;
using container::Vector2D;
using container::XY;
using database::Resource;
using database::Site;
using database::SiteMap;

using SiteRefVec = std::vector<std::reference_wrapper<const Site>>;

namespace detail {

int32_t SelectResource(database::PlaceDB const &placedb, int32_t area_type_id) {
  int32_t num_resources   = placedb.numResources();
  int32_t selected_rsc_id = -1;
  for (int32_t rsc_id = 0; rsc_id < num_resources; rsc_id++) {
    const auto &area_type_ids = placedb.resourceAreaTypes(Resource(rsc_id));
    bool is_contained = std::find(area_type_ids.begin(), area_type_ids.end(), area_type_id) != area_type_ids.end();
    if (is_contained) {
      openparfAssert(selected_rsc_id == -1);
      selected_rsc_id = rsc_id;
    }
  }
  return selected_rsc_id;
}

std::vector<int32_t> BuildValidSiteMap(const SiteMap &site_map) {
  std::vector<int32_t> valid_site_map(site_map.width() * site_map.height(), std::numeric_limits<int32_t>::max());
  for (auto const &site : site_map) {
    auto const &bbox = site.bbox();
    for (int32_t ix = bbox.xl(); ix < bbox.xh(); ++ix) {
      for (int32_t iy = bbox.yl(); iy < bbox.yh(); ++iy) {
        valid_site_map.at(ix * site_map.height() + iy) = site_map.index1D(site.siteMapId().x(), site.siteMapId().y());
      }
    }
  }
  return valid_site_map;
}

class Cell {
 public:
  explicit Cell(XY<float> pos, float height, int32_t original_index)
      : pos_(pos),
        height_(height),
        original_index_(original_index) {}

  // Identity oriented getters and setters
  XY<float>       &pos() { return pos_; }
  const XY<float> &pos() const { return pos_; }
  float           &height() { return height_; }
  const float     &height() const { return height_; }
  float            yh() { return pos_.y() + height_; }
  float            yl() { return pos_.y(); }
  int32_t         &original_index() { return original_index_; }
  const int32_t   &original_index() const { return original_index_; }

 private:
  XY<float> pos_;
  float     height_;
  int32_t   original_index_;
};

class Cluster {
 public:
  std::vector<Cell>        cells;
  std::vector<float>       cell_offsets;

  float                    height() const { return cell_offsets.back() + cells.back().height(); }

  std::shared_ptr<Cluster> next;

  float                    cost;
  float                    prefix_cost;

  float                    Yc;

  explicit Cluster(Cell c, std::shared_ptr<Cluster> next_ptr)
      : Yc(c.pos().y()),
        cost(0),
        prefix_cost(0),
        next(std::move(next_ptr)) {
    cells.push_back(c);
    cell_offsets.push_back(0);
  }

  /**
   * @brief return true if |this| is below the |other|
   *
   * @param other
   * @return true
   * @return false
   */
  bool overlap_with(const Cluster &other) const {
    float top    = this->Yc;
    float bottom = other.Yc + other.height();
    return top < bottom;
  }

  /**
   * @brief |this| must have lower y than |other|. Meanwhile, the caller should take care of
   * the |next| pointer of the return Cluster object. This pointer by default is set
   * to |this->next|
   * @param other
   * @return Cluster
   */
  Cluster merge_with(const Cluster &other) const {
    Cluster cluster      = *this;

    auto    cells_to_add = other.cells;

    for (auto &cell : cells_to_add) {
      float cell_offset = cluster.height();
      cluster.cells.push_back(cell);
      cluster.cell_offsets.push_back(cell_offset);
    }

    return cluster;
  }

  void update(int32_t yl, int32_t yh, int32_t site_height) {
    update_Yc(yl, yh, site_height);
    update_cost();
  }

 private:
  void update_cost() {
    float new_cost = 0;

    for (size_t i = 0; i < cells.size(); i++) {
      auto &cell   = cells[i];
      float weight = cell.height();
      float off    = Yc + cell_offsets[i] - cell.pos().y();
      new_cost += off * off * weight;
    }

    cost = new_cost;
    if (next != nullptr) {
      prefix_cost = next->prefix_cost + cost;
    }
  }

  /**
   * Updated the optimal y coordinate of the left-bottom corner of this cluster
   * @param n_rows # of rows
   */
  void update_Yc(float yl, float yh, int32_t site_height) {
    float coefficient = 0;
    float weight_sum  = 0;

    for (size_t i = 0; i < cells.size(); i++) {
      auto &cell   = cells[i];
      float weight = cell.height();
      coefficient += (cell.pos().y() - cell_offsets[i]) * weight;
      weight_sum += weight;
    }

    float optimal_y = coefficient / weight_sum;
    // round to the nearest yl + site_height * k
    Yc              = std::round((optimal_y - yl) / site_height) * site_height + yl;

    if (Yc + height() > yh) {
      Yc = yh - height();
      assert(Yc >= 0);
    }
    if (Yc < yl) {
      Yc = yl;
    }
  }
};

class ColumnPlacement {
 public:
  ColumnPlacement() : root(nullptr), height_sum(0) {}
  std::shared_ptr<Cluster> root;
  float                    height_sum;
};
}   // namespace detail

template<class T>
void DispatchedSsrLegalizerForward(database::PlaceDB const &placedb,
                                   int32_t                  num_concerned_insts,
                                   T                       *pos,
                                   int32_t                 *concerned_inst_ids,
                                   int32_t                  area_type_id,
                                   int32_t                  num_chains,
                                   int32_t                 *chain_ssr_ids_bs,
                                   int32_t                 *chain_ssr_ids_b_starts) {
  using detail::Cell;
  using detail::Cluster;
  using detail::ColumnPlacement;
  auto const                  &layout         = placedb.db()->layout();
  auto const                  &place_params   = placedb.place_params();
  int32_t                      rsc_id         = detail::SelectResource(placedb, area_type_id);
  const database::SiteMap     &site_map       = layout.siteMap();
  std::vector<int32_t>         valid_site_map = detail::BuildValidSiteMap(site_map);
  int32_t                      num_site_x     = placedb.siteMapDim().x();
  int32_t                      num_site_y     = placedb.siteMapDim().y();
  std::vector<int32_t>         ssr_col_heights(num_site_x, 0);
  std::vector<int32_t>         col_yl(num_site_x, num_site_y);
  std::vector<int32_t>         col_yh(num_site_x, 0);
  int32_t                      total_ssr_sup = 0;
  std::vector<int32_t>         col_ssr_sups(num_site_x, 0);
  std::vector<ColumnPlacement> heads(num_site_x);
  std::vector<Cell>            cells;
  int32_t                      site_height = -1;

  for (const Site &site : site_map) {
    if (layout.siteType(site).resourceCapacity(rsc_id) > 0) {
      int32_t xl = site.bbox().xl();
      int32_t yl = site.bbox().yl();
      int32_t yh = site.bbox().yh();
      col_yl[xl] = std::min(col_yl[xl], yl);
      col_yh[xl] = std::max(col_yh[xl], yh);
      if (site_height == -1) {
        site_height = yh - yl;
      } else {
        openparfAssert(site_height == yh - yl);
      }
      ssr_col_heights[xl] += yh - yl;
    }
  }

  openparfPrint(kDebug, "area type: %s(%d)\n", place_params.area_type_names_[area_type_id].c_str(), area_type_id);
  openparfPrint(kDebug, "resouce type: %d\n", rsc_id);
  openparfPrint(kDebug, "site height: %d\n", site_height);

  cells.empty();
  for (int chain_id = 0; chain_id < num_chains; chain_id++) {
    int   st       = chain_ssr_ids_b_starts[chain_id];
    int   en       = chain_ssr_ids_b_starts[chain_id + 1];
    int   len      = en - st;
    float center_x = 0;
    float center_y = 0;
    float height   = site_height * len;
    float bl_x;
    float bl_y;
    for (int i = st; i < en; i++) {
      int inst_id = chain_ssr_ids_bs[i];
      center_x += pos[inst_id << 1];
      center_y += pos[inst_id << 1 | 1];
    }
    center_x /= len;
    center_y /= len;
    bl_x = center_x - 0.5;
    bl_y = center_y - height * 0.5;
    cells.emplace_back(XY<float>(bl_x, bl_y), height, chain_id);
  }

  std::sort(cells.begin(), cells.end(), [](const Cell &a, const Cell &b) -> bool {
    return a.height() == b.height() ? a.pos().y() < b.pos().y() : a.height() > b.height();
  });

  for (auto &cell : cells) {
    float                    best_added_cost = std::numeric_limits<float>::max();
    int                      best_col        = std::numeric_limits<int>::max();
    std::shared_ptr<Cluster> best_col_first_cluster;
    int                      ix = std::floor(cell.pos().x());
    for (int dx = 0; dx < num_site_x; dx++) {
      for (int sign : {1, -1}) {
        int Xc = ix + dx * sign;
        if (!(0 <= Xc && Xc < num_site_x)) {
          continue;
        }

        float x_cost = (cell.pos().x() - Xc) * (cell.pos().x() - Xc);

        if (x_cost >= best_added_cost) {
          continue;
        }

        ColumnPlacement &head               = heads[Xc];

        float            column_origin_cost = head.root != nullptr ? head.root->prefix_cost : 0;

        if (head.height_sum + cell.height() > ssr_col_heights[Xc]) {
          // the column overflows
          continue;
        }

        auto cluster = std::make_shared<Cluster>(cell, head.root);
        cluster->update(col_yl[Xc], col_yh[Xc], site_height);

        while (cluster->next != nullptr && cluster->overlap_with(*cluster->next)) {
          *cluster = cluster->next->merge_with(*cluster);
          cluster->update(col_yl[Xc], col_yh[Xc], site_height);
        }

        float added_cost = x_cost + cluster->prefix_cost - column_origin_cost;
        if (added_cost < best_added_cost) {
          best_col               = Xc;
          best_added_cost        = added_cost;
          best_col_first_cluster = cluster;
        }
      }
    }

    assert(best_col != std::numeric_limits<int>::max());

    /* update */ {
      ColumnPlacement &head = heads[best_col];
      head.root             = best_col_first_cluster;
      head.height_sum += cell.height();
    }
  }

  for (size_t Xc = 0; Xc < num_site_x; Xc++) {
    ColumnPlacement &head = heads[Xc];
    for (auto cluster = head.root; cluster; cluster = cluster->next) {
      for (size_t j = 0; j < cluster->cells.size(); j++) {
        auto     &cell = cluster->cells[j];
        XY<float> rv(Xc, cluster->Yc + cluster->cell_offsets[j]);
        int       chain_id = cell.original_index();
        int       ssr_st   = chain_ssr_ids_b_starts[chain_id];
        int       ssr_en   = chain_ssr_ids_b_starts[chain_id + 1];
        for (int i = ssr_st; i < ssr_en; i++) {
          int   ssr_inst_id         = chain_ssr_ids_bs[i];
          float lb_y                = rv.y() + site_height * (i - ssr_st);
          pos[ssr_inst_id << 1]     = rv.x() + 0.5;
          pos[ssr_inst_id << 1 | 1] = std::floor(lb_y) + 0.5;
        }
      }
    }
  }

  if (false) {
    // debug print
    std::ofstream of("pos_ssr_abacus.txt");
    for (int i = 0; i < placedb.numInsts(); i++) {
      std::string name = placedb.instName(i);
      of << name.c_str() << " " << pos[i << 1] << " " << pos[i << 1 | 1] << std::endl;
    }
    of.close();
  }
}

void SsrLegalizerForward(database::PlaceDB const &placedb,
                         at::Tensor               pos,
                         at::Tensor               concerned_inst_ids,
                         int32_t                  area_type_id,
                         at::Tensor               chain_ssr_ids_bs,
                         at::Tensor               chain_ssr_ids_b_starts) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(concerned_inst_ids);
  CHECK_CONTIGUOUS(concerned_inst_ids);
  CHECK_FLAT_CPU(chain_ssr_ids_bs);
  CHECK_CONTIGUOUS(chain_ssr_ids_bs);
  CHECK_FLAT_CPU(chain_ssr_ids_b_starts);
  CHECK_CONTIGUOUS(chain_ssr_ids_b_starts);
  AT_ASSERTM(concerned_inst_ids.dtype() == torch::kInt32, "`concerned_inst_ids` must be a Int32 tensor.");
  int     num_concerned_insts = concerned_inst_ids.numel();
  int32_t num_chains          = chain_ssr_ids_b_starts.numel() - 1;
  OPENPARF_DISPATCH_FLOATING_TYPES(pos, "ChainLegalizerForward", [&] {
    DispatchedSsrLegalizerForward(placedb,
                                  num_concerned_insts,
                                  OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),
                                  OPENPARF_TENSOR_DATA_PTR(concerned_inst_ids, int32_t),
                                  area_type_id,
                                  num_chains,
                                  OPENPARF_TENSOR_DATA_PTR(chain_ssr_ids_bs, int32_t),
                                  OPENPARF_TENSOR_DATA_PTR(chain_ssr_ids_b_starts, int32_t));
  });
}

#define REGISTER_KERNEL_LAUNCHER(T)                                                                                    \
  template void DispatchedSsrLegalizerForward(database::PlaceDB const &placedb,                                        \
                                              int32_t                  num_concerned_insts,                            \
                                              T                       *pos,                                            \
                                              int32_t                 *concerned_inst_ids,                             \
                                              int32_t                  area_type_id,                                   \
                                              int32_t                  num_chains,                                     \
                                              int32_t                 *chain_ssr_ids_bs,                               \
                                              int32_t                 *chain_ssr_ids_b_starts);

REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

#undef REGISTER_KERNEL_LAUNCHER

}   // namespace ssr_abacus_legalizer

OPENPARF_END_NAMESPACE
