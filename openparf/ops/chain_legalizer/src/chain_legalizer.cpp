/**
 * File              : chain_legalizer.cpp
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 09.10.2021
 * Last Modified Date: 09.10.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#include "chain_legalizer.h"

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

// other libraries's headers
#include "lemon/list_graph.h"
#include "lemon/network_simplex.h"

// project headers
#include "container/spiral_accessor.hpp"
#include "container/vector_2d.hpp"

OPENPARF_BEGIN_NAMESPACE

namespace chain_legalizer {

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
  XY<float> &      pos() { return pos_; }
  const XY<float> &pos() const { return pos_; }
  float &          height() { return height_; }
  const float &    height() const { return height_; }
  float            yh() { return pos_.y() + height_; }
  float            yl() { return pos_.y(); }
  int32_t &        original_index() { return original_index_; }
  const int32_t &  original_index() const { return original_index_; }

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

  void update(int32_t yl, int32_t yh) {
    update_Yc(yl, yh);
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
  void update_Yc(float yl, float yh) {
    float coefficient = 0;
    float weight_sum  = 0;

    for (size_t i = 0; i < cells.size(); i++) {
      auto &cell   = cells[i];
      float weight = cell.height();
      coefficient += (cell.pos().y() - cell_offsets[i]) * weight;
      weight_sum += weight;
    }

    float optimal_y = coefficient / weight_sum;
    // round to the nearest 0.5k
    Yc              = std::round(optimal_y * 2) / 2.;

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
void DispatchedChainLegalizerForward(database::PlaceDB const &placedb,
                                     int32_t                  num_concerned_insts,
                                     T *                      pos_xyz,
                                     int32_t *                concerned_inst_ids,
                                     int32_t                  area_type_id,
                                     int32_t                  search_manh_dist_increment,
                                     int32_t                  max_iter,
                                     int32_t                  num_chains,
                                     int32_t *                chain_cla_ids_bs,
                                     int32_t *                chain_cla_ids_b_starts,
                                     int32_t *                chain_lut_ids_bs,
                                     int32_t *                chain_lut_ids_b_starts) {
  using detail::Cell;
  using detail::Cluster;
  using detail::ColumnPlacement;
  auto const &                 layout         = placedb.db()->layout();
  auto const &                 place_params   = placedb.place_params();
  int32_t                      rsc_id         = detail::SelectResource(placedb, area_type_id);
  const database::SiteMap &    site_map       = layout.siteMap();
  std::vector<int32_t>         valid_site_map = detail::BuildValidSiteMap(site_map);
  int32_t                      num_site_x     = placedb.siteMapDim().x();
  int32_t                      num_site_y     = placedb.siteMapDim().y();
  std::vector<int32_t>         cla_col_heights(num_site_x, 0);
  std::vector<int32_t>         col_yl(num_site_x, num_site_y);
  std::vector<int32_t>         col_yh(num_site_x, 0);
  int32_t                      total_cla_sup = 0;
  std::vector<int32_t>         col_cla_sups(num_site_x, 0);
  std::vector<ColumnPlacement> heads(num_site_x);
  std::vector<Cell>            cells;

  for (const Site &site : site_map) {
    if (layout.siteType(site).resourceCapacity(rsc_id) > 0) {
      int32_t xl = site.bbox().xl();
      int32_t yl = site.bbox().yl();
      col_yl[xl] = std::min(col_yl[xl], yl);
      col_yh[xl] = std::max(col_yh[xl], yl);
      cla_col_heights[xl] += 1;
    }
  }

  openparfPrint(kDebug, "area type: %s(%d)\n", place_params.area_type_names_[area_type_id].c_str(), area_type_id);
  openparfPrint(kDebug, "resouce type: %d\n", rsc_id);

  cells.empty();
  for (int chain_id = 0; chain_id < num_chains; chain_id++) {
    int   st       = chain_cla_ids_b_starts[chain_id];
    int   en       = chain_cla_ids_b_starts[chain_id + 1];
    int   len      = en - st;
    float center_x = 0;
    float center_y = 0;
    float height   = 0.5 * len;
    float bl_x;
    float bl_y;
    for (int i = st; i < en; i++) {
      int inst_id = chain_cla_ids_bs[i];
      center_x += pos_xyz[inst_id * 3];
      center_y += pos_xyz[inst_id * 3 + 1];
    }
    center_x /= len;
    center_y /= len;
    bl_x = center_x - 0.5;
    bl_y = center_y - height * 0.5;
    cells.emplace_back(XY<float>(bl_x, bl_y), height, chain_id);
  }

  sort(cells.begin(), cells.end(), [](const Cell &a, const Cell &b) -> bool { return a.pos().y() < b.pos().y(); });

  for (auto &cell : cells) {
    float                    best_added_cost = std::numeric_limits<float>::max();
    int                      best_col = std::numeric_limits<int>::max();
    std::shared_ptr<Cluster> best_col_first_cluster;
    int                      ix = std::floor(cell.pos().x());
    for (int dx = 0; dx < 2; dx++) {
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

        if (head.height_sum + cell.height() > cla_col_heights[Xc]) {
          // the column overflows
          continue;
        }

        auto cluster = std::make_shared<Cluster>(cell, head.root);
        cluster->update(col_yl[Xc], col_yh[Xc]);

        while (cluster->next != nullptr && cluster->overlap_with(*cluster->next)) {
          *cluster = cluster->next->merge_with(*cluster);
          cluster->update(col_yl[Xc], col_yh[Xc]);
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
        auto &    cell = cluster->cells[j];
        XY<float> rv(Xc, cluster->Yc + cluster->cell_offsets[j]);
        int       chain_id = cell.original_index();
        int       cla_st   = chain_cla_ids_b_starts[chain_id];
        int       cla_en   = chain_cla_ids_b_starts[chain_id + 1];
        int       lut_st   = chain_lut_ids_b_starts[chain_id];
        int       lut_en   = chain_lut_ids_b_starts[chain_id + 1];
        for (int i = cla_st; i < cla_en; i++) {
          int   cla_inst_id            = chain_cla_ids_bs[i];
          float lb_y                   = rv.y() + 0.5 * (i - cla_st);
          pos_xyz[cla_inst_id * 3]     = rv.x() + 0.5;
          pos_xyz[cla_inst_id * 3 + 1] = std::floor(lb_y) + 0.5;
          pos_xyz[cla_inst_id * 3 + 2] = std::fabs(lb_y - std::round(lb_y)) < 0.1 ? 0 : 1;
          float x                      = pos_xyz[cla_inst_id * 3];
          float y                      = pos_xyz[cla_inst_id * 3 + 1];
          int   z                      = pos_xyz[cla_inst_id * 3 + 2] == 0 ? 1 : 9;
          for (int j = 0; j < 4; j++) {
            int lut_inst_id              = chain_lut_ids_bs[lut_st + (i - cla_st) * 4 + j];
            pos_xyz[lut_inst_id * 3]     = x;
            pos_xyz[lut_inst_id * 3 + 1] = y;
            pos_xyz[lut_inst_id * 3 + 2] = z + j * 2;
          }
        }
      }
    }
  }

  if (false) {
    std::ofstream of("pos_xyz.txt");
    for (int i = 0; i < placedb.numInsts(); i++) {
      std::string name = placedb.instName(i);
      of << name.c_str() << " " << pos_xyz[i * 3] << " " << pos_xyz[i * 3 + 1] << " " << pos_xyz[i * 3 + 2]
         << std::endl;
    }
    of.close();
  }
}

void ChainLegalizerForward(database::PlaceDB const &placedb,
                           at::Tensor               pos_xyz,
                           at::Tensor               concerned_inst_ids,
                           int32_t                  area_type_id,
                           int32_t                  search_manh_dist_increment,
                           int32_t                  max_iter,
                           at::Tensor               chain_cla_ids_bs,
                           at::Tensor               chain_cla_ids_b_starts,
                           at::Tensor               chain_lut_ids_bs,
                           at::Tensor               chain_lut_ids_b_starts) {
  CHECK_FLAT_CPU(pos_xyz);
  CHECK_DIVISIBLE(pos_xyz, 3);
  CHECK_CONTIGUOUS(pos_xyz);
  CHECK_FLAT_CPU(concerned_inst_ids);
  CHECK_CONTIGUOUS(concerned_inst_ids);
  CHECK_FLAT_CPU(chain_cla_ids_bs);
  CHECK_CONTIGUOUS(chain_cla_ids_bs);
  CHECK_FLAT_CPU(chain_cla_ids_b_starts);
  CHECK_CONTIGUOUS(chain_cla_ids_b_starts);
  CHECK_FLAT_CPU(chain_lut_ids_bs);
  CHECK_CONTIGUOUS(chain_lut_ids_bs);
  CHECK_FLAT_CPU(chain_lut_ids_b_starts);
  CHECK_CONTIGUOUS(chain_lut_ids_b_starts);
  AT_ASSERTM(concerned_inst_ids.dtype() == torch::kInt32, "`concerned_inst_ids` must be a Int32 tensor.");
  int     num_concerned_insts = concerned_inst_ids.numel();
  int32_t num_chains          = chain_cla_ids_b_starts.numel() - 1;
  OPENPARF_DISPATCH_FLOATING_TYPES(pos_xyz, "ChainLegalizerForward", [&] {
    DispatchedChainLegalizerForward(placedb,
                                    num_concerned_insts,
                                    OPENPARF_TENSOR_DATA_PTR(pos_xyz, scalar_t),
                                    OPENPARF_TENSOR_DATA_PTR(concerned_inst_ids, int32_t),
                                    area_type_id,
                                    search_manh_dist_increment,
                                    max_iter,
                                    num_chains,
                                    OPENPARF_TENSOR_DATA_PTR(chain_cla_ids_bs, int32_t),
                                    OPENPARF_TENSOR_DATA_PTR(chain_cla_ids_b_starts, int32_t),
                                    OPENPARF_TENSOR_DATA_PTR(chain_lut_ids_bs, int32_t),
                                    OPENPARF_TENSOR_DATA_PTR(chain_lut_ids_b_starts, int32_t));
  });
}

#define REGISTER_KERNEL_LAUNCHER(T)                                                                                    \
  template void DispatchedChainLegalizerForward(database::PlaceDB const &placedb,                                      \
                                                int32_t                  num_concerned_insts,                          \
                                                T *                      pos_xyz,                                      \
                                                int32_t *                concerned_inst_ids,                           \
                                                int32_t                  area_type_id,                                 \
                                                int32_t                  search_manh_dist_increment,                   \
                                                int32_t                  max_iter,                                     \
                                                int32_t                  num_chains,                                   \
                                                int32_t *                chain_cla_ids_bs,                             \
                                                int32_t *                chain_cla_ids_b_starts,                       \
                                                int32_t *                chain_lut_ids_bs,                             \
                                                int32_t *                chain_lut_ids_b_starts);

REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

#undef REGISTER_KERNEL_LAUNCHER

}   // namespace chain_legalizer

OPENPARF_END_NAMESPACE
