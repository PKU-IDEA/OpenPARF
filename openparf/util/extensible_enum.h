/**
 * File              : extensible_enum.h
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 05.14.2020
 * Last Modified Date: 05.14.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#ifndef OPENPARF_UTIL_EXTENSIBLE_ENUM_H_
#define OPENPARF_UTIL_EXTENSIBLE_ENUM_H_

#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>
#include "util/message.h"

OPENPARF_BEGIN_NAMESPACE

///// @brief Extensible enum types
//class ExtensibleEnum {
// public:
//  using IndexType = uint8_t;
//  using NameMapType = std::unordered_map<std::string, IndexType>;
//  using IndexMapType = std::vector<std::pair<IndexType, std::string>>;
//  using AliasMapType = std::unordered_map<std::string, std::string>;
//
//  /// @brief default constructor
//  ExtensibleEnum() { initial_size_ = 0; }
//
//  /// @brief constructor with initial map
//  ExtensibleEnum(NameMapType const &initial_map, AliasMapType const &alias_map)
//      : name2index_map_(initial_map), alias_map_(alias_map) {
//    index2name_map_.resize(name2index_map_.size());
//    for (auto const &kvp : name2index_map_) {
//      index2name_map_[kvp.second] = std::make_pair(kvp.second, kvp.first);
//    }
//    initial_size_ = index2name_map_.size();
//  }
//
//  /// @brief number of enum elements
//  IndexType size() const { return index2name_map_.size(); }
//
//  /// @brief number of initial elements
//  IndexType sizeInitial() const { return initial_size_; }
//
//  /// @brief number of extended elements
//  IndexType sizeExt() const { return size() - sizeInitial(); }
//
//  /// @brief get index from name
//  IndexType id(std::string const &n) const { return find(n); }
//
//  /// @grief get name from index
//  std::string const &name(IndexType i) const {
//    openparfAssert(i < size());
//    return index2name_map_[i].second;
//  }
//
//  /// @brief get the alias name if exists
//  std::string alias(std::string const &n) const {
//    auto found = alias_map_.find(n);
//    if (found == alias_map_.end()) {  // not found
//      return n;
//    } else {
//      return found->second;
//    }
//  }
//
//  /// @brief try to extend the elements
//  std::pair<IndexType, bool> tryExtend(std::string const &n) {
//    auto found = find(n);
//    if (found == std::numeric_limits<IndexType>::max()) {  // not exist
//      IndexType id = index2name_map_.size();
//      name2index_map_.emplace(n, id);
//      index2name_map_.emplace_back(id, n);
//      return std::make_pair(id, true);
//    } else {  // already exist
//      return std::make_pair(found, false);
//    }
//  }
//
//  /// @brief iterator for traversing
//  IndexMapType::const_iterator begin() const { return index2name_map_.begin(); }
//
//  /// @brief iterator for traversing
//  IndexMapType::const_iterator end() const { return index2name_map_.end(); }
//
//  /// @brief iterator for traversing the extended elements
//  IndexMapType::const_iterator beginExt() const {
//    return index2name_map_.begin() + initial_size_;
//  }
//
//  /// @brief iterator for traversing the extended elements
//  IndexMapType::const_iterator endExt() const { return end(); }
//
//  /// @brief iterator for traversing the initial elements
//  IndexMapType::const_iterator beginInitial() const {
//    return index2name_map_.begin();
//  }
//
//  /// @brief iterator for traversing the extended elements
//  IndexMapType::const_iterator endInitial() const {
//    return index2name_map_.begin() + initial_size_;
//  }
//
// protected:
//  /// @brief find index from name
//  IndexType find(std::string const &n) const {
//    auto found1 = name2index_map_.find(n);
//    if (found1 == name2index_map_.end()) {  // not found
//      auto found2 = alias_map_.find(n);
//      if (found2 == alias_map_.end()) {  // not even found in alias map
//        return std::numeric_limits<IndexType>::max();
//      } else {  // found in alias map
//        found1 = name2index_map_.find(found2->second);
//        openparfAssertMsg(
//            found1 != name2index_map_.end(),
//            "alias map entry {%s, %s} not found in name2index_map_", n.c_str(),
//            found2->second.c_str());
//      }
//    }
//    return found1->second;
//  }
//
//  /// @brief overload output stream
//  friend std::ostream &operator<<(std::ostream &os, ExtensibleEnum const &rhs);
//
//  NameMapType name2index_map_;   ///< map resource type name to index
//  IndexMapType index2name_map_;  ///< resource type names
//  AliasMapType
//      alias_map_;  ///< sometimes the names do not match exactly, create alias
//
//  IndexType initial_size_;  ///< number of initial elements
//};
//
//////////////////////////////////////////////////////////////////////////
//// This is rather a hard-coded way to map resource type to area type
//// Here I list typical settings in FPGA
//// SITEs: SLICEL, SLICEM, DSP, RAM, IO
//// RESOURCEs: LUTL, LUTM, FF, CARRY, DSP, RAM, IO
//// The problem comes from LUTL and LUTM.
//// SLICEL often contains {LUTL, FF, CARRY};
//// SLICEM often contains {LUTM, FF, CARRY}.
//// LUTL can host LUT1-6, while
//// LUTM can host {LUT1-6, LRAM, SHIFT}, which makes things complicated.
//// Considering that SLICEL and SLICEM usually appear in an interleaved way,
//// e.g., 68 SLICELs and 60 SLICEMs, I regard them as one kind of SLICE.
//// This means that there are only one area type called LUT.
//// The areas for LRAM and SHIFT are increased as LUTL cannot host them.
//////////////////////////////////////////////////////////////////////////
//
///// @brief extensible resource types;
///// The reason I would like to use a custom type is that
///// single-site blocks like DSP/RAM may have many customized variations.
///// It is difficult to use enum types.
//class ResourceTypeEnum : public ExtensibleEnum {
// public:
//  using BaseType = ExtensibleEnum;
//
//  /// @brief default constructor
//  /// with basic types defined.
//  ResourceTypeEnum()
//      : BaseType(
//            // name2index_map_
//            {
//
//                {"LUTL", 0},   ///< logic LUT
//                {"LUTM", 1},   ///< LUT with memory control block
//                {"FF", 2},     ///< flip-flops
//                {"Carry", 3},  ///< carry that is inside LUT
//                {"IO", 4}      ///< IO
//            },
//            // alias_map_
//            {
//                {"LUT", "LUTL"},  // LUT is an alias of LUTL
//                {"CARRY8", "Carry"},
//                {"CLA4", "Carry"},  // alias for Carry
//                {"DFF", "FF"}       // alias for flip-flip
//            }) {}
//};
//
///// @brief area type
///// Area type is different from resource type;
///// resource type is from the benchmarks of resources;
///// area type is for global placement to build the bin maps.
//class AreaTypeEnum : public ExtensibleEnum {
// public:
//  using BaseType = ExtensibleEnum;
//  using CoordinateType = float;
//
//  /// @brief default constructor
//  /// with basic types defined
//  AreaTypeEnum()
//      : BaseType(
//            // name2index_map_
//            {
//                {"LUT", 0},     ///< logic LUT in LUTL and LUTM
//                {"LUTMCB", 1},  ///< memory control block, LUT + LUTMCB = LUTM
//                {"FF", 2},      ///< flip-flops
//                {"Carry", 3},   ///< carry inside a LUT sites, separate resource
//                {"IO", 4}       ///< IO
//            },
//            // alias_map_
//            {}) {}
//
//  /// @brief initialize the extended area types from resouce types
//  void initialize(ResourceTypeEnum const &rhs) {
//    for (auto it = rhs.beginExt(); it != rhs.endExt(); ++it) {
//      tryExtend(it->second);
//    }
//  }
//
//  /// @brief map resource to area types
//  /// One resource may map to multiple area types
//  std::vector<std::string> resourceAreaTypes(std::string const &n) const {
//    if (n == "LUTL") {
//      return {"LUT"};
//    } else if (n == "LUTM") {
//      return {"LUT", "LUTMCB"};
//    } else {
//      return {n};
//    }
//  }
//
//  /// @brief map resource to area types
//  /// One resource may map to multiple area types.
//  std::vector<IndexType> resourceAreaTypeIds(std::string const &n) const {
//    auto area_types = resourceAreaTypes(n);
//    std::vector<IndexType> area_type_ids;
//    for (auto const &v : area_types) {
//      area_type_ids.push_back(id(v));
//    }
//
//    return area_type_ids;
//  }
//};

OPENPARF_END_NAMESPACE

#endif
