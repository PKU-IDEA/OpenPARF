/**
 * File              : extensible_enum.cpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 05.13.2020
 * Last Modified Date: 05.14.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#include "util/extensible_enum.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

//std::ostream &operator<<(std::ostream &os, ExtensibleEnum const &rhs) {
//  os << className<decltype(rhs)>() << "(";
//  os << "name2index_map_ : {";
//  const char *delimiter = "";
//  for (auto const &kvp : rhs.name2index_map_) {
//    os << delimiter << "(" << kvp.first << ", " << kvp.second << ")";
//    delimiter = ", ";
//  }
//  os << "}";
//  os << ", index2name_map_ : {";
//  delimiter = "";
//  for (auto const &kvp : rhs.index2name_map_) {
//    os << delimiter << "(" << kvp.first << ", " << kvp.second << ")";
//    delimiter = ", ";
//  }
//  os << "}";
//  os << "alias_map_ : {";
//  delimiter = "";
//  for (auto const &kvp : rhs.alias_map_) {
//    os << delimiter << "(" << kvp.first << ", " << kvp.second << ")";
//    delimiter = ", ";
//  }
//  os << "}";
//  os << ", initial_size_ : " << rhs.initial_size_;
//  os << ")";
//  return os;
//}

OPENPARF_END_NAMESPACE
