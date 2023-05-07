/**
 * @file   object.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "database/object.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

void Object::copy(Object const &rhs) { id_ = rhs.id_; }
void Object::move(Object &&rhs) {
  id_ = std::exchange(rhs.id_, std::numeric_limits<IndexType>::max());
}

} // namespace database

OPENPARF_END_NAMESPACE
