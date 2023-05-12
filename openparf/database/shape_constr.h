/**
 * @file   shape_constr.h
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_DATABASE_SHAPE_CONSTR_H_
#define OPENPARF_DATABASE_SHAPE_CONSTR_H_

#include "database/object.h"
#include "geometry/geometry.hpp"

OPENPARF_BEGIN_NAMESPACE

namespace database {

/// @brief Class for shape constraint in placement.
/// A shape constraint requires that a set of cells must be placed in a certain
/// pattern. Typically the pattern is a column with width of 1
class ShapeConstr : public Object {
 public:
  using BaseType  = Object;
  using PointType = geometry::Point<IndexType, 2>;

  class ShapeElement : public Object {
   public:
    /// @brief getter for offset
    PointType const &offset() const { return offset_; }
    /// @brief setter for offset
    void             setOffset(PointType const &o) { offset_ = o; }
    /// @brief getter for instance id
    IndexType        instId() const { return inst_id_; }
    /// @brief setter for instance id
    void             setInstId(IndexType i) { inst_id_ = i; }

   protected:
    /// @brief overload output stream
    friend std::ostream &operator<<(std::ostream &os, ShapeElement const &rhs);

    PointType            offset_;    ///< The x, y of this element. A relative position to the
                                     ///< first node
    IndexType            inst_id_;   ///< instance index
  };

  /// @brief default constructor
  ShapeConstr() : BaseType(), type_(ShapeConstrType::kUnknown) {}

  /// @brief constructor
  ShapeConstr(IndexType id) : BaseType(id), type_(ShapeConstrType::kUnknown) {}

  /// @brief copy constructor
  ShapeConstr(ShapeConstr const &rhs) { copy(rhs); }

  /// @brief move constructor
  ShapeConstr(ShapeConstr &&rhs) noexcept { move(std::move(rhs)); }

  /// @brief copy assignment
  ShapeConstr &operator=(ShapeConstr const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move constructor
  ShapeConstr &operator=(ShapeConstr &&rhs) noexcept {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

  /// @brief getter for type
  ShapeConstrType                  type() const { return type_; }

  /// @brief setter for type
  void                             setType(ShapeConstrType const &v) { type_ = v; }

  /// @brief number of elements
  IndexType                        numShapeElements() const { return elements_.size(); }

  /// @brief getter for element
  ShapeElement const &             shapeElement(IndexType i) const { return elements_[i]; }

  /// @brief getter for element
  ShapeElement &                   shapeElement(IndexType i) { return elements_[i]; }

  /// @brief getter for elements
  std::vector<ShapeElement> const &shapeElements() const { return elements_; }

  /// @brief add a shape element
  ShapeElement &                   addShapeElement(PointType const &offset, IndexType inst_id);

  /// @brief summarize memory usage of the object in bytes
  IndexType                        memory() const;

 protected:
  /// @brief copy object
  void                      copy(ShapeConstr const &rhs);
  /// @brief move object
  void                      move(ShapeConstr &&rhs);
  /// @brief overload output stream
  friend std::ostream &     operator<<(std::ostream &os, ShapeConstr const &rhs);

  ShapeConstrType           type_;       ///< type of the shape constraint
  std::vector<ShapeElement> elements_;   ///< all shape elements of this constraint
};

}   // namespace database

OPENPARF_END_NAMESPACE

#endif
