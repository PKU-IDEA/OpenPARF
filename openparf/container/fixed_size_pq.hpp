/**
 * File              : fixed_size_pq.h
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 06.03.2020
 * Last Modified Date: 06.03.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */
#pragma once

// C++ standard library headers
#include <set>

// project headers
#include "util/namespace.h"

OPENPARF_BEGIN_NAMESPACE

namespace container {
/// Fixed size priority queue
template<typename T, typename Comp>
class FixedSizePQ {
  private:
  // using SetType = typename boost::container::flat_set<T, Comp>;
  using SetType = typename std::set<T, Comp>;

  public:
  using iterator       = typename SetType::iterator;
  using const_iterator = typename SetType::const_iterator;

  public:
  explicit FixedSizePQ(std::size_t n) : _n(n) {}

  bool push(const T& item) {
    if (full() && _comp(bottom(), item)) {
      return false;
    }

    auto ret = _pq.insert(item);
    if (size() > _n) {
      _pq.erase(std::prev(_pq.end()));
    }
    return ret.second;
  }

  void           clear() { _pq.clear(); }
  void           pop() { _pq.erase(_pq.begin()); }
  bool           empty() const { return _pq.empty(); }
  bool           full() const { return _pq.size() == _n; }
  std::size_t    size() const { return _pq.size(); }
  const T&       top() const { return *_pq.begin(); }
  const T&       bottom() const { return *_pq.rbegin(); }
  iterator       begin() { return _pq.begin(); }
  const_iterator begin() const { return _pq.begin(); }
  iterator       end() { return _pq.end(); }
  const_iterator end() const { return _pq.end(); }
  const T&       operator[](std::size_t idx) const {
    auto it = _pq.begin();
    std::advance(it, idx);
    return *it;
  }
  iterator erase(const_iterator it) { return _pq.erase(it); }

  template<typename B>
  bool contain(const B& b) const {
    for (const T& e : _pq) {
      if (b(e)) {
        return true;
      }
    }
    return false;
  }

  private:
  SetType     _pq;
  Comp        _comp;
  std::size_t _n;   // The fixed size of this PQ
};
}   // namespace container
OPENPARF_END_NAMESPACE
