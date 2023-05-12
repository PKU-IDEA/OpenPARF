#ifndef OPENPARF_INDEX_WRAPPER_H
#define OPENPARF_INDEX_WRAPPER_H

#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

namespace container {

/*
 * TODO(Jing Mai, magic3007@pku.edu.cn): Replace `NullValue` by `std::optional` if C++17 is allowed.
 */
template<typename T, typename IndexType>
class IndexWrapper {
public:
    using RefType  = std::reference_wrapper<T>;
    using SelfType = IndexWrapper<T, IndexType>;
    static T NullValue;

    // default constructor.
    IndexWrapper() = default;

    explicit IndexWrapper(RefType const &ref) : ref_(ref) {}

    // Copy constructor.
    IndexWrapper(SelfType const &rhs) { copy(rhs); }

    // Move constructor.
    IndexWrapper(SelfType &&rhs) noexcept { move(rhs); }

    // Copy assignment operator.
    IndexWrapper &operator=(SelfType const &rhs) {
        if (this != &rhs) {
            copy(rhs);
        }
        return *this;
    }

    // Move assignment operator.
    IndexWrapper &operator=(SelfType &&rhs) noexcept {
        if (this != &rhs) { move(rhs); }
        return *this;
    }

    virtual ~IndexWrapper() = default;

    void setId(IndexType const &id) { id_ = id; }

    void setRef(RefType const &ref) { ref_ = ref; }

    IndexType id() const { return id_; }

    RefType ref() const { return ref_; }

    bool isNullRef() const { return &ref_.get() == &NullValue; }

protected:
    void copy(SelfType const &rhs) {
        ref_ = rhs.ref_;
        id_  = rhs.id_;
    }
    void move(SelfType const &rhs) {
        ref_ = std::move(rhs.ref_);
        id_  = std::move(rhs.id_);
    }

    RefType   ref_ = std::ref(NullValue);
    IndexType id_;
};

// Definition of static member in template class.
template<typename T, typename IndexType>
T IndexWrapper<T, IndexType>::NullValue;

}   // namespace container

OPENPARF_END_NAMESPACE

#endif   //OPENPARF_INDEX_WRAPPER_H
