// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_BLASFEO_VEC_HPP
#define PIQP_BLASFEO_VEC_HPP

#include <cstring>

#include "blasfeo.h"

namespace piqp
{

class BlasfeoVec
{
protected:
    blasfeo_dvec vec{}; // note that {} initializes all values to zero here

public:
    BlasfeoVec() = default;

    explicit BlasfeoVec(int m)
    {
        resize(m);
    }

    BlasfeoVec(BlasfeoVec&& other) noexcept
    {
        this->vec = other.vec;
        other.vec.mem = nullptr;
        other.vec.m = 0;
    }

    BlasfeoVec(const BlasfeoVec& other)
    {
        if (other.vec.mem) {
            this->resize(other.rows());
            // y <= x
            blasfeo_dveccp(other.rows(), const_cast<BlasfeoVec&>(other).ref(), 0, this->ref(), 0);
        }
    }

    BlasfeoVec& operator=(BlasfeoVec&& other) noexcept
    {
        this->vec = other.vec;
        other.vec.mem = nullptr;
        other.vec.m = 0;
        return *this;
    }

    BlasfeoVec& operator=(const BlasfeoVec& other)
    {
        if (other.vec.mem) {
            this->resize(other.rows());
            // y <= x
            blasfeo_dveccp(other.rows(), const_cast<BlasfeoVec&>(other).ref(), 0, this->ref(), 0);
        } else {
            if (vec.mem) {
                blasfeo_free_dvec(&vec);
                vec.mem = nullptr;
            }
            vec.m = 0;
        }
        return *this;
    }

    ~BlasfeoVec()
    {
        if (vec.mem) {
            blasfeo_free_dvec(&vec);
        }
    }

    int rows() const { return vec.m; }

    void resize(int m)
    {
        // reuse memory
        if (this->rows() == m) return;

        if (vec.mem) {
            blasfeo_free_dvec(&vec);
        }

        if (m == 0) {
            vec.mem = nullptr;
            vec.m = 0;
            return;
        }

        blasfeo_allocate_dvec(m, &vec);
        // make sure we don't have corrupted memory
        // which can result in massive slowdowns
        // https://github.com/giaf/blasfeo/issues/103
        setZero();
    }

    void setZero() const
    {
        if (vec.mem) {
            // zero out vector
            std::memset(vec.mem, 0, static_cast<std::size_t>(vec.memsize));
        }
    }

    void setConstant(double c)
    {
        blasfeo_dvecse(this->rows(), c, &vec, 0);
    }

    blasfeo_dvec* ref() { return &vec; }

    void print()
    {
        blasfeo_print_dvec(rows(), ref(), 0);
    }
};

} // namespace piqp

#endif //PIQP_BLASFEO_VEC_HPP
