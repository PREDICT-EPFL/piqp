// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_BLASFEO_MAT_HPP
#define PIQP_BLASFEO_MAT_HPP

#include <cstring>

#include "blasfeo.h"

namespace piqp
{

class BlasfeoMat
{
protected:
    blasfeo_dmat mat{}; // note that {} initializes all values to zero here

public:
    BlasfeoMat() = default;

    BlasfeoMat(int m, int n)
    {
        resize(m, n);
    }

    BlasfeoMat(BlasfeoMat&& other) noexcept
    {
        this->mat = other.mat;
        other.mat.mem = nullptr;
        other.mat.m = 0;
        other.mat.n = 0;
    }

    BlasfeoMat(const BlasfeoMat& other)
    {
        if (other.mat.mem) {
            this->resize(other.rows(), other.cols());
            // B <= A
            blasfeo_dgecp(other.rows(), other.cols(), const_cast<BlasfeoMat&>(other).ref(), 0, 0, this->ref(), 0, 0);
        }
    }

    BlasfeoMat& operator=(BlasfeoMat&& other) noexcept
    {
        this->mat = other.mat;
        other.mat.mem = nullptr;
        other.mat.m = 0;
        other.mat.n = 0;
        return *this;
    }

    BlasfeoMat& operator=(const BlasfeoMat& other)
    {
        if (other.mat.mem) {
            this->resize(other.rows(), other.cols());
            // B <= A
            blasfeo_dgecp(other.rows(), other.cols(), const_cast<BlasfeoMat&>(other).ref(), 0, 0, this->ref(), 0, 0);
        } else {
            if (mat.mem) {
                blasfeo_free_dmat(&mat);
                mat.mem = nullptr;
            }
            mat.m = 0;
            mat.n = 0;
        }
        return *this;
    }

    ~BlasfeoMat()
    {
        if (mat.mem) {
            blasfeo_free_dmat(&mat);
        }
    }

    int rows() const { return mat.m; }

    int cols() const { return mat.n; }

    void resize(int m, int n)
    {
        // reuse memory
        if (this->rows() == m && this->cols() == n) return;

        if (mat.mem) {
            blasfeo_free_dmat(&mat);
        }

        if (m == 0 || n == 0) {
            mat.mem = nullptr;
            mat.m = m;
            mat.n = n;
            return;
        }

        blasfeo_allocate_dmat(m, n, &mat);
        // make sure we don't have corrupted memory
        // which can result in massive slowdowns
        // https://github.com/giaf/blasfeo/issues/103
        setZero();
    }

    void setZero() const
    {
        if (mat.mem) {
            // zero out matrix
            std::memset(mat.mem, 0, static_cast<std::size_t>(mat.memsize));
        }
    }

    blasfeo_dmat* ref() { return &mat; }

    void print()
    {
        blasfeo_print_dmat(rows(), cols(), ref(), 0, 0);
    }
};

} // namespace piqp

#endif //PIQP_BLASFEO_MAT_HPP
