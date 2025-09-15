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

    double getEntry(size_t i, size_t j) const
    {
        if (static_cast<int>(i) >= mat.m || static_cast<int>(j) >= mat.n) {
            throw std::out_of_range("BlasfeoMat::getEntry: index (" + std::to_string(i) + ", " + std::to_string(j) +
                                    ") out of bounds for matrix of size (" + std::to_string(mat.m) + ", " +
                                    std::to_string(mat.n) + ")");
        }
        return BLASFEO_DMATEL(&mat, i, j);
    }

    void setEntry(size_t i, size_t j, double val)
    {
        if (static_cast<int>(i) >= mat.m || static_cast<int>(j) >= mat.n) {
            throw std::out_of_range("BlasfeoMat::getEntry: index (" + std::to_string(i) + ", " + std::to_string(j) +
                                    ") out of bounds for matrix of size (" + std::to_string(mat.m) + ", " +
                                    std::to_string(mat.n) + ")");
        }
        BLASFEO_DMATEL(&mat, i, j) = val;
    }

    void load(Mat<double>& A) const
    {
        assert(A.rows() == rows() && A.cols() == cols() && "size mismatch");
        for (int r = 0; r < rows(); r++)
            for (int c = 0; c < cols(); c++)
                A(r, c) = BLASFEO_DMATEL(&this->mat, r, c);
    }

    void assign(const Mat<double>& A)
    {
        assert(A.rows() == rows() && A.cols() == cols() && "size mismatch");
        for (int r = 0; r < rows(); r++)
            for (int c = 0; c < cols(); c++)
                BLASFEO_DMATEL(&this->mat, r, c) = A(r, c);
    }

    bool hasNan() const {
        for (int j = 0; j < mat.n; ++j) {
            for (int i = 0; i < mat.m; ++i) {
                if (std::isnan(BLASFEO_DMATEL(&mat, i, j))) {
                    // std::cerr << "BlasfeoMat contains NaN at (" + std::to_string(i) + ", " + std::to_string(j) + ")\n";
                    return true;
                }
            }
        }
        return false;
    }

    bool hasInf() const {
        for (int j = 0; j < mat.n; ++j) {
            for (int i = 0; i < mat.m; ++i) {
                if (std::isinf(BLASFEO_DMATEL(&mat, i, j))) {
                    // std::cerr << "BlasfeoMat contains Inf at (" + std::to_string(i) + ", " + std::to_string(j) + ")\n";
                    return true;
                }
            }
        }
        return false;
    }
};

} // namespace piqp

#endif //PIQP_BLASFEO_MAT_HPP
