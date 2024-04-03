// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_UTILS_IO_UTILS_HPP
#define PIQP_UTILS_IO_UTILS_HPP

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "piqp/utils/eigen_matio.hpp"
#include "piqp/dense/model.hpp"
#include "piqp/sparse/model.hpp"

namespace piqp
{

template<typename T>
void save_dense_model(const dense::Model<T>& model, const std::string& path)
{
    Eigen::MatioFile file(path.c_str());
    file.write_mat("P", model.P);
    file.write_mat("c", model.c);
    file.write_mat("A", model.A);
    file.write_mat("b", model.b);
    file.write_mat("G", model.G);
    file.write_mat("h", model.h);
    file.write_mat("x_lb", model.x_lb);
    file.write_mat("x_ub", model.x_ub);
    file.close();
}

template<typename T, typename I>
void save_sparse_model(const sparse::Model<T, I>& model, const std::string& path)
{
    Eigen::MatioFile file(path.c_str());
    file.write_mat("P", model.P);
    file.write_mat("c", model.c);
    file.write_mat("A", model.A);
    file.write_mat("b", model.b);
    file.write_mat("G", model.G);
    file.write_mat("h", model.h);
    file.write_mat("x_lb", model.x_lb);
    file.write_mat("x_ub", model.x_ub);
    file.close();
}

template<typename T>
dense::Model<T> load_dense_model(const std::string& path)
{
    Mat<T> P, A, G;
    Vec<T> c, b, h, x_lb, x_ub;
    Eigen::MatioFile file(path.c_str());
    file.read_mat("P", P);
    file.read_mat("c", c);
    file.read_mat("A", A);
    file.read_mat("b", b);
    file.read_mat("G", G);
    file.read_mat("h", h);
    file.read_mat("x_lb", x_lb);
    file.read_mat("x_ub", x_ub);
    file.close();

    dense::Model<T> model(P, c, A, b, G, h, x_lb, x_ub);
    return model;
}

template<typename T, typename I>
sparse::Model<T, I> load_sparse_model(const std::string& path)
{
    SparseMat<T, I> P, A, G;
    Vec<T> c, b, h, x_lb, x_ub;
    Eigen::MatioFile file(path.c_str());
    file.read_mat("P", P);
    file.read_mat("c", c);
    file.read_mat("A", A);
    file.read_mat("b", b);
    file.read_mat("G", G);
    file.read_mat("h", h);
    file.read_mat("x_lb", x_lb);
    file.read_mat("x_ub", x_ub);
    file.close();

    sparse::Model<T, I> model(P, c, A, b, G, h, x_lb, x_ub);
    return model;
}

} // namespace piqp

#endif //PIQP_UTILS_IO_UTILS_HPP
