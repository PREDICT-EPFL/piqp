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
    file.write_mat("h_l", model.h_l);
    file.write_mat("h_u", model.h_u);
    file.write_mat("x_l", model.x_l);
    file.write_mat("x_u", model.x_u);
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
    file.write_mat("h_l", model.h_l);
    file.write_mat("h_u", model.h_u);
    file.write_mat("x_l", model.x_l);
    file.write_mat("x_u", model.x_u);
    file.close();
}

template<typename T>
dense::Model<T> load_dense_model(const std::string& path)
{
    Mat<T> P, A, G;
    Vec<T> c, b, h_l, h_u, x_l, x_u;
    Eigen::MatioFile file(path.c_str());
    file.read_mat("P", P);
    file.read_mat("c", c);
    file.read_mat("A", A);
    file.read_mat("b", b);
    file.read_mat("G", G);
    file.read_mat("h_l", h_l);
    file.read_mat("h_u", h_u);
    file.read_mat("x_l", x_l);
    file.read_mat("x_u", x_u);
    file.close();

    dense::Model<T> model(P, c, A, b, G, h_l, h_u, x_l, x_u);
    return model;
}

template<typename T, typename I>
sparse::Model<T, I> load_sparse_model(const std::string& path)
{
    SparseMat<T, I> P, A, G;
    Vec<T> c, b, h_l, h_u, x_l, x_u;
    Eigen::MatioFile file(path.c_str());
    file.read_mat("P", P);
    file.read_mat("c", c);
    file.read_mat("A", A);
    file.read_mat("b", b);
    file.read_mat("G", G);
    file.read_mat("h_l", h_l);
    file.read_mat("h_u", h_u);
    file.read_mat("x_l", x_l);
    file.read_mat("x_u", x_u);
    file.close();

    sparse::Model<T, I> model(P, c, A, b, G, h_l, h_u, x_l, x_u);
    return model;
}

} // namespace piqp

#endif //PIQP_UTILS_IO_UTILS_HPP
