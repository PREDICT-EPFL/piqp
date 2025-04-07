// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#define GHC_WITH_EXCEPTIONS
#include "piqp/utils/filesystem.hpp"
#include "piqp/utils/io_utils.hpp"
#include "piqp/utils/random_utils.hpp"

#include "gtest/gtest.h"
#include "utils.hpp"

using namespace piqp;

using T = double;
using I = int;

TEST(IOUtils, DenseImportExportTest)
{
    isize dim = 10;
    isize n_eq = 8;
    isize n_ineq = 9;

    dense::Model<T> qp_model = rand::dense_strongly_convex_qp<T>(dim, n_eq, n_ineq);

    fs::path path = fs::temp_directory_path() / "dense_qp_model.mat";
    save_dense_model(qp_model, path.string());

    dense::Model<T> loaded_qp_model = load_dense_model<T>(path.string());

    ASSERT_EQ(qp_model.P, loaded_qp_model.P);
    ASSERT_EQ(qp_model.A, loaded_qp_model.A);
    ASSERT_EQ(qp_model.G, loaded_qp_model.G);
    ASSERT_EQ(qp_model.c, loaded_qp_model.c);
    ASSERT_EQ(qp_model.b, loaded_qp_model.b);
    ASSERT_EQ(qp_model.h_l, loaded_qp_model.h_l);
    ASSERT_EQ(qp_model.h_u, loaded_qp_model.h_u);
    ASSERT_EQ(qp_model.x_l, loaded_qp_model.x_l);
    ASSERT_EQ(qp_model.x_u, loaded_qp_model.x_u);
}

TEST(IOUtils, SparseImportExportTest)
{
    isize dim = 10;
    isize n_eq = 8;
    isize n_ineq = 9;
    T sparsity_factor = 0.2;

    sparse::Model<T, I> qp_model = rand::sparse_strongly_convex_qp<T, I>(dim, n_eq, n_ineq, sparsity_factor);

    fs::path path = fs::temp_directory_path() / "sparse_qp_model.mat";
    save_sparse_model(qp_model, path.string());

    sparse::Model<T, I> loaded_qp_model = load_sparse_model<T, I>(path.string());

    assert_sparse_matrices_equal(qp_model.P, loaded_qp_model.P);
    assert_sparse_matrices_equal(qp_model.A, loaded_qp_model.A);
    assert_sparse_matrices_equal(qp_model.G, loaded_qp_model.G);
    ASSERT_EQ(qp_model.c, loaded_qp_model.c);
    ASSERT_EQ(qp_model.b, loaded_qp_model.b);
    ASSERT_EQ(qp_model.h_l, loaded_qp_model.h_l);
    ASSERT_EQ(qp_model.h_u, loaded_qp_model.h_u);
    ASSERT_EQ(qp_model.x_l, loaded_qp_model.x_l);
    ASSERT_EQ(qp_model.x_u, loaded_qp_model.x_u);
}
