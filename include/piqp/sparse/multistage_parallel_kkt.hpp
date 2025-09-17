#ifndef PIQP_MULTISTAGE_PARALLEL_KKT_H
#define PIQP_MULTISTAGE_PARALLEL_KKT_H

#define KKT_SOLVE_NUM_THREADS 4

#include "piqp/sparse/multistage_kkt.hpp"
#include "piqp/sparse/blocksparse/block_kkt_parallel.hpp"
#ifdef PIQP_HAS_OPENMP
#include "omp.h"
#endif


namespace piqp
{

namespace sparse
{
    template<typename T, typename I>
    class MultistageParallelKKT : public MultistageKKT<T, I>
    {

    protected:
        // For parallel factorization
        size_t kkt_solve_num_threads = KKT_SOLVE_NUM_THREADS;
        BlockKKTParallel kkt_fac_parallel;
        std::vector<size_t> pivots;
        std::vector<std::vector<size_t>> segments;

    public:
        explicit MultistageParallelKKT(const Data<T, I>& data)
            : MultistageKKT<T, I>(data) {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::constructor");

            init();
        }

        void init() {
            if (this->block_info.size() - 1 < 3 * kkt_solve_num_threads) {
                throw std::runtime_error("Cannot use parallel multistage kkt solver. Use serial solver instead.");
            }
            generate_partitions();  // Generate partitions for multi-threads
            init_kkt_fac();

        }


        void generate_partitions() {
            pivots.clear();
            segments.clear();

            const size_t N = this->block_info.size() - 1;
            const size_t P = kkt_solve_num_threads;

            // Compute segment size such that first segment is ~19/7 times others
            std::div_t s_div = std::div(static_cast<I>(7*N), static_cast<I>(7*P + 12));
            const size_t seg_len = s_div.quot > 0 ? static_cast<size_t>(s_div.quot) : 1;  // ensure at least one element per segment

            // Compute pivot indices
            pivots.reserve(P - 1);
            for (size_t i = 0; i < P - 1; ++i) {
                assert(N > 1 + seg_len * (i+1) + i && "pivot index out of bounds");
                pivots.insert(pivots.begin(), N - 1 - seg_len * (i+1) - i);
            }


            // Build segments using pivots
            std::vector<size_t> pivots_tmp = pivots;
            pivots_tmp.insert(pivots_tmp.begin(), static_cast<size_t>(-1));  // underflows to max size_t
            pivots_tmp.back() = std::min(pivots_tmp.back(), N - 1);  // guard overflow
            pivots_tmp.push_back(N);

            for (size_t i = 0; i < pivots_tmp.size() - 1; ++i) {
                std::vector<size_t> segment;
                for (size_t j = pivots_tmp[i] + 1; j < std::min(pivots_tmp[i + 1], N); ++j) {
                    segment.push_back(j);
                }
                segments.push_back(segment);
            }
        }


        void init_kkt_fac() override
        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::init_kkt_fac");
            construct_kkt_fac<true>(this->work_x);
        }

        void populate_kkt_fac(const Vec<T>& x_reg) override
        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::populate_kkt_fac");
            construct_kkt_fac<false>(x_reg);
        }

        template<bool allocate>
        void construct_kkt_fac(const Vec<T>& x_reg) {

            I arrow_width = this->block_info.back().diag_size;  // TODO: consider arrow parts
            T delta_inv = 1.0 / this->m_delta;
            BlockVec& x_reg_block = this->work_x_block_1;
            x_reg_block.assign(x_reg);

            if (allocate) {

                kkt_fac_parallel.sub_blocks.resize(kkt_solve_num_threads);
                kkt_fac_parallel.segments = segments;
                kkt_fac_parallel.pivots = pivots;
                kkt_fac_parallel.num_threads = kkt_solve_num_threads;

                // TODO: parallelize the following loop
                for (size_t k = 0; k < kkt_solve_num_threads; k++) {
                    auto &sub_block = kkt_fac_parallel.sub_blocks[k];
                    sub_block.D.clear();
                    sub_block.D.resize(segments[k].size());
                    sub_block.E.clear();
                    sub_block.E.resize(segments[k].size() - 1);
                    sub_block.B.clear();
                    k > 0 ? sub_block.B.resize(segments[k].size()) : sub_block.B.resize(0);
                }

                for (size_t k = 0; k < kkt_solve_num_threads; k++) {
                    auto& sub_block = kkt_fac_parallel.sub_blocks[k];
                    const auto& segment_k = segments[k];
                    sub_block.index = k;

                    // D
                    for (size_t i = 0; i < segment_k.size(); i++) {
                        I m_D = this->block_info[segment_k[i]].diag_size;
                        sub_block.D[i] = std::make_unique<BlasfeoMat>(m_D, m_D);
                    }

                    // E
                    for (size_t i = 0; i < segment_k.size() - 1; i++) {
                        I m_E = this->block_info[segment_k[i]].off_diag_size;
                        I n_E = this->block_info[segment_k[i]].diag_size;
                        sub_block.E[i] = std::make_unique<BlasfeoMat>(m_E, n_E);
                    }

                    // F
                    if (k < segments.size() - 1) {
                        I m_F = this->block_info[pivots[k]-1].off_diag_size;
                        I n_F = this->block_info[pivots[k]-1].diag_size;
                        sub_block.F = std::make_unique<BlasfeoMat>(m_F, n_F);
                    } else {
                        sub_block.F = nullptr;  // The last sub-block does not have an F matrix
                    }

                    // A
                    if (k > 0) {
                        I m_A = this->block_info[pivots[k-1]].diag_size;
                        sub_block.A = std::make_unique<BlasfeoMat>(m_A, m_A);
                    } else {
                        sub_block.A = nullptr;  // The first sub-block does not have an A matrix
                    }

                    // H
                    if (k > 0 && k < segments.size() - 1) {
                        sub_block.H = std::make_unique<BlasfeoMat>(sub_block.F->rows(), sub_block.A->cols());
                    } else {
                        sub_block.H = nullptr;  // The first and last sub-blocks do not have an H matrix
                    }

                    // B
                    if (k > 0) {
                        sub_block.B[0] = std::make_unique<BlasfeoMat>(sub_block.D[0]->cols(), sub_block.A->rows());
                        for (size_t i = 1; i < segments[k].size(); i++) {
                            sub_block.B[i] = std::make_unique<BlasfeoMat>(sub_block.D[i]->cols(), sub_block.B[i-1]->cols());
                        }
                    }

                    if (arrow_width > 0) {
                        // TODO: consider arrow parts
                    }
                }

            } else {
                // allocate == false

#ifdef PIQP_HAS_OPENMP
#pragma omp for nowait
#endif
                for (size_t k = 0; k < kkt_solve_num_threads; k++) {
                    PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::populate_kkt_fac");
                    PIQP_TRACY_ZoneValue(k);

                    auto& sub_block_k = kkt_fac_parallel.sub_blocks[k];
                    const auto& segment_k = segments[k];

                    // ----- D -----
                    for (size_t i = 0; i < segment_k.size(); i++) {

                        bool D_mat_set = false;
                        if (this->P.D[segment_k[i]]) {
                            // D_i = P.D_i, lower triangular
                            assert(this->P.D[segment_k[i]]->rows() <= sub_block_k.D[i]->rows() && "size mismatch");
                            assert(this->P.D[segment_k[i]]->cols() <= sub_block_k.D[i]->cols() && "size mismatch");
                            blasfeo_dtrcp_l(*this->P.D[segment_k[i]], *sub_block_k.D[i]);
                            D_mat_set = true;
                        }


                        if (this->AtA.D[segment_k[i]]) {
                            assert(this->AtA.D[segment_k[i]]->rows() <= sub_block_k.D[i]->rows() && "size mismatch");
                            assert(this->AtA.D[segment_k[i]]->cols() <= sub_block_k.D[i]->cols() && "size mismatch");
                            if (D_mat_set) {
                                // D_i += delta^{-1} * AtA.D_i
                                blasfeo_dgead(delta_inv, *this->AtA.D[segment_k[i]], *sub_block_k.D[i]); // TODO: is there a blasfeo function that only add the lower triangular part?
                            } else {
                                // D_i = delta^{-1} * AtA.D_i, lower triangular
                                blasfeo_dtrcpsc_l(delta_inv, *this->AtA.D[segment_k[i]], *sub_block_k.D[i]);
                                D_mat_set = true;
                            }
                        }


                        if (this->GtG.D[segment_k[i]]) {
                            assert(this->GtG.D[segment_k[i]]->rows() <= sub_block_k.D[i]->rows() && "size mismatch");
                            assert(this->GtG.D[segment_k[i]]->cols() <= sub_block_k.D[i]->cols() && "size mismatch");
                            if (D_mat_set) {
                                // D_i += GtG.D_i
                                blasfeo_dgead(1.0, *this->GtG.D[segment_k[i]], *sub_block_k.D[i]);
                            } else {
                                // D_i = GtG.D_i, lower triangular
                                blasfeo_dtrcp_l(*this->GtG.D[segment_k[i]], *sub_block_k.D[i]);
                                D_mat_set = true;
                            }
                        }

                        if (D_mat_set) {
                            // diag(D_i) += diag
                            blasfeo_ddiaad(1.0, x_reg_block.x[segment_k[i]], *sub_block_k.D[i]);
                        } else {
                            // D_i = diag
                            sub_block_k.D[i]->setZero();
                            blasfeo_ddiain(1.0, x_reg_block.x[segment_k[i]], *sub_block_k.D[i]);
                        }


                        assert(!sub_block_k.D[i]->hasNan() && "D matrix has NaN values");
                        assert(!sub_block_k.D[i]->hasInf() && "D matrix has Inf values");
                    }

                    // ----- A -----
                    if (k > 0) {
                        bool A_mat_set = false;
                        if (this->P.D[pivots[k-1]]) {
                            assert(this->P.D[pivots[k-1]]->rows() <= sub_block_k.A->rows() && "size mismatch");
                            assert(this->P.D[pivots[k-1]]->cols() <= sub_block_k.A->cols() && "size mismatch");
                            blasfeo_dtrcp_l(*this->P.D[pivots[k-1]], *sub_block_k.A);
                            A_mat_set = true;
                        }

                        if (this->AtA.D[pivots[k-1]]) {
                            assert(this->AtA.D[pivots[k-1]]->rows() <= sub_block_k.A->rows() && "size mismatch");
                            assert(this->AtA.D[pivots[k-1]]->cols() <= sub_block_k.A->cols() && "size mismatch");
                            if (A_mat_set) {
                                blasfeo_dgead(delta_inv, *this->AtA.D[pivots[k-1]], *sub_block_k.A);
                            } else {
                                blasfeo_dtrcpsc_l(delta_inv, *this->AtA.D[pivots[k-1]], *sub_block_k.A);
                                A_mat_set = true;
                            }
                        }

                        if (this->GtG.D[pivots[k-1]]) {
                            assert(this->GtG.D[pivots[k-1]]->rows() <= sub_block_k.A->rows() && "size mismatch");
                            assert(this->GtG.D[pivots[k-1]]->cols() <= sub_block_k.A->cols() && "size mismatch");
                            if (A_mat_set) {
                                blasfeo_dgead(1.0, *this->GtG.D[pivots[k-1]], *sub_block_k.A);
                            } else {
                                blasfeo_dtrcp_l(*this->GtG.D[pivots[k-1]], *sub_block_k.A);
                                A_mat_set = true;
                            }
                        }

                        if (A_mat_set) {
                            // diag(D_i) += diag
                            blasfeo_ddiaad(1.0, x_reg_block.x[pivots[k-1]], *sub_block_k.A);
                        } else {
                            // D_i = diag
                            sub_block_k.A->setZero();
                            blasfeo_ddiain(1.0, x_reg_block.x[pivots[k-1]], *sub_block_k.A);
                        }

                        assert(!sub_block_k.A->hasNan() && "A matrix has NaN values");
                        assert(!sub_block_k.A->hasInf() && "A matrix has Inf values");
                    }


                    // ----- E -----
                    for (size_t i = 0; i < segment_k.size() - 1; i++) {

                        bool E_mat_set = false;
                        if (this->P.B[segment_k[i]]) {
                            assert(this->P.B[segment_k[i]]->rows() <= sub_block_k.E[i]->rows() && "size mismatch");
                            assert(this->P.B[segment_k[i]]->cols() <= sub_block_k.E[i]->cols() && "size mismatch");
                            // E_i = P.B_i
                            blasfeo_dgecp(*this->P.B[segment_k[i]], *sub_block_k.E[i]);
                            E_mat_set = true;
                        }

                        if (this->AtA.B[segment_k[i]]) {
                            assert(this->AtA.B[segment_k[i]]->rows() <= sub_block_k.E[i]->rows() && "size mismatch");
                            assert(this->AtA.B[segment_k[i]]->cols() <= sub_block_k.E[i]->cols() && "size mismatch");
                            if (E_mat_set) {
                                // E_i += delta^{-1} * AtA.B_i
                                blasfeo_dgead(delta_inv, *this->AtA.B[segment_k[i]], *sub_block_k.E[i]); // TODO: is there a blasfeo function that only add the lower triangular part?
                            } else {
                                // E_i = delta^{-1} * AtA.B_i
                                blasfeo_dgecpsc(delta_inv, *this->AtA.B[segment_k[i]], *sub_block_k.E[i]);
                                E_mat_set = true;
                            }
                        }

                        if (this->GtG.B[segment_k[i]]) {
                            assert(this->GtG.B[segment_k[i]]->rows() <= sub_block_k.E[i]->rows() && "size mismatch");
                            assert(this->GtG.B[segment_k[i]]->cols() <= sub_block_k.E[i]->cols() && "size mismatch");
                            if (E_mat_set) {
                                // E_i += GtG.B_i
                                blasfeo_dgead(1.0, *this->GtG.B[segment_k[i]], *sub_block_k.E[i]);
                            } else {
                                // E_i = GtG.B_i
                                blasfeo_dgecp(*this->GtG.B[segment_k[i]], *sub_block_k.E[i]);
                                E_mat_set = true;
                            }
                        }

                        assert(!sub_block_k.E[i]->hasNan() && "E matrix has NaN values");
                        assert(!sub_block_k.E[i]->hasInf() && "E matrix has Inf values");
                    }

                    // ----- F -----
                    if (k < segments.size() - 1) {
                        bool F_mat_set = false;
                        if (this->P.B[pivots[k]-1]) {
                            assert(this->P.B[pivots[k]-1]->rows() <= sub_block_k.F->rows() && "size mismatch");
                            assert(this->P.B[pivots[k]-1]->cols() <= sub_block_k.F->cols() && "size mismatch");
                            blasfeo_dgecp(*this->P.B[pivots[k]-1], *sub_block_k.F);
                            F_mat_set = true;
                        }

                        if (this->AtA.B[pivots[k]-1]) {
                            assert(this->AtA.B[pivots[k]-1]->rows() <= sub_block_k.F->rows() && "size mismatch");
                            assert(this->AtA.B[pivots[k]-1]->cols() <= sub_block_k.F->cols() && "size mismatch");
                            if (F_mat_set) {
                                blasfeo_dgead(delta_inv, *this->AtA.B[pivots[k]-1], *sub_block_k.F);
                            } else {
                                blasfeo_dgecpsc(delta_inv, *this->AtA.B[pivots[k]-1], *sub_block_k.F);
                                F_mat_set = true;
                            }
                        }

                        if (this->GtG.B[pivots[k]-1]) {
                            assert(this->GtG.B[pivots[k]-1]->rows() <= sub_block_k.F->rows() && "size mismatch");
                            assert(this->GtG.B[pivots[k]-1]->cols() <= sub_block_k.F->cols() && "size mismatch");
                            if (F_mat_set) {
                                blasfeo_dgead(1.0, *this->GtG.B[pivots[k]-1], *sub_block_k.F);
                            } else {
                                blasfeo_dgecp(*this->GtG.B[pivots[k]-1], *sub_block_k.F);
                                F_mat_set = true;
                            }
                        }

                        assert(!sub_block_k.F->hasNan() && "F matrix has NaN values");
                        assert(!sub_block_k.F->hasInf() && "F matrix has Inf values");
                    }

                    // ----- H -----
                    if (k > 0 && k < segments.size() - 1) {
                        sub_block_k.H->setZero();
                    }

                    // ----- B -----
                    if (k > 0) {
                        // B_1 - B_end are all zeros. Also set B_0 to zeros hereby.
                        for (size_t i = 0; i < segments[k].size(); i++) {
                            sub_block_k.B[i]->setZero();
                        }

                        // B0
                        bool B0_mat_set = false;
                        if (this->P.B[pivots[k-1]]) {
                            assert(this->P.B[pivots[k-1]]->rows() <= sub_block_k.B[0]->rows() && "size mismatch");
                            assert(this->P.B[pivots[k-1]]->cols() <= sub_block_k.B[0]->cols() && "size mismatch");
                            // B_0 = P.B_
                            blasfeo_dgecp(*this->P.B[pivots[k-1]], *sub_block_k.B[0]);
                            B0_mat_set = true;
                        }

                        if (this->AtA.B[pivots[k-1]]) {
                            assert(this->AtA.B[pivots[k-1]]->rows() <= sub_block_k.B[0]->rows() && "size mismatch");
                            assert(this->AtA.B[pivots[k-1]]->cols() <= sub_block_k.B[0]->cols() && "size mismatch");
                            if (B0_mat_set) {
                                // B_0 += delta^{-1} * AtA.B_
                                blasfeo_dgead(delta_inv, *this->AtA.B[pivots[k-1]], *sub_block_k.B[0]);
                            } else {
                                // B_0 = delta^{-1} * AtA.B_
                                blasfeo_dgecpsc(delta_inv, *this->AtA.B[pivots[k-1]], *sub_block_k.B[0]);
                                B0_mat_set = true;
                            }
                        }

                        if (this->GtG.B[pivots[k-1]]) {
                            assert(this->GtG.B[pivots[k-1]]->rows() <= sub_block_k.B[0]->rows() && "size mismatch");
                            assert(this->GtG.B[pivots[k-1]]->cols() <= sub_block_k.B[0]->cols() && "size mismatch");
                            if (B0_mat_set) {
                                // B_0 += GtG.B_
                                blasfeo_dgead(1.0, *this->GtG.B[pivots[k-1]], *sub_block_k.B[0]);
                            } else {
                                // B_0 = GtG.B_
                                blasfeo_dgecp(*this->GtG.B[pivots[k-1]], *sub_block_k.B[0]);
                                B0_mat_set = true;
                            }
                        }

                        assert(!sub_block_k.B[0]->hasNan() && "B matrix has NaN values");
                        assert(!sub_block_k.B[0]->hasInf() && "B matrix has Inf values");
                    }
                }
            }  // end of allocate if-else

        }


        bool update_scalings_and_factor(const Data<T, I>&, const T& delta, const Vec<T>& x_reg, const Vec<T>& z_reg) override
        {
            this->m_delta = delta;
            this->m_z_reg_inv.array() = z_reg.array().inverse();

            // populate G scaling vector
            Eigen::Index i = 0;
            for (I block_idx = 0; block_idx < this->GT.block_row_sizes.rows(); ++block_idx)
            {
                I block_size = this->GT.block_row_sizes(block_idx);
                for (I inner_idx = 0; inner_idx < block_size; ++inner_idx)
                {
                    I perm_idx = this->GT.perm_inv(i);
                    BLASFEO_DVECEL(this->G_scaling.x[static_cast<std::size_t>(block_idx)].ref(), inner_idx) = std::sqrt(this->m_z_reg_inv(perm_idx));
                    i++;
                }
            }
#ifdef PIQP_HAS_OPENMP
#pragma omp parallel
            {
                PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::update_scalings_and_factor:parallel");
#endif

                this->block_gemm_nd(this->GT, this->G_scaling, this->GT_scaled);
#ifdef PIQP_HAS_OPENMP
#pragma omp barrier
#endif
                this->block_syrk_ln_calc(this->GT_scaled, this->GT_scaled, this->GtG);
#ifdef PIQP_HAS_OPENMP
#pragma omp barrier
#endif
                populate_kkt_fac(x_reg);
                factor_kkt();

#ifdef PIQP_HAS_OPENMP
            } // end of parallel region
#endif

            return true;
        }


        void solve(const Data<T, I>&, const Vec<T>& rhs_x, const Vec<T>& rhs_y, const Vec<T>& rhs_z, Vec<T>& lhs_x, Vec<T>& lhs_y, Vec<T>& lhs_z) override
        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::solve");

            Vec<T>& rhs_z_bar = this->work_z;
            BlockVec& block_rhs = this->work_x_block_1;
            BlockVec& block_rhs_y = this->work_y_block_1;
            BlockVec& block_rhs_z_bar = this->work_z_block_1;

            BlockVec& block_lhs_x = block_rhs;
            BlockVec& block_lhs_y = this->work_y_block_1;
            BlockVec& block_lhs_z = this->work_z_block_1;

            T delta_inv = T(1) / this->m_delta;

            rhs_z_bar.array() = this->m_z_reg_inv.array() * rhs_z.array();

            block_rhs.assign(rhs_x);
            block_rhs_y.assign(rhs_y, this->AT.perm_inv);
            block_rhs_z_bar.assign(rhs_z_bar, this->GT.perm_inv);


#ifdef PIQP_HAS_OPENMP
#pragma omp parallel
            {
                PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::solve:parallel");
#endif

                // block_rhs += GT * block_rhs_z_bar
                this->block_t_gemv_n(1.0, this->GT, block_rhs_z_bar, 1.0, block_rhs, block_rhs);
#ifdef PIQP_HAS_OPENMP
#pragma omp barrier
#endif
                // block_rhs += delta_inv * AT * block_rhs_y
                this->block_t_gemv_n(delta_inv, this->AT, block_rhs_y, 1.0, block_rhs, block_rhs);

#ifdef PIQP_HAS_OPENMP
#pragma omp barrier
#endif

                solve_llt_in_place(block_rhs);

#ifdef PIQP_HAS_OPENMP
#pragma omp barrier
#endif

                // block_lhs_y = delta_inv * A * block_lhs_x
                this->block_t_gemv_t(delta_inv, this->AT, block_lhs_x, 0.0, block_lhs_y, block_lhs_y);
                // block_lhs_z = G * block_lhs_x
                this->block_t_gemv_t(1.0, this->GT, block_lhs_x, 0.0, block_lhs_z, block_lhs_z);

#ifdef PIQP_HAS_OPENMP
            } // end of parallel region
#endif


            block_lhs_x.load(lhs_x);
            block_lhs_y.load(lhs_y, this->AT.perm_inv);
            block_lhs_z.load(lhs_z, this->GT.perm_inv);

            lhs_y.noalias() -= delta_inv * rhs_y;
            lhs_z.noalias() -= rhs_z;
            lhs_z.array() *= this->m_z_reg_inv.array();
        }



        void factor_kkt() override {
            auto& sub_blocks = kkt_fac_parallel.sub_blocks;

            // Phase 1, parallel

#ifdef PIQP_HAS_OPENMP
#pragma omp for
#endif

            for (size_t index = 0; index < kkt_solve_num_threads; index++) {
                PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::update_scalings_and_factor:factor_kkt:phase_1");
                PIQP_TRACY_ZoneValue(index);
                for (size_t i = 0; i < sub_blocks[index].D.size() - 1; i++) {
                    std::unique_ptr<BlasfeoMat>& D_i = sub_blocks[index].D[i];
                    std::unique_ptr<BlasfeoMat>& E_i = sub_blocks[index].E[i];
                    std::unique_ptr<BlasfeoMat>& D_ip1 = sub_blocks[index].D[i + 1];
                    // D[i] = chol(D[i])
                    assert(D_i->rows() == D_i->cols() && "size mismatch");
                    blasfeo_dpotrf_l(D_i->rows(), D_i->ref(), 0, 0, D_i->ref(), 0, 0);
                    assert(!D_i->hasNan() && "matrix has NaN values");
                    // E[i] = E[i] * D[i]^-T
                    assert(E_i->cols() == D_i->cols() && "size mismatch");
                    blasfeo_dtrsm_rltn(E_i->rows(), E_i->cols(), 1.0, D_i->ref(), 0, 0, E_i->ref(), 0, 0, E_i->ref(), 0, 0);
                    assert(!E_i->hasNan() && "matrix has NaN values");
                    // D[i+1] -= E[i] * E[i].T, rows of E[i] can be smaller than D[i+1]
                    assert(D_ip1->rows() >= E_i->rows() && "size mismatch");
                    blasfeo_dsyrk_ln(E_i->rows(), E_i->cols(), -1.0, E_i->ref(), 0, 0, E_i->ref(), 0, 0, 1.0, D_ip1->ref(), 0, 0, D_ip1->ref(), 0, 0);
                    assert(!D_ip1->hasNan() && "matrix has NaN values");

                    if (index > 0) {
                        std::unique_ptr<BlasfeoMat>& B_i = sub_blocks[index].B[i];
                        std::unique_ptr<BlasfeoMat>& B_ip1 = sub_blocks[index].B[i + 1];
                        // B[i] = D[i]^-1 * B[i]
                        assert(D_i->cols() == B_i->rows() && "size mismatch");
                        blasfeo_dtrsm_llnn(B_i->rows(), B_i->cols(), 1.0, D_i->ref(), 0, 0, B_i->ref(), 0, 0, B_i->ref(), 0, 0);

                        // A -= B[i].T * B[i]
                        std::unique_ptr<BlasfeoMat>& A = sub_blocks[index].A;
                        assert(A->rows() == B_i->cols() && "size mismatch");
                        blasfeo_dsyrk_lt(B_i->cols(), B_i->rows(), -1.0, B_i->ref(), 0, 0, B_i->ref(), 0, 0, 1.0, A->ref(), 0, 0, A->ref(), 0, 0);
                        // B[i+1] -= E[i] * B[i], NOTICE rows of E[i] can be smaller than rows of B[i+1]
                        assert(E_i->cols() == B_i->rows() && "size mismatch");
                        assert(B_ip1->rows() >= E_i->rows() && B_ip1->cols() == B_i->cols() && "size mismatch");
                        blasfeo_dgemm_nn(E_i->rows(), B_i->rows(), B_i->cols(), -1.0, E_i->ref(), 0, 0, B_i->ref(), 0, 0, 1.0, B_ip1->ref(), 0, 0, B_ip1->ref(), 0, 0);
                    }
                }

                // D[-1] = chol(D[-1])
                std::unique_ptr<BlasfeoMat>& D_last = sub_blocks[index].D.back();
                assert(D_last->rows() == D_last->cols() && "size mismatch");
                blasfeo_dpotrf_l(D_last->rows(), D_last->ref(), 0, 0, D_last->ref(), 0, 0);

                if (index > 0) {
                    std::unique_ptr<BlasfeoMat>& B_last = sub_blocks[index].B.back();
                    // B[-1] = D[-1]^-1 * B[-1]
                    assert(B_last->rows() == D_last->cols() && "size mismatch");
                    blasfeo_dtrsm_llnn(B_last->rows(), B_last->cols(), 1.0, D_last->ref(), 0, 0, B_last->ref(), 0, 0, B_last->ref(), 0, 0);
                    // A -= B[i].T * B[i]
                    std::unique_ptr<BlasfeoMat>& A = sub_blocks[index].A;
                    assert(A->rows() == A->cols() && A->rows() == B_last->cols() && "size mismatch");
                    // NOTICE THAT it should be B.cols(), B.rows() in dsyrk_lt, not the other way around!!!!!
                    blasfeo_dsyrk_lt(B_last->cols(), B_last->rows(), -1.0, B_last->ref(), 0, 0, B_last->ref(), 0, 0, 1.0, A->ref(), 0, 0, A->ref(), 0, 0);
                }

                if (index < kkt_solve_num_threads - 1) {
                    // F = F * D[-1]^-T
                    std::unique_ptr<BlasfeoMat>& F = sub_blocks[index].F;
                    assert(F->cols() == D_last->cols() && "size mismatch");
                    blasfeo_dtrsm_rltn(F->rows(), F->cols(), 1.0, D_last->ref(), 0, 0, F->ref(), 0, 0, F->ref(), 0, 0);
                }

                if (index > 0 && index < kkt_solve_num_threads - 1) {
                    // H = -F * B[-1]
                    std::unique_ptr<BlasfeoMat>& B_last = sub_blocks[index].B.back();
                    std::unique_ptr<BlasfeoMat>& F = sub_blocks[index].F;
                    std::unique_ptr<BlasfeoMat>& H = sub_blocks[index].H;
                    assert(F->cols() == B_last->rows() && H->rows() == F->rows() && H->cols() == B_last->cols() && "size mismatch");
                    blasfeo_dgemm_nn(F->rows(), F->cols(), B_last->cols(), -1.0, F->ref(), 0, 0, B_last->ref(), 0, 0, 1.0, H->ref(), 0, 0, H->ref(), 0, 0);
                }

            }

#ifdef PIQP_HAS_OPENMP
#pragma omp single
            {
#endif

                // Phase 2
                {
                    PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::update_scalings_and_factor:factor_kkt:phase_2");
                    for (size_t k = 1; k < kkt_solve_num_threads - 1; k++) {
                        std::unique_ptr<BlasfeoMat> &F_km1 = sub_blocks[k - 1].F;
                        std::unique_ptr<BlasfeoMat> &A_k = sub_blocks[k].A;
                        std::unique_ptr<BlasfeoMat> &A_kp1 = sub_blocks[k + 1].A;
                        std::unique_ptr<BlasfeoMat> &H_k = sub_blocks[k].H;
                        // A[i] -= F[i-1] * F[i-1]^T, notice that rows of F[i-1] might be smaller than rows of A[i]
                        assert(A_k->rows() >= F_km1->rows() && "size mismatch");
                        blasfeo_dsyrk_ln(F_km1->rows(), F_km1->cols(), -1.0, F_km1->ref(), 0, 0, F_km1->ref(), 0, 0, 1.0,
                                         A_k->ref(), 0, 0, A_k->ref(), 0, 0);
                        // A[i] = chol(A[i])
                        assert(A_k->rows() == A_k->cols() && "size mismatch");
                        blasfeo_dpotrf_l(A_k->rows(), A_k->ref(), 0, 0, A_k->ref(), 0, 0);
                        // H[i] = H[i] * A[i]^-T
                        assert(H_k->cols() == A_k->cols() && "size mismatch");
                        blasfeo_dtrsm_rltn(H_k->rows(), H_k->cols(), 1.0, A_k->ref(), 0, 0, H_k->ref(), 0, 0, H_k->ref(), 0,
                                           0);
                        // A[i+1] -= H[i] * H[i]^T, notice that rows of H[i] might be smaller than rows of A[i+1]
                        assert(A_kp1->rows() >= H_k->rows() && "size mismatch");
                        blasfeo_dsyrk_ln(H_k->rows(), H_k->cols(), -1.0, H_k->ref(), 0, 0, H_k->ref(), 0, 0, 1.0,
                                         A_kp1->ref(),
                                         0, 0, A_kp1->ref(), 0, 0);
                    }

                    // A[-1] -= F[-2] * F[-2]^T
                    const size_t k = kkt_solve_num_threads - 1;
                    assert(sub_blocks[k].A->rows() >= sub_blocks[k - 1].F->rows() && "size mismatch");
                    blasfeo_dsyrk_ln(sub_blocks[k - 1].F->rows(), sub_blocks[k - 1].F->cols(), -1.0,
                                     sub_blocks[k - 1].F->ref(),
                                     0, 0, sub_blocks[k - 1].F->ref(), 0, 0, 1.0, sub_blocks[k].A->ref(), 0, 0,
                                     sub_blocks[k].A->ref(), 0, 0);
                    // A[-1] = chol(A[-1])
                    assert(sub_blocks[k].A->rows() == sub_blocks[k].A->cols() && "size mismatch");
                    blasfeo_dpotrf_l(sub_blocks[k].A->rows(), sub_blocks[k].A->ref(), 0, 0, sub_blocks[k].A->ref(), 0, 0);

                }
#ifdef PIQP_HAS_OPENMP
            } // end of single region
#endif
        }


        void solve_llt_in_place(BlockVec& b_and_x) override
        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::solve_llt_in_place");
            solve_llt_in_place_forward(b_and_x);
            solve_llt_in_place_backward(b_and_x);
        }

        void solve_llt_in_place_forward(BlockVec& b_and_x) const {
            // --- Forward Substitution
            PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::solve_llt_in_place:forward");
            const auto& sub_blocks = kkt_fac_parallel.sub_blocks;

#ifdef PIQP_HAS_OPENMP
#pragma omp for
#endif
            for (size_t k = 0; k < kkt_solve_num_threads; k++) {
                PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::solve_llt_in_place:forward:segments");
                PIQP_TRACY_ZoneValue(k);
                // y_1 = D_1^{-1} * b_1
                auto& vec = b_and_x.x[segments[k][0]];
                assert(vec.rows() == sub_blocks[k].D[0]->rows() && "size mismatch");
                blasfeo_dtrsv_lnn(vec.rows(), sub_blocks[k].D[0]->ref(), 0, 0, vec.ref(), 0, vec.ref(), 0);
                // rhs - B_1^T * b_1
                if (k > 0) {
                    const auto& B_0 = sub_blocks[k].B[0];
                    assert(vec.rows() == B_0->rows() && "size mismatch");
                    assert(b_and_x.x[pivots[k-1]].rows() == B_0->cols() && "size mismatch");
                    blasfeo_dgemv_t(B_0->rows(), B_0->cols(), -1.0, B_0->ref(), 0, 0,
                                    vec.ref(), 0, 1.0, b_and_x.x[pivots[k-1]].ref(), 0, b_and_x.x[pivots[k-1]].ref(), 0);
                }
                assert(!vec.hasNan() && "vector has NaN values");
                for (size_t i = 1; i < segments[k].size(); i++) {
                    // y2 = D_2^{-1} * (b_2 - E_1 * y_1)
                    const auto& E_im1 = sub_blocks[k].E[i-1];
                    const auto& D_i = sub_blocks[k].D[i];
                    auto& vec_i = b_and_x.x[segments[k][i]];
                    auto& vec_im1 = b_and_x.x[segments[k][i-1]];
                    assert(vec_im1.rows() == E_im1->cols() && "size mismatch");
                    blasfeo_dgemv_n(E_im1->rows(), E_im1->cols(), -1.0, E_im1->ref(), 0, 0, vec_im1.ref(), 0, 1.0, vec_i.ref(), 0, vec_i.ref(), 0);
                    assert(!vec_im1.hasNan() && "vector has NaN values");
                    blasfeo_dtrsv_lnn(D_i->rows(), D_i->ref(), 0, 0, vec_i.ref(), 0, vec_i.ref(), 0);

                    // rhs - B_1^T * b_1
                    if (k > 0) {
                        const auto& B_i = sub_blocks[k].B[i];
                        assert(vec_i.rows() == B_i->rows() && "size mismatch");
                        assert(b_and_x.x[pivots[k-1]].rows() == B_i->cols() && "size mismatch");
                        // blasfeo_dgemv_t(B_i->rows(), B_i->cols(), -1.0, B_i->ref(), 0, 0,
                        //                 vec_i.ref(), 0, 1.0, b_and_x.x[pivots[k-1]].ref(), 0, b_and_x.x[pivots[k-1]].ref(), 0);
                        blasfeo_dgemv_t(-1.0, *B_i, vec_i, 1.0, b_and_x.x[pivots[k-1]], b_and_x.x[pivots[k-1]]);
                    }
                    assert(!vec_i.hasNan() && "vector has NaN values");
                }
            }

            // deal with pivots
#ifdef PIQP_HAS_OPENMP
#pragma omp barrier
#pragma omp master
        {
#endif
            {
                PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::solve_llt_in_place:forward:pivots");
                for (size_t k = 1; k < kkt_solve_num_threads; k++) {
                    blasfeo_dvec *vec_k = b_and_x.x[pivots[k - 1]].ref();
                    // rhs - F[k-1] * v
                    const std::unique_ptr<BlasfeoMat> &mat_F = sub_blocks[k - 1].F;
                    assert(vec_k->m == mat_F->cols() && "size mismatch");
                    blasfeo_dgemv_n(mat_F->rows(), mat_F->cols(), -1.0, mat_F->ref(), 0, 0,
                                    b_and_x.x[segments[k - 1].back()].ref(), 0, 1.0, vec_k, 0, vec_k, 0);
                    assert(!b_and_x.x[pivots[k - 1]].hasNan() && "vector has NaN values");

                    // rhs - H * v
                    if (k > 1) {
                        const std::unique_ptr<BlasfeoMat> &mat_H = sub_blocks[k - 1].H;
                        assert(vec_k->m == mat_H->cols() && "size mismatch");
                        blasfeo_dgemv_n(mat_H->rows(), mat_H->cols(), -1.0, mat_H->ref(), 0, 0,
                                        b_and_x.x[pivots[k - 2]].ref(), 0, 1.0, vec_k, 0, vec_k, 0);
                        assert(!b_and_x.x[pivots[k - 2]].hasNan() && "vector has NaN values");
                        assert(!b_and_x.x[pivots[k - 1]].hasNan() && "vector has NaN values");
                    }

                    assert(vec_k->m == sub_blocks[k].A->rows() && "size mismatch");
                    blasfeo_dtrsv_lnn(sub_blocks[k].A->rows(), sub_blocks[k].A->ref(), 0, 0, vec_k, 0, vec_k, 0);
                    assert(!b_and_x.x[pivots[k - 1]].hasNan() && "vector has NaN values");
                }
            }

#ifdef PIQP_HAS_OPENMP
            }
#endif
        }

        void solve_llt_in_place_backward(BlockVec& b_and_x) const {
            // --- Backward Substitution
            PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::solve_llt_in_place:backward");
            const auto& sub_blocks = kkt_fac_parallel.sub_blocks;

#ifdef PIQP_HAS_OPENMP
#pragma omp master
        {
#endif

            {
                PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::solve_llt_in_place:backward:pivots");
                assert(sub_blocks.back().A->rows() == b_and_x.x[pivots.back()].rows() && "size mismatch");
                blasfeo_dtrsv_ltn(sub_blocks.back().A->rows(), sub_blocks.back().A->ref(), 0, 0,
                                  b_and_x.x[pivots.back()].ref(), 0, b_and_x.x[pivots.back()].ref(), 0);
                for (size_t k = pivots.size() - 2; k != SIZE_MAX; k--) {
                    assert(sub_blocks[k + 1].H->rows() <= b_and_x.x[pivots[k + 1]].rows() && "size mismatch");
                    assert(sub_blocks[k + 1].H->cols() == b_and_x.x[pivots[k]].rows() && "size mismatch");
                    blasfeo_dgemv_t(sub_blocks[k + 1].H->rows(), sub_blocks[k + 1].H->cols(), -1.0,
                                    sub_blocks[k + 1].H->ref(), 0, 0, b_and_x.x[pivots[k + 1]].ref(), 0, 1.0,
                                    b_and_x.x[pivots[k]].ref(), 0, b_and_x.x[pivots[k]].ref(), 0);
                    assert(sub_blocks[k + 1].A->rows() == b_and_x.x[pivots[k]].rows() && "size mismatch");
                    blasfeo_dtrsv_ltn(sub_blocks[k + 1].A->rows(), sub_blocks[k + 1].A->ref(), 0, 0,
                                      b_and_x.x[pivots[k]].ref(), 0, b_and_x.x[pivots[k]].ref(), 0);
                }
            }

#ifdef PIQP_HAS_OPENMP
            }
#pragma omp barrier
#pragma omp for
#endif
            for (size_t k = 0; k < kkt_solve_num_threads; k++) {
                PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::solve_llt_in_place:backward:segments");
                PIQP_TRACY_ZoneValue(k);
                if (k > 0) {
                    for (size_t i = segments[k].size() - 1; i != SIZE_MAX; i--) {
                        const std::unique_ptr<BlasfeoMat>& B_i = sub_blocks[k].B[i];
                        assert(B_i->cols() == b_and_x.x[pivots[k-1]].rows() && "size mismatch");
                        assert(B_i->rows() == b_and_x.x[segments[k][i]].rows() && "size mismatch");
                        blasfeo_dgemv_n(B_i->rows(), B_i->cols(), -1.0, B_i->ref(), 0, 0, b_and_x.x[pivots[k-1]].ref(), 0, 1.0, b_and_x.x[segments[k][i]].ref(), 0, b_and_x.x[segments[k][i]].ref(), 0);
                    }
                }

                if (k < kkt_solve_num_threads - 1) {
                    assert(sub_blocks[k].F->rows() <= b_and_x.x[pivots[k]].rows() && "size mismatch");
                    assert(sub_blocks[k].F->cols() == b_and_x.x[segments[k].back()].rows() && "size mismatch");
                    blasfeo_dgemv_t(sub_blocks[k].F->rows(), sub_blocks[k].F->cols(), -1.0, sub_blocks[k].F->ref(), 0, 0, b_and_x.x[pivots[k]].ref(), 0, 1.0, b_and_x.x[segments[k].back()].ref(), 0, b_and_x.x[segments[k].back()].ref(), 0);
                }

                blasfeo_dtrsv_ltn(sub_blocks[k].D.back()->rows(), sub_blocks[k].D.back()->ref(), 0, 0, b_and_x.x[segments[k].back()].ref(), 0, b_and_x.x[segments[k].back()].ref(), 0);
                for (size_t i = segments[k].size()-2; i != SIZE_MAX; i--) {
                    assert(sub_blocks[k].E[i]->rows() <= b_and_x.x[segments[k][i+1]].rows() && "size mismatch");
                    assert(sub_blocks[k].E[i]->cols() == b_and_x.x[segments[k][i]].rows() && "size mismatch");
                    blasfeo_dgemv_t(sub_blocks[k].E[i]->rows(), sub_blocks[k].E[i]->cols(), -1.0, sub_blocks[k].E[i]->ref(), 0, 0, b_and_x.x[segments[k][i+1]].ref(), 0, 1.0, b_and_x.x[segments[k][i]].ref(), 0, b_and_x.x[segments[k][i]].ref(), 0);
                    assert(sub_blocks[k].D[i]->rows() == b_and_x.x[segments[k][i]].rows() && "size mismatch");
                    blasfeo_dtrsv_ltn(sub_blocks[k].D[i]->rows(), sub_blocks[k].D[i]->ref(), 0, 0, b_and_x.x[segments[k][i]].ref(), 0, b_and_x.x[segments[k][i]].ref(), 0);
                }
            }
        }


    };
}
}

#endif //PIQP_MULTISTAGE_PARALLEL_KKT_H

