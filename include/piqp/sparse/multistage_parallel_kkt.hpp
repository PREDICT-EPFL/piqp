#ifndef PIQP_MULTISTAGE_PARALLEL_KKT_H
#define PIQP_MULTISTAGE_PARALLEL_KKT_H

#include "piqp/sparse/multistage_kkt.hpp"
#include "piqp/sparse/blocksparse/block_kkt_parallel.hpp"
#ifdef PIQP_HAS_OPENMP
#include "omp.h"
#endif


__attribute__((constructor))
inline void setup_omp_options() {
#ifdef PIQP_HAS_OPENMP

    omp_set_dynamic(0); // disable dynamic teams
    omp_set_schedule(omp_sched_static, 0); // fix scheduling method

    // Print the number of threads being used
    // piqp_print("OpenMP: Using %d threads.\n", omp_get_max_threads());

    // Bind threads onto cores
#ifdef __linux__
    if (setenv("OMP_DISPLAY_ENV", "TRUE", 1) != 0) {
        piqp_eprint("Error setting OMP_DISPLAY_ENV\n");
    }
    if (setenv("OMP_WAIT_POLICY", "ACTIVE", 1) != 0) {
        piqp_eprint("Error setting OMP_WAIT_POLICY\n");
    }

#pragma omp parallel num_threads(omp_get_max_threads())
    {
        int tid = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(tid, &cpuset);

        if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) < 0) {
            piqp_eprint("Failed to set thread affinity for thread %d\n", tid);
        }
        // piqp_print("Thread %d running on CPU %d\n", tid, sched_getcpu());
    }
#else
    piqp_print("Cannot bind threads onto cores. The performance might be considerably compromised.\n");
#endif
#endif
}



namespace piqp
{

namespace sparse
{
    template<typename T, typename I>
    class MultistageParallelKKT : public MultistageKKT<T, I>
    {

    protected:
        static_assert(std::is_same<T, double>::value, "sparse_multistage_parallel only supports doubles");

        // For parallel factorization
        size_t kkt_solve_num_threads = 0;
        BlockKKTParallel kkt_fac_parallel;
        std::vector<size_t> pivots;
        std::vector<std::vector<size_t>> segments;
        std::vector<std::unique_ptr<BlasfeoVec>> work_rhs_g;  // store the r_g for each thread in forward substitution

    public:
        explicit MultistageParallelKKT(const Data<T, I>& data)
            : MultistageKKT<T, I>(data) {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::constructor");

            init();
        }

        void init() {
            kkt_solve_num_threads = static_cast<size_t>(omp_get_max_threads());
            if (omp_get_max_threads() <= 1) {
                throw std::runtime_error("Number of threads must be greater than 1 when using Multistage Parallel KKT solver.");
            }
            if (this->block_info.size() - 1 <= 2 * kkt_solve_num_threads) {
                throw std::runtime_error("The multistage problem's horizon is too short for given number thread to use parallel multistage kkt solver. Please decrease the number of threads of use serial solver instead.");
            }
            omp_set_num_threads(static_cast<int>(kkt_solve_num_threads));

            // setup_omp_options();

            generate_partitions();  // Generate partitions for multi-threads
            init_kkt_fac();

            if (this->block_info.back().diag_size > 0) {
                work_rhs_g.resize(kkt_solve_num_threads);
                for (size_t i = 0; i < kkt_solve_num_threads; i++) {
                    work_rhs_g[i] = std::make_unique<BlasfeoVec>(this->block_info.back().diag_size);
                }
            }

        }

        /**
         * @brief This constructor function is automatically executed when the library is loaded.
         *
         * It runs before the user's main() function (if dynamically linked at startup)
         * and definitely before any of your solver's functions are called.
         * This is the correct place to configure the OpenMP environment.
         */

        void generate_partitions() {
            pivots.clear();
            segments.clear();

            const size_t N = this->block_info.size() - 1;  // number of diagonal blocks excluding arrow head

            // this->print_info();

            // Detect scenerio mpc structure
            std::vector<size_t> idx_empty_off_diag_blocks;
            idx_empty_off_diag_blocks.clear();
            for (size_t i = 0; i < N; i++) {
                if (this->block_info[i].off_diag_size == 0) {
                    idx_empty_off_diag_blocks.push_back(i);
                }
            }

            if (idx_empty_off_diag_blocks.size() <= 1) {
                const size_t P = kkt_solve_num_threads;
                // Compute segment size such that first segment is ~19/7 times others
                T ratio = T(19.0) / T(7.0);  // the optimal ratio of first segment length over intermediate segments lengths
                T Ni_ideal = T(N - P + 1) / (T(P - 1) + ratio);
                auto Ni_ceil = static_cast<size_t>(std::ceil(Ni_ideal));
                auto Ni_floor = static_cast<size_t>(std::floor(Ni_ideal));

                size_t Ni = 0, N1 = 0;
                if (N <= (P - 1) * (Ni_ceil + 1) ) {
                    // If ceiling Ni makes the left N1 zero or negative, we have to pick flooring
                    Ni = Ni_floor;
                    if ((P - 1) * (Ni + 1) >= N) {
                        throw std::runtime_error("The multistage problem's horizon is too short for given number thread to use parallel multistage kkt solver. Please decrease the number of threads of use serial solver instead.");
                    }
                    N1 = N - (P - 1) * (Ni + 1);
                } else {
                    // Compare ceiling and floor
                    // lambda: time complexity of the parallel part if we pick Ni
                    auto cost_parallel_part = [N, P](size_t Ni) -> T {
                        size_t N1 = N - (P - 1) * (Ni + 1);
                        // 7/3 * N1  vs  19/3 * Ni
                        return std::max(T(7.0)/T(3.0)*static_cast<T>(N1),
                                        T(19.0)/T(3.0)*static_cast<T>(Ni));
                    };
                    Ni = cost_parallel_part(Ni_ceil) < cost_parallel_part(Ni_floor)? Ni_ceil : Ni_floor;
                    N1 = N - (P - 1) * (Ni + 1);
                }

                // Compute pivot indices
                pivots.reserve(P - 1);
                for (size_t j = 0; j < P - 1; ++j) {
                    size_t pivot_j = N1 + Ni * j + j;
                    pivots.push_back(pivot_j);
                }

                std::vector<size_t> segment_i;
                for (size_t i = 0; i < N1; i++) { segment_i.push_back(i); }
                segments.push_back(segment_i);
                for (size_t j = 0; j < P-1; ++j) {
                    segment_i.clear();
                    for (size_t i = 0; i < Ni; i++) { segment_i.push_back(pivots[j] + 1 + i); }
                    segments.push_back(segment_i);
                }
            } else {
                // Scenario MPC, naturally parallelizable structure
                kkt_solve_num_threads = idx_empty_off_diag_blocks.size();  // TODO: what if too many scenarios? more than number of threads?

                std::vector<size_t> pivots_tmp = idx_empty_off_diag_blocks;
                pivots_tmp.insert(pivots_tmp.begin(), static_cast<size_t>(-1));  // underflows to max size_t
                for (size_t i = 0; i < pivots_tmp.size() - 1; ++i) {
                    std::vector<size_t> segment;
                    for (size_t j = pivots_tmp[i] + 1; j <= pivots_tmp[i + 1]; ++j) {
                        segment.push_back(j);
                    }
                    segments.push_back(segment);
                }
            }


            // std::cout << "segments:";
            // for (size_t i = 0; i < segments.size(); i++) {
            //     std::cout << "\n  segment " << i << ": ";
            //     for (size_t j = 0; j < segments[i].size(); j++) {
            //         std::cout << segments[i][j] << " ";
            //     }
            // }


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

            I arrow_width = this->block_info.back().diag_size;
            T delta_inv = 1.0 / this->m_delta;
            BlockVec& x_reg_block = this->work_x_block_1;
            x_reg_block.assign(x_reg);

            if (allocate) {

                kkt_fac_parallel.sub_blocks.resize(segments.size());
                kkt_fac_parallel.num_threads = segments.size();

                // TODO: parallelize the following loop
                for (size_t k = 0; k < segments.size(); k++) {
                    auto &sub_block = kkt_fac_parallel.sub_blocks[k];
                    sub_block.D.clear();
                    sub_block.D.resize(segments[k].size());
                    sub_block.E.clear();
                    sub_block.E.resize(segments[k].size() - 1);
                    sub_block.Bt.clear();
                    k > 0 && !pivots.empty() ? sub_block.Bt.resize(segments[k].size()) : sub_block.Bt.resize(0);
                    sub_block.G.clear();
                    arrow_width > 0 ? sub_block.G.resize(segments[k].size()) : sub_block.G.resize(0);
                }

                for (size_t k = 0; k < segments.size(); k++) {
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
                    if (k < segments.size() - 1 && !pivots.empty()) {
                        I m_F = this->block_info[pivots[k]-1].off_diag_size;
                        I n_F = this->block_info[pivots[k]-1].diag_size;
                        sub_block.F = std::make_unique<BlasfeoMat>(m_F, n_F);
                    } else {
                        sub_block.F = nullptr;  // The last sub-block does not have an F matrix
                    }

                    // A
                    if (k > 0 && !pivots.empty()) {
                        I m_A = this->block_info[pivots[k-1]].diag_size;
                        sub_block.A = std::make_unique<BlasfeoMat>(m_A, m_A);
                    } else {
                        sub_block.A = nullptr;  // The first sub-block does not have an A matrix
                    }

                    // H
                    if (k > 0 && k < segments.size() - 1 && !pivots.empty()) {
                        sub_block.H = std::make_unique<BlasfeoMat>(sub_block.F->rows(), sub_block.A->cols());
                    } else {
                        sub_block.H = nullptr;  // The first and last sub-blocks do not have an H matrix
                    }

                    // B
                    if (k > 0 && !pivots.empty()) {
                        sub_block.Bt[0] = std::make_unique<BlasfeoMat>(sub_block.A->rows(), sub_block.D[0]->cols());
                        sub_block.Bt0_tmp = std::make_unique<BlasfeoMat>(sub_block.A->rows(), sub_block.D[0]->cols());
                        for (size_t i = 1; i < segments[k].size(); i++) {
                            // ! B[i] must have the size (nx[i+1], nx[i]) !!!  Cannot use the size of offdiagonal matrix in the original KKT!
                            sub_block.Bt[i] = std::make_unique<BlasfeoMat>(sub_block.Bt[i-1]->rows(), sub_block.D[i]->cols());
                        }
                    }

                    // Arrow part (coupling with global variables)
                    if (arrow_width > 0) {
                        // G
                        for (size_t i = 0; i < segments[k].size(); i++) {
                            sub_block.G[i] = std::make_unique<BlasfeoMat>(arrow_width, sub_block.D[i]->cols());
                        }
                        // Q
                        if (k > 0 && !pivots.empty()) { sub_block.Q = std::make_unique<BlasfeoMat>(arrow_width, sub_block.A->cols()); }
                        // R
                        sub_block.R = std::make_unique<BlasfeoMat>(arrow_width, arrow_width);
                    }
                }

            } else {
                // allocate == false

#ifdef PIQP_HAS_OPENMP
#pragma omp for nowait
#endif
                for (size_t k = 0; k < segments.size(); k++) {
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
                    if (sub_block_k.A) {
                        assert(k > 0 && !pivots.empty());
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
                    if (sub_block_k.F) {
                        assert(k < segments.size() - 1 && !pivots.empty());
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
                    if (sub_block_k.H) {
                        assert(k < segments.size() - 1 && !pivots.empty() && !pivots.empty());
                        sub_block_k.H->setZero();
                    }

                    // ----- B -----
                    if (k > 0 && !pivots.empty()) {  // TODO:
                        assert(k > 0 && !pivots.empty());
                        // B_1 - B_end are all zeros. Also set B_0 to zeros hereby.
                        for (size_t i = 0; i < segments[k].size(); i++) {
                            sub_block_k.Bt[i]->setZero();
                        }

                        // B0
                        bool B0_mat_set = false;
                        if (this->P.B[pivots[k-1]]) {
                            assert(this->P.B[pivots[k-1]]->rows() <= sub_block_k.Bt[0]->cols() && "size mismatch");
                            assert(this->P.B[pivots[k-1]]->cols() <= sub_block_k.Bt[0]->rows() && "size mismatch");
                            // B_0t = P.B_
                            blasfeo_dgetr(*this->P.B[pivots[k-1]], *sub_block_k.Bt[0]);
                            B0_mat_set = true;
                        }

                        if (this->AtA.B[pivots[k-1]]) {
                            assert(this->AtA.B[pivots[k-1]]->rows() <= sub_block_k.Bt[0]->cols() && "size mismatch");
                            assert(this->AtA.B[pivots[k-1]]->cols() <= sub_block_k.Bt[0]->rows() && "size mismatch");
                            if (B0_mat_set) {
                                // B_0 += delta^{-1} * AtA.B_
                                blasfeo_dgetr(*this->AtA.B[pivots[k-1]], *sub_block_k.Bt0_tmp);
                                blasfeo_dgead(delta_inv, *sub_block_k.Bt0_tmp, *sub_block_k.Bt[0]);
                            } else {
                                // B_0 = delta^{-1} * AtA.B_
                                blasfeo_dgetr(*this->AtA.B[pivots[k-1]], *sub_block_k.Bt0_tmp);
                                blasfeo_dgecpsc(delta_inv, *sub_block_k.Bt0_tmp, *sub_block_k.Bt[0]);
                                B0_mat_set = true;
                            }
                        }

                        if (this->GtG.B[pivots[k-1]]) {
                            assert(this->GtG.B[pivots[k-1]]->rows() <= sub_block_k.Bt[0]->cols() && "size mismatch");
                            assert(this->GtG.B[pivots[k-1]]->cols() <= sub_block_k.Bt[0]->rows() && "size mismatch");
                            if (B0_mat_set) {
                                // B_0 += GtG.B_
                                blasfeo_dgetr(*this->GtG.B[pivots[k-1]], *sub_block_k.Bt0_tmp);
                                blasfeo_dgead(1.0, *sub_block_k.Bt0_tmp, *sub_block_k.Bt[0]);
                            } else {
                                // B_0 = GtG.B_
                                blasfeo_dgetr(*this->GtG.B[pivots[k-1]], *sub_block_k.Bt0_tmp);
                                blasfeo_dgecp(*sub_block_k.Bt0_tmp, *sub_block_k.Bt[0]);
                                B0_mat_set = true;
                            }
                        }

                        assert(!sub_block_k.Bt[0]->hasNan() && "B matrix has NaN values");
                        assert(!sub_block_k.Bt[0]->hasInf() && "B matrix has Inf values");
                    }

                    // Arrow part (coupling with global variables)
                    if (arrow_width > 0) {
                        // ----- G -----
                        for (size_t i = 0; i < segment_k.size(); i++) {

                            bool G_mat_set = false;
                            if (this->P.E[segment_k[i]]) {
                                assert(this->P.E[segment_k[i]]->rows() <= sub_block_k.G[i]->rows() && "size mismatch");
                                assert(this->P.E[segment_k[i]]->cols() <= sub_block_k.D[i]->cols() && "size mismatch");
                                blasfeo_dgecp(*this->P.E[segment_k[i]], *sub_block_k.G[i]);
                                G_mat_set = true;
                            }

                            // the terms AtA.E or GtG.E might be smaller,
                            // thus we have to zero the whole matrix just in case
                            if (sub_block_k.G[i] && !G_mat_set) {
                                sub_block_k.G[i]->setZero();
                            }

                            if (this->AtA.E[segment_k[i]]) {
                                assert(this->AtA.E[segment_k[i]]->rows() <= sub_block_k.G[i]->rows() && "size mismatch");
                                assert(this->AtA.E[segment_k[i]]->cols() <= sub_block_k.D[i]->cols() && "size mismatch");
                                if (G_mat_set) {
                                    // G_i += delta^{-1} * AtA.E_i
                                    blasfeo_dgead(delta_inv, *this->AtA.E[segment_k[i]], *sub_block_k.G[i]); // TODO: is there a blasfeo function that only add the lower triangular part?
                                } else {
                                    // G_i = delta^{-1} * AtA.E_i, lower triangular
                                    blasfeo_dgecpsc(delta_inv, *this->AtA.E[segment_k[i]], *sub_block_k.G[i]);
                                    G_mat_set = true;
                                }
                            }

                            if (this->GtG.E[segment_k[i]]) {
                                assert(this->GtG.E[segment_k[i]]->rows() <= sub_block_k.G[i]->rows() && "size mismatch");
                                assert(this->GtG.E[segment_k[i]]->cols() <= sub_block_k.G[i]->cols() && "size mismatch");
                                if (G_mat_set) {
                                    // D_i += GtG.D_i
                                    blasfeo_dgead(1.0, *this->GtG.E[segment_k[i]], *sub_block_k.G[i]);
                                } else {
                                    // D_i = GtG.D_i, lower triangular
                                    blasfeo_dgecp(*this->GtG.E[segment_k[i]], *sub_block_k.G[i]);
                                    G_mat_set = true;
                                }
                            }

                            assert(!sub_block_k.G[i]->hasNan() && "G matrix has NaN values");
                            assert(!sub_block_k.G[i]->hasInf() && "G matrix has Inf values");
                        }

                        // ----- Q -----
                        if (sub_block_k.Q) {
                            assert(k > 0 && !pivots.empty());
                            bool Q_mat_set = false;
                            if (this->P.E[pivots[k-1]]) {
                                assert(this->P.E[pivots[k-1]]->rows() <= sub_block_k.Q->rows() && "size mismatch");
                                assert(this->P.E[pivots[k-1]]->cols() <= sub_block_k.Q->cols() && "size mismatch");
                                blasfeo_dgecp(*this->P.E[pivots[k-1]], *sub_block_k.Q);
                                Q_mat_set = true;
                            }

                            // the terms AtA.E or GtG.E might be smaller,
                            // thus we have to zero the whole matrix just in case
                            if (sub_block_k.Q && !Q_mat_set) {
                                sub_block_k.Q->setZero();
                            }

                            if (this->AtA.E[pivots[k-1]]) {
                                assert(this->AtA.E[pivots[k-1]]->rows() <= sub_block_k.Q->rows() && "size mismatch");
                                assert(this->AtA.E[pivots[k-1]]->cols() <= sub_block_k.Q->cols() && "size mismatch");
                                if (Q_mat_set) {
                                    blasfeo_dgead(delta_inv, *this->AtA.E[pivots[k-1]], *sub_block_k.Q);
                                } else {
                                    blasfeo_dgecpsc(delta_inv, *this->AtA.E[pivots[k-1]], *sub_block_k.Q);
                                    Q_mat_set = true;
                                }
                            }

                            if (this->GtG.E[pivots[k-1]]) {
                                assert(this->GtG.E[pivots[k-1]]->rows() <= sub_block_k.Q->rows() && "size mismatch");
                                assert(this->GtG.E[pivots[k-1]]->cols() <= sub_block_k.Q->cols() && "size mismatch");
                                if (Q_mat_set) {
                                    blasfeo_dgead(1.0, *this->GtG.E[pivots[k-1]], *sub_block_k.Q);
                                } else {
                                    blasfeo_dgecp(*this->GtG.E[pivots[k-1]], *sub_block_k.Q);
                                    Q_mat_set = true;
                                }
                            }

                            assert(!sub_block_k.Q->hasNan() && "Q matrix has NaN values");
                            assert(!sub_block_k.Q->hasInf() && "Q matrix has Inf values");
                        }

                        // ----- R -----
                        bool R_mat_set = false;
                        T num_segments_inv = static_cast<T>(static_cast<T>(1.0) / static_cast<T>(segments.size()));
                        if (this->P.D.back()) {
                            // R = P.D_i, lower triangular
                            assert(this->P.D.back()->rows() <= sub_block_k.R->rows() && "size mismatch");
                            assert(this->P.D.back()->cols() <= sub_block_k.R->cols() && "size mismatch");
                            blasfeo_dtrcpsc_l(num_segments_inv, *this->P.D.back(), *sub_block_k.R);
                            R_mat_set = true;
                        }

                        if (this->AtA.D.back()) {
                            assert(this->AtA.D.back()->rows() <= sub_block_k.R->rows() && "size mismatch");
                            assert(this->AtA.D.back()->cols() <= sub_block_k.R->cols() && "size mismatch");
                            if (R_mat_set) {
                                // R += delta^{-1} * AtA.D_i
                                blasfeo_dgead(delta_inv * num_segments_inv, *this->AtA.D.back(), *sub_block_k.R); // TODO: is there a blasfeo function that only add the lower triangular part?
                            } else {
                                // R = delta^{-1} * AtA.D_i, lower triangular
                                blasfeo_dtrcpsc_l(delta_inv * num_segments_inv, *this->AtA.D.back(), *sub_block_k.R);
                                R_mat_set = true;
                            }
                        }

                        if (this->GtG.D.back()) {
                            assert(this->GtG.D.back()->rows() <= sub_block_k.R->rows() && "size mismatch");
                            assert(this->GtG.D.back()->cols() <= sub_block_k.R->cols() && "size mismatch");
                            if (R_mat_set) {
                                // R += GtG.D_i
                                blasfeo_dgead(num_segments_inv, *this->GtG.D.back(), *sub_block_k.R);
                            } else {
                                // R = GtG.D_i, lower triangular
                                blasfeo_dtrcpsc_l(num_segments_inv, *this->GtG.D.back(), *sub_block_k.R);
                                R_mat_set = true;
                            }
                        }

                        if (R_mat_set) {
                            // diag(R) += diag
                            blasfeo_ddiaad(num_segments_inv, x_reg_block.x.back(), *sub_block_k.R);
                        } else {
                            // R = diag
                            sub_block_k.R->setZero();
                            blasfeo_ddiain(num_segments_inv, x_reg_block.x.back(), *sub_block_k.R);
                        }

                        assert(!sub_block_k.R->hasNan() && "R matrix has NaN values");
                        assert(!sub_block_k.R->hasInf() && "R matrix has Inf values");

                    }
                }
            }  // end of allocate if-else

        }


        bool update_scalings_and_factor(const Data<T, I>&, const T& delta, const Vec<T>& x_reg, const Vec<T>& z_reg) override
        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::update_scalings_and_factor");
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
                // PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::update_scalings_and_factor:parallel");
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
            const I arrow_width = this->block_info.back().diag_size;

            // Phase 1, parallel

#ifdef PIQP_HAS_OPENMP
#pragma omp for
#endif

            for (size_t index = 0; index < segments.size(); index++) {
                PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::factor_kkt::phase_1");
                PIQP_TRACY_ZoneValue(index);
                std::unique_ptr<BlasfeoMat>& R = sub_blocks[index].R;
                for (size_t i = 0; i < sub_blocks[index].D.size() - 1; i++) {
                    std::unique_ptr<BlasfeoMat>& D_i = sub_blocks[index].D[i];
                    std::unique_ptr<BlasfeoMat>& E_i = sub_blocks[index].E[i];
                    std::unique_ptr<BlasfeoMat>& D_ip1 = sub_blocks[index].D[i + 1];
                    // D[i] = chol(D[i])
                    blasfeo_dpotrf_l(*D_i);
                    assert(!D_i->hasNan() && "matrix has NaN values");
                    // E[i] = E[i] * D[i]^-T
                    assert(E_i->cols() == D_i->cols() && "size mismatch");
                    blasfeo_dtrsm_rltn(E_i->rows(), E_i->cols(), 1.0, D_i->ref(), 0, 0, E_i->ref(), 0, 0, E_i->ref(), 0, 0);
                    assert(!E_i->hasNan() && "matrix has NaN values");
                    // D[i+1] -= E[i] * E[i].T, rows of E[i] can be smaller than D[i+1]
                    assert(D_ip1->rows() >= E_i->rows() && "size mismatch");
                    blasfeo_dsyrk_ln(E_i->rows(), E_i->cols(), -1.0, E_i->ref(), 0, 0, E_i->ref(), 0, 0, 1.0, D_ip1->ref(), 0, 0, D_ip1->ref(), 0, 0);
                    assert(!D_ip1->hasNan() && "matrix has NaN values");

                    if (arrow_width > 0) {
                        std::unique_ptr<BlasfeoMat>& G_i = sub_blocks[index].G[i];
                        std::unique_ptr<BlasfeoMat>& G_ip1 = sub_blocks[index].G[i + 1];
                        // G[i] = G[i] * D[i]^-T
                        assert(G_i->cols() == D_i->cols() && "size mismatch");
                        blasfeo_dtrsm_rltn(G_i->rows(), G_i->cols(), 1.0, D_i->ref(), 0, 0, G_i->ref(), 0, 0, G_i->ref(), 0, 0);
                        assert(!G_i->hasNan() && "matrix has NaN values");
                        // R -= G[i] * G[i].T
                        assert(R->rows() >= G_i->rows() && "size mismatch");
                        blasfeo_dsyrk_ln(-1.0, *G_i, *G_i, 1.0, *R, *R);
                        assert(!R->hasNan() && "matrix has NaN values");
                        // G[i+1] -= G[i] * E[i].T
                        assert(G_i->cols() == D_i->cols() && "size mismatch");
                        blasfeo_dgemm_nt(-1.0, *G_i, *E_i, 1.0, *G_ip1, *G_ip1);
                        assert(!G_ip1->hasNan() && "matrix has NaN values");
                    }

                    if (!sub_blocks[index].Bt.empty()) {
                        assert(index > 0 && !pivots.empty());
                        // TODO: check if B[i] and B[i+1] are nullptr or not
                        std::unique_ptr<BlasfeoMat>& Bt_i = sub_blocks[index].Bt[i];
                        std::unique_ptr<BlasfeoMat>& Bt_ip1 = sub_blocks[index].Bt[i + 1];
                        // Bt[i] = Bt[i] *  D[i]^-T
                        assert(D_i->cols() == Bt_i->cols() && "size mismatch");
                        blasfeo_dtrsm_rltn(Bt_i->rows(), Bt_i->cols(), 1.0, D_i->ref(), 0, 0, Bt_i->ref(), 0, 0, Bt_i->ref(), 0, 0);

                        // A -= B[i].T * B[i]
                        std::unique_ptr<BlasfeoMat>& A = sub_blocks[index].A;
                        assert(A->rows() == Bt_i->rows() && "size mismatch");
                        blasfeo_dsyrk_ln(-1.0, *Bt_i, *Bt_i, 1.0, *A, *A);
                        // B[i+1].T -= B[i].T * E[i].T, NOTICE rows of E[i] can be smaller than rows of B[i+1]
                        assert(E_i->cols() == Bt_i->cols() && "size mismatch");
                        assert(Bt_ip1->cols() >= E_i->rows() && Bt_ip1->rows() == Bt_i->rows() && "size mismatch");
                        blasfeo_dgemm_nt(-1.0, *Bt_i, *E_i, 1.0, *Bt_ip1, *Bt_ip1);

                        if (arrow_width > 0) {
                            // Q -= G[i] * B[i]
                            std::unique_ptr<BlasfeoMat>& G_i = sub_blocks[index].G[i];
                            std::unique_ptr<BlasfeoMat>& Q = sub_blocks[index].Q;
                            assert(G_i->cols() == Bt_i->cols() && "size mismatch");
                            blasfeo_dgemm_nt(-1.0, *G_i, *Bt_i, 1.0, *Q, *Q);
                            assert(!Q->hasNan() && "matrix has NaN values");
                        }
                    }
                }

                // D[-1] = chol(D[-1])
                std::unique_ptr<BlasfeoMat>& D_last = sub_blocks[index].D.back();
                blasfeo_dpotrf_l(*D_last);

                if (arrow_width > 0) {
                    // G[-1] = G[-1] * D[-1]^-T
                    std::unique_ptr<BlasfeoMat>& G_last = sub_blocks[index].G.back();
                    assert(G_last->cols() == D_last->cols() && "size mismatch");
                    blasfeo_dtrsm_rltn(G_last->rows(), G_last->cols(), 1.0, D_last->ref(), 0, 0, G_last->ref(), 0, 0, G_last->ref(), 0, 0);
                    assert(!G_last->hasNan() && "matrix has NaN values");
                    // R -= G[-1].T * G[-1]
                    assert(R->rows() >= G_last->rows() && "size mismatch");
                    blasfeo_dsyrk_ln(G_last->rows(), G_last->cols(), -1.0, G_last->ref(), 0, 0, G_last->ref(), 0, 0, 1.0, R->ref(), 0, 0, R->ref(), 0, 0);
                    assert(!R->hasNan() && "matrix has NaN values");
                }


                if (!sub_blocks[index].Bt.empty()) {
                    assert(index > 0 && !pivots.empty());
                    std::unique_ptr<BlasfeoMat>& Bt_last = sub_blocks[index].Bt.back();
                    // B[-1].T = B[-1].T * D[-1]^-T
                    assert(Bt_last->cols() == D_last->cols() && "size mismatch");
                    blasfeo_dtrsm_rltn(Bt_last->rows(), Bt_last->cols(), 1.0, D_last->ref(), 0, 0, Bt_last->ref(), 0, 0, Bt_last->ref(), 0, 0);
                    // A -= B[i].T * B[i]
                    std::unique_ptr<BlasfeoMat>& A = sub_blocks[index].A;
                    assert(A->rows() == A->cols() && A->rows() == Bt_last->rows() && "size mismatch");
                    // NOTICE THAT it should be B.cols(), B.rows() in dsyrk_lt, not the other way around!!!!!
                    blasfeo_dsyrk_ln(-1.0, *Bt_last, *Bt_last, 1.0, *A, *A);

                    if (arrow_width > 0) {
                        // Q -= G[-1] * B[-1]
                        std::unique_ptr<BlasfeoMat>& G_last = sub_blocks[index].G.back();
                        std::unique_ptr<BlasfeoMat>& Q = sub_blocks[index].Q;
                        assert(G_last->cols() == Bt_last->cols() && "size mismatch");
                        blasfeo_dgemm_nt(-1.0, *G_last, *Bt_last, 1.0, *Q, *Q);
                        assert(!Q->hasNan() && "matrix has NaN values");
                    }
                }

                if (sub_blocks[index].F) {
                    assert(index < segments.size() - 1 && !pivots.empty());
                    // F = F * D[-1]^-T
                    std::unique_ptr<BlasfeoMat>& F = sub_blocks[index].F;
                    assert(F->cols() == D_last->cols() && "size mismatch");
                    blasfeo_dtrsm_rltn(F->rows(), F->cols(), 1.0, D_last->ref(), 0, 0, F->ref(), 0, 0, F->ref(), 0, 0);
                }

                if (sub_blocks[index].H) {
                    assert(index > 0 && index < segments.size() - 1);
                    assert(!pivots.empty() && !sub_blocks[index].Bt.empty() && sub_blocks[index].F);
                    // H = -F * B[-1]
                    std::unique_ptr<BlasfeoMat>& Bt_last = sub_blocks[index].Bt.back();
                    std::unique_ptr<BlasfeoMat>& F = sub_blocks[index].F;
                    std::unique_ptr<BlasfeoMat>& H = sub_blocks[index].H;
                    assert(F->cols() == Bt_last->cols() && H->rows() == F->rows() && H->cols() == Bt_last->rows() && "size mismatch");
                    blasfeo_dgemm_nt(-1.0, *F, *Bt_last, 1.0, *H, *H);
                }

            }

#ifdef PIQP_HAS_OPENMP
#pragma omp single
            {
#endif

                // Phase 2, sequential
                {
                    PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::factor_kkt::phase_2");
                    for (size_t k = 1; k < segments.size(); k++) {
                        std::unique_ptr<BlasfeoMat> &F_km1 = sub_blocks[k - 1].F;
                        std::unique_ptr<BlasfeoMat> &A_k = sub_blocks[k].A;
                        std::unique_ptr<BlasfeoMat> &H_k = sub_blocks[k].H;
                        if (F_km1) {
                            // A[k] -= F[k-1] * F[k-1]^T, notice that rows of F[k-1] might be smaller than rows of A[k]
                            assert(A_k->rows() >= F_km1->rows() && "size mismatch");
                            blasfeo_dsyrk_ln(F_km1->rows(), F_km1->cols(), -1.0, F_km1->ref(), 0, 0, F_km1->ref(), 0, 0, 1.0,
                                             A_k->ref(), 0, 0, A_k->ref(), 0, 0);
                        }
                        // A[k] = chol(A[k])
                        if (A_k) {
                            blasfeo_dpotrf_l(*A_k);
                        }

                        if (H_k) {
                            assert(k < segments.size() - 1);
                            // H[k] = H[k] * A[k]^-T
                            assert(H_k->cols() == A_k->cols() && "size mismatch");
                            blasfeo_dtrsm_rltn(H_k->rows(), H_k->cols(), 1.0, A_k->ref(), 0, 0, H_k->ref(), 0, 0, H_k->ref(), 0,
                                               0);
                            // A[k+1] -= H[k] * H[k]^T, notice that rows of H[i] might be smaller than rows of A[i+1]
                            std::unique_ptr<BlasfeoMat> &A_kp1 = sub_blocks[k + 1].A;
                            assert(A_kp1->rows() >= H_k->rows() && "size mismatch");
                            blasfeo_dsyrk_ln(H_k->rows(), H_k->cols(), -1.0, H_k->ref(), 0, 0, H_k->ref(), 0, 0, 1.0,
                                             A_kp1->ref(),
                                             0, 0, A_kp1->ref(), 0, 0);
                        }

                        if (arrow_width > 0) {
                            std::unique_ptr<BlasfeoMat>& Q_k = sub_blocks[k].Q;
                            std::unique_ptr<BlasfeoMat>& G_km1_last = sub_blocks[k-1].G.back();
                            std::unique_ptr<BlasfeoMat>& R_k = sub_blocks[k].R;
                            // Q[k] -= G[k-1,-1] * F[k-1]^T
                            if (G_km1_last && F_km1 && Q_k) {
                                assert(G_km1_last->cols() == F_km1->cols() && "size mismatch");
                                blasfeo_dgemm_nt(-1.0, *G_km1_last, *F_km1, 1.0, *Q_k, *Q_k);
                            }

                            // Q[k] = Q[k] * A[k]^-T
                            if (Q_k && A_k) {
                                assert(Q_k->cols() == A_k->cols() && "size mismatch");
                                blasfeo_dtrsm_rltn(Q_k->rows(), Q_k->cols(), 1.0, A_k->ref(), 0, 0, Q_k->ref(), 0, 0, Q_k->ref(), 0, 0);
                                assert(!Q_k->hasNan() && "matrix has NaN values");
                            }

                            // R[k] -= Q[k] * Q[k]^T
                            if (Q_k) {
                                assert(R_k->rows() == Q_k->rows() && "size mismatch");
                                blasfeo_dsyrk_ln(Q_k->rows(), Q_k->cols(), -1.0, Q_k->ref(), 0, 0, Q_k->ref(), 0, 0, 1.0,
                                                 R_k->ref(), 0, 0, R_k->ref(), 0, 0);
                                assert(!R_k->hasNan() && "matrix has NaN values");
                            }

                            if (Q_k && H_k) {
                                assert(k < segments.size() - 1 && H_k->cols() == Q_k->cols() && "size mismatch");
                                assert(H_k->cols() == Q_k->cols() && "size mismatch");
                                // Q[k+1] -= Q[k] * H[k]^T
                                std::unique_ptr<BlasfeoMat>& Q_kp1 = sub_blocks[k+1].Q;
                                blasfeo_dgemm_nt(-1.0, *Q_k, *H_k, 1.0, *Q_kp1, *Q_kp1);
                            }
                        }
                    }

                    if (arrow_width > 0) {
                        // R = sum(R_k), add all R_k onto R_1
                        for (size_t j = 1; j < segments.size(); ++j) {
                            blasfeo_dgead(1.0, *sub_blocks[j].R, *sub_blocks[0].R);
                            assert(!sub_blocks[0].R->hasNan() && "matrix has NaN values");
                        }
                        // chol(R)
                        blasfeo_dpotrf_l(*sub_blocks[0].R);
                        assert(!sub_blocks[0].R->hasNan() && "matrix has NaN values");
                    }


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
            const I arrow_width = this->block_info.back().diag_size;

#ifdef PIQP_HAS_OPENMP
#pragma omp for
#endif
            for (size_t k = 0; k < segments.size(); k++) {
                PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::solve_llt_in_place:forward:segments");
                PIQP_TRACY_ZoneValue(k);
                // y_1 = D_1^{-1} * b_1
                auto& vec = b_and_x.x[segments[k][0]];
                assert(vec.rows() == sub_blocks[k].D[0]->rows() && "size mismatch");
                blasfeo_dtrsv_lnn(*sub_blocks[k].D[0], vec, vec);
                // rhs - B_1^T * b_1
                if (!sub_blocks[k].Bt.empty()) {
                    assert(k > 0);
                    const auto& Bt_0 = sub_blocks[k].Bt[0];
                    assert(vec.rows() == Bt_0->cols() && "size mismatch");
                    assert(b_and_x.x[pivots[k-1]].rows() == Bt_0->rows() && "size mismatch");
                    blasfeo_dgemv_n(-1.0, *Bt_0, vec, 1.0, b_and_x.x[pivots[k-1]], b_and_x.x[pivots[k-1]]);
                }
                assert(!vec.hasNan() && "vector has NaN values");

                if (arrow_width > 0) {
                    // y_g -= G_1 * y_1
                    auto& vec_g = b_and_x.x.back();
                    const auto& G_0 = sub_blocks[k].G[0];

                    T scaling = static_cast<T>(static_cast<T>(1.0) / static_cast<T>(segments.size()));
                    blasfeo_dgemv_n(-1.0, *G_0, vec, scaling, vec_g, *work_rhs_g[k]);
                }

                for (size_t i = 1; i < segments[k].size(); i++) {
                    // y2 = D_2^{-1} * (b_2 - E_1 * y_1)
                    const auto& E_im1 = sub_blocks[k].E[i-1];
                    const auto& D_i = sub_blocks[k].D[i];
                    auto& vec_i = b_and_x.x[segments[k][i]];
                    auto& vec_im1 = b_and_x.x[segments[k][i-1]];
                    assert(vec_im1.rows() == E_im1->cols() && "size mismatch");
                    blasfeo_dgemv_n(-1.0, *E_im1, vec_im1, 1.0, vec_i, vec_i);
                    assert(!vec_im1.hasNan() && "vector has NaN values");
                    blasfeo_dtrsv_lnn(*D_i, vec_i, vec_i);

                    // rhs - B_1^T * b_1
                    if (!sub_blocks[k].Bt.empty()) {
                        assert(k > 0);
                        const auto& Bt_i = sub_blocks[k].Bt[i];
                        assert(vec_i.rows() == Bt_i->cols() && "size mismatch");
                        assert(b_and_x.x[pivots[k-1]].rows() == Bt_i->rows() && "size mismatch");
                        blasfeo_dgemv_n(-1.0, *Bt_i, vec_i, 1.0, b_and_x.x[pivots[k-1]], b_and_x.x[pivots[k-1]]);
                    }
                    assert(!vec_i.hasNan() && "vector has NaN values");

                    if (arrow_width > 0) {
                        const auto& G_i = sub_blocks[k].G[i];
                        blasfeo_dgemv_n(-1.0, *G_i, vec_i, 1.0, *work_rhs_g[k], *work_rhs_g[k]);
                    }
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

                if (arrow_width > 0) {
                    // get back vec_g
                    b_and_x.x.back().setZero();
                    for (size_t i = 0; i < segments.size(); i++) {
                        blasfeo_dvecad(b_and_x.x.back().rows(), 1.0, work_rhs_g[i]->ref(), 0, b_and_x.x.back().ref(), 0);
                    }
                }

                for (size_t k = 1; k < segments.size(); k++) {
                    // r[k] -= F[k-1] * r[k-1][-1]
                    const std::unique_ptr<BlasfeoMat> &F_km1 = sub_blocks[k - 1].F;
                    if (F_km1) {
                        assert(!pivots.empty() && k < segments.size());
                        BlasfeoVec& r_km1_last = b_and_x.x[segments[k - 1].back()];
                        BlasfeoVec& r_k = b_and_x.x[pivots[k - 1]];
                        assert(r_k.rows() >= F_km1->rows() && "size mismatch");  // r[k] might have more rows than F[k-1]
                        assert(r_km1_last.rows() == F_km1->cols() && "size mismatch");
                        blasfeo_dgemv_n(-1.0, *F_km1, r_km1_last, 1.0, r_k, r_k);
                    }

                    // r[k+1] -= H[k] * r[k]
                    const std::unique_ptr<BlasfeoMat> &H_k = sub_blocks[k - 1].H;
                    if (H_k) {
                        assert(!pivots.empty() && k > 1);
                        BlasfeoVec& r_k = b_and_x.x[pivots[k - 2]];
                        BlasfeoVec& r_kp1 = b_and_x.x[pivots[k - 1]];
                        assert(r_k.rows() == H_k->cols() && "size mismatch");
                        assert(r_kp1.rows() >= H_k->rows() && "size mismatch");  // r[k+1] might have more rows than H[k-1]
                        blasfeo_dgemv_n(-1.0, *H_k, r_k, 1.0, r_kp1, r_kp1);
                    }

                    // r[k] = A[k]^-1 * r[k]
                    const std::unique_ptr<BlasfeoMat> &A_k = sub_blocks[k].A;
                    if (A_k) {
                        BlasfeoVec& r_k = b_and_x.x[pivots[k - 1]];
                        assert(r_k.rows() == A_k->rows() && "size mismatch");
                        blasfeo_dtrsv_lnn(*A_k, r_k, r_k);
                        assert(!r_k.hasNan() && "vector has NaN values");
                    }

                    // r[-1] -= Q[k] * r[k]
                    const auto& Q_k = sub_blocks[k].Q;
                    if (arrow_width > 0 && Q_k) {
                        BlasfeoVec& r_k = b_and_x.x[pivots[k - 1]];
                        BlasfeoVec& r_g = b_and_x.x.back();
                        blasfeo_dgemv_n(-1.0, *Q_k, r_k, 1.0, r_g, r_g);
                    }
                }

                if (arrow_width > 0) {
                    assert(sub_blocks[0].R);
                    // r_g = R^-1 * r_g
                    BlasfeoVec& vec_g = b_and_x.x.back();
                    blasfeo_dtrsv_lnn(*sub_blocks[0].R, vec_g, vec_g);
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
            const I arrow_width = this->block_info.back().diag_size;

#ifdef PIQP_HAS_OPENMP
#pragma omp master
        {
#endif

            {
                PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::solve_llt_in_place:backward:pivots");

                if (arrow_width > 0) {
                    // r_g = R^-T * r_g
                    BlasfeoVec& r_g = b_and_x.x.back();
                    if (sub_blocks[0].R) {
                        blasfeo_dtrsv_ltn(*sub_blocks[0].R, r_g, r_g);
                    }

                    if (sub_blocks.back().Q) {
                        assert(!pivots.empty());
                        // r_p -= Q[-1]^T * r_g
                        BlasfeoVec& r_p = b_and_x.x[pivots.back()];
                        blasfeo_dgemv_t(-1.0, *sub_blocks.back().Q, r_g, 1.0, r_p, r_p);
                    }
                }

                if (sub_blocks.back().A) {
                    // r_p = A_p^-T * r_p
                    assert(sub_blocks.back().A->rows() == b_and_x.x[pivots.back()].rows() && "size mismatch");
                    blasfeo_dtrsv_ltn(*sub_blocks.back().A, b_and_x.x[pivots.back()], b_and_x.x[pivots.back()]);
                }

                if (pivots.size() >= 1) {
                    for (size_t k = pivots.size() - 2; k != SIZE_MAX; k--) {
                        if (arrow_width > 0) {
                            // r_k -= Q[k]^T * r_g
                            auto& vec_g = b_and_x.x.back();
                            blasfeo_dgemv_t(-1.0, *sub_blocks[k+1].Q, vec_g, 1.0, b_and_x.x[pivots[k]], b_and_x.x[pivots[k]]);
                        }
                        // r[k] -= H[k]^T * r[k+1]
                        assert(sub_blocks[k + 1].H->rows() <= b_and_x.x[pivots[k + 1]].rows() && "size mismatch");
                        assert(sub_blocks[k + 1].H->cols() == b_and_x.x[pivots[k]].rows() && "size mismatch");
                        // blasfeo_dgemv_t(-1.0, *sub_blocks[k + 1].H, b_and_x.x[pivots[k + 1]], 1.0, b_and_x.x[pivots[k]], b_and_x.x[pivots[k]]);
                        // NOTICE that if the original off-diagonal block B[i] has less rows than D[i+1], then H[k] will
                        // also have less rows than A[k+1]. This will cause H[k].T to have less cols than r[k+1]
                        blasfeo_dgemv_t(sub_blocks[k + 1].H->rows(), sub_blocks[k + 1].H->cols(), -1.0,
                                sub_blocks[k + 1].H->ref(), 0, 0, b_and_x.x[pivots[k + 1]].ref(), 0, 1.0,
                                b_and_x.x[pivots[k]].ref(), 0, b_and_x.x[pivots[k]].ref(), 0);
                        // r[k] = A[k]^-T * r[k]
                        assert(sub_blocks[k + 1].A->rows() == b_and_x.x[pivots[k]].rows() && "size mismatch");
                        blasfeo_dtrsv_ltn(*sub_blocks[k+1].A, b_and_x.x[pivots[k]], b_and_x.x[pivots[k]]);
                    }
                }
            }

#ifdef PIQP_HAS_OPENMP
            }
#pragma omp barrier
#pragma omp for
#endif
            for (size_t k = 0; k < segments.size(); k++) {
                PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::solve_llt_in_place:backward:segments");
                PIQP_TRACY_ZoneValue(k);
                if (k > 0) {
                    if (!sub_blocks[k].Bt.empty()) {
                        for (size_t i = segments[k].size() - 1; i != SIZE_MAX; i--) {
                            const std::unique_ptr<BlasfeoMat>& Bt_i = sub_blocks[k].Bt[i];
                            assert(Bt_i->rows() == b_and_x.x[pivots[k-1]].rows() && "size mismatch");
                            assert(Bt_i->cols() == b_and_x.x[segments[k][i]].rows() && "size mismatch");
                            blasfeo_dgemv_t(-1.0, *Bt_i, b_and_x.x[pivots[k-1]], 1.0, b_and_x.x[segments[k][i]], b_and_x.x[segments[k][i]]);
                        }
                    }
                }

                if (arrow_width > 0) {
                    // r_i -= G_i^T * r_g
                    auto& vec_g = b_and_x.x.back();
                    for (size_t i = segments[k].size() - 1; i != SIZE_MAX; i--) {
                        const std::unique_ptr<BlasfeoMat>& G_i = sub_blocks[k].G[i];
                        if (G_i) {
                            assert(G_i->rows() == vec_g.rows() && "size mismatch");
                            assert(G_i->cols() <= b_and_x.x[segments[k][i]].rows() && "size mismatch");
                            blasfeo_dgemv_t(-1.0, *G_i, vec_g, 1.0, b_and_x.x[segments[k][i]], b_and_x.x[segments[k][i]]);
                        }
                    }
                }

                if (k < segments.size() - 1) {
                    if (sub_blocks[k].F) {
                        assert(sub_blocks[k].F->rows() <= b_and_x.x[pivots[k]].rows() && "size mismatch");
                        assert(sub_blocks[k].F->cols() == b_and_x.x[segments[k].back()].rows() && "size mismatch");
                        blasfeo_dgemv_t(sub_blocks[k].F->rows(), sub_blocks[k].F->cols(), -1.0, sub_blocks[k].F->ref(), 0, 0, b_and_x.x[pivots[k]].ref(), 0, 1.0, b_and_x.x[segments[k].back()].ref(), 0, b_and_x.x[segments[k].back()].ref(), 0);
                    }
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

