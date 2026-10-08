

#ifndef PIQP_PERMUTED_BLOCK_KKT_H
#define PIQP_PERMUTED_BLOCK_KKT_H

#ifdef PIQP_HAS_OPENMP
#include "omp.h"
#endif

#include <memory>
#include <vector>
#include "piqp/utils/blasfeo_mat.hpp"
#include "piqp/sparse/blocksparse/block_kkt.hpp"

namespace piqp {
    namespace sparse {

        struct alignas(64) SubBlockKKTParallel {
            size_t index = 0;
            std::vector<std::unique_ptr<BlasfeoMat>> D; // lower triangular diagonal
            std::vector<std::unique_ptr<BlasfeoMat>> E; // off diagonal
            std::vector<std::unique_ptr<BlasfeoMat>> Bt;
            std::unique_ptr<BlasfeoMat> Bt0_tmp;
            std::unique_ptr<BlasfeoMat> F;
            std::unique_ptr<BlasfeoMat> A;
            std::unique_ptr<BlasfeoMat> H;
            std::vector<std::unique_ptr<BlasfeoMat>> G;
            std::unique_ptr<BlasfeoMat> Q;
            std::unique_ptr<BlasfeoMat> R;

            SubBlockKKTParallel() = default;

            SubBlockKKTParallel(SubBlockKKTParallel&&) = default;

            SubBlockKKTParallel(const SubBlockKKTParallel& other)
            {
                *this = other;
            }

            SubBlockKKTParallel& operator=(SubBlockKKTParallel&&) = default;

            SubBlockKKTParallel& operator=(const SubBlockKKTParallel& other)
            {
                if (this == &other) return *this;

                index = other.index;
                assign_mats(D, other.D);
                assign_mats(E, other.E);
                assign_mats(Bt, other.Bt);
                assign_mat(Bt0_tmp, other.Bt0_tmp);
                assign_mat(F, other.F);
                assign_mat(A, other.A);
                assign_mat(H, other.H);
                assign_mats(G, other.G);
                assign_mat(Q, other.Q);
                assign_mat(R, other.R);

                return *this;
            }

        private:
            static void assign_mat(std::unique_ptr<BlasfeoMat>& dst, const std::unique_ptr<BlasfeoMat>& src)
            {
                if (!src) {
                    dst = nullptr;
                } else if (!dst) {
                    dst = std::make_unique<BlasfeoMat>(*src);
                } else {
                    *dst = *src;
                }
            }

            static void assign_mats(std::vector<std::unique_ptr<BlasfeoMat>>& dst, const std::vector<std::unique_ptr<BlasfeoMat>>& src)
            {
                dst.resize(src.size());
                for (std::size_t i = 0; i < src.size(); i++) {
                    assign_mat(dst[i], src[i]);
                }
            }
        };

        // stores the lower triangular data of a permuted arrow KKT structure
        struct BlockKKTParallel {
            size_t num_threads = 1;
            std::vector<SubBlockKKTParallel> sub_blocks; // stores the permuted sub-blocks

            BlockKKTParallel() = default;

        };
    }
}

#endif //PIQP_PERMUTED_BLOCK_KKT_H
