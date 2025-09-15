

#ifndef PIQP_PERMUTED_BLOCK_KKT_H
#define PIQP_PERMUTED_BLOCK_KKT_H

#ifdef PIQP_HAS_OPENMP
#include "omp.h"
#endif

#include <vector>
#include "piqp/utils/blasfeo_mat.hpp"
#include "piqp/sparse/blocksparse/block_kkt.hpp"

namespace piqp {
    namespace sparse {

        struct alignas(64) SubBlockKKTParallel {
            size_t index = 0;
            std::vector<std::unique_ptr<BlasfeoMat>> D; // lower triangular diagonal
            std::vector<std::unique_ptr<BlasfeoMat>> E; // off diagonal
            std::vector<std::unique_ptr<BlasfeoMat>> B; //
            std::unique_ptr<BlasfeoMat> F;
            std::unique_ptr<BlasfeoMat> A;
            std::unique_ptr<BlasfeoMat> H;

            SubBlockKKTParallel() = default;

            SubBlockKKTParallel(const BlockKKT& kkt, const std::vector<size_t>& pivots, const std::vector<std::vector<size_t>>& segments, const size_t index): index(index) {
                const std::vector<size_t>& segment = segments[index];
                // D
                D.clear(); D.resize(segment.size());
                for (size_t i = 0; i < segment.size(); i++) {
                    D[i] = std::make_unique<BlasfeoMat>(*kkt.D[segment[i]]);
                }

                // E
                E.clear(); E.resize(segment.size()-1);
                for (size_t i = 0; i < segment.size() - 1; i++) {
                    E[i] = std::make_unique<BlasfeoMat>(*kkt.B[segment[i]]);
                }

                // F
                if (index < segments.size() - 1) {
                    F = std::make_unique<BlasfeoMat>(*kkt.B[pivots[index]-1]);
                } else {
                    F = nullptr;  // The last sub-block does not have an F matrix
                }

                // A
                if (index > 0) {
                    A = std::make_unique<BlasfeoMat>(*kkt.D[pivots[index-1]]);
                } else {
                    A = nullptr;  // The first sub-block does not have an A matrix
                }

                // H
                if (index > 0 && index < segments.size() - 1) {
                    BlasfeoMat mat(F->rows(), A->cols());
                    mat.setZero();
                    H = std::make_unique<BlasfeoMat>(mat);
                } else {
                    H = nullptr;  // The first and last sub-blocks do not have an H matrix
                }

                // B
                B.clear();
                if (index > 0) {
                    B.resize(segment.size());
                    // ! B[i] must have the size (nx[i+1], nx[i]) !!!  Cannot use the size of offdiagonal matrix in the original KKT!
                    B[0] = std::make_unique<BlasfeoMat>(D[0]->cols(), A->rows());
                    B[0]->setZero();
                    blasfeo_dgecp(kkt.B[pivots[index-1]]->rows(), kkt.B[pivots[index-1]]->cols(), kkt.B[pivots[index-1]]->ref(), 0, 0, B[0]->ref(), 0, 0);
                    for (size_t i = 1; i < segment.size(); i++) {
                        B[i] = std::make_unique<BlasfeoMat>(D[i]->cols(), B[i-1]->cols());
                        B[i]->setZero();
                    }
                } else {
                    B.resize(0);
                }
            }


            SubBlockKKTParallel(SubBlockKKTParallel&&) = default;

            SubBlockKKTParallel(const SubBlockKKTParallel& other)
            {
                index = other.index;
                D.resize(other.D.size());
                E.resize(other.E.size());
                B.resize(other.B.size());
                F = std::make_unique<BlasfeoMat>(*other.F);
                A = std::make_unique<BlasfeoMat>(*other.A);
                H = std::make_unique<BlasfeoMat>(*other.H);

                for (std::size_t i = 0; i < other.D.size(); i++) {
                    if (other.D[i]) {
                         D[i] = std::make_unique<BlasfeoMat>(*other.D[i]);
                    }
                }
                for (std::size_t i = 0; i < other.E.size(); i++) {
                    if (other.E[i]) {
                        E[i] = std::make_unique<BlasfeoMat>(*other.E[i]);
                    }
                }
                for (std::size_t i = 0; i < other.B.size(); i++) {
                    if (other.B[i]) {
                        B[i] = std::make_unique<BlasfeoMat>(*other.B[i]);
                    }
                }


            }

            SubBlockKKTParallel& operator=(SubBlockKKTParallel&&) = default;

            SubBlockKKTParallel& operator=(const SubBlockKKTParallel& other)
            {
                if (this == &other) return *this;

                index = other.index;

                D.clear(); D.resize(other.D.size());
                for (std::size_t i = 0; i < other.D.size(); i++) {
                    if (other.D[i]) {
                        if (!D[i]) {
                            D[i] = std::make_unique<BlasfeoMat>(*other.D[i]);
                        } else {
                            *D[i] = *other.D[i];
                        }
                    } else {
                        D[i] = nullptr;
                    }
                }

                B.clear(); B.resize(other.B.size());
                for (std::size_t i = 0; i < other.B.size(); i++) {
                    if (other.B[i]) {
                        if (!B[i]) {
                            B[i] = std::make_unique<BlasfeoMat>(*other.B[i]);
                        } else {
                            *B[i] = *other.B[i];
                        }
                    } else {
                        B[i] = nullptr;
                    }
                }

                E.clear(); E.resize(other.E.size());
                for (std::size_t i = 0; i < other.E.size(); i++) {
                    if (other.E[i]) {
                        if (!E[i]) {
                            E[i] = std::make_unique<BlasfeoMat>(*other.E[i]);
                        } else {
                            *E[i] = *other.E[i];
                        }
                    } else {
                        E[i] = nullptr;
                    }
                }

                if (other.F) {
                    if (!F) {
                        F = std::make_unique<BlasfeoMat>(*other.F);
                    } else {
                        *F = *other.F;
                    }
                } else {
                    F = nullptr;
                }

                if (other.A) {
                    if (!A) {
                        A = std::make_unique<BlasfeoMat>(*other.A);
                    } else {
                        *A = *other.A;
                    }
                } else {
                    A = nullptr;
                }

                if (other.H) {
                    if (!H) {
                        H = std::make_unique<BlasfeoMat>(*other.H);
                    } else {
                        *H = *other.H;
                    }
                } else {
                    H = nullptr;
                }

                return *this;
            }

        };

        // stores the lower triangular data of a permuted arrow KKT structure
        struct BlockKKTParallel {
            size_t num_threads = 1;
            std::vector<SubBlockKKTParallel> sub_blocks; // stores the permuted sub-blocks
            std::vector<size_t> pivots;
            std::vector<std::vector<size_t>> segments;

            BlockKKTParallel() = default;

        };
    }
}

#endif //PIQP_PERMUTED_BLOCK_KKT_H
