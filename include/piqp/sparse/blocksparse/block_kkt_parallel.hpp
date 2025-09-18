

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
            std::vector<std::unique_ptr<BlasfeoMat>> G;
            std::unique_ptr<BlasfeoMat> Q;
            std::unique_ptr<BlasfeoMat> R;

            SubBlockKKTParallel() = default;

            SubBlockKKTParallel(SubBlockKKTParallel&&) = default;

            SubBlockKKTParallel(const SubBlockKKTParallel& other)
            {
                index = other.index;
                D.resize(other.D.size());
                E.resize(other.E.size());
                B.resize(other.B.size());
                G.resize(other.G.size());
                F = std::make_unique<BlasfeoMat>(*other.F);
                A = std::make_unique<BlasfeoMat>(*other.A);
                H = std::make_unique<BlasfeoMat>(*other.H);
                Q = std::make_unique<BlasfeoMat>(*other.Q);
                R = std::make_unique<BlasfeoMat>(*other.R);

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
                for (std::size_t i = 0; i < other.G.size(); i++) {
                    if (other.G[i]) {
                        G[i] = std::make_unique<BlasfeoMat>(*other.G[i]);
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

                G.clear(); G.resize(other.G.size());
                for (std::size_t i = 0; i < other.G.size(); i++) {
                    if (other.G[i]) {
                        if (!G[i]) {
                            G[i] = std::make_unique<BlasfeoMat>(*other.G[i]);
                        } else {
                            *G[i] = *other.G[i];
                        }
                    } else {
                        G[i] = nullptr;
                    }
                }

                if (other.Q) {
                    if (!Q) {
                        Q = std::make_unique<BlasfeoMat>(*other.Q);
                    } else {
                        *Q = *other.Q;
                    }
                } else {
                    Q = nullptr;
                }

                if (other.R) {
                    if (!R) {
                        R = std::make_unique<BlasfeoMat>(*other.R);
                    } else {
                        *R = *other.R;
                    }
                } else {
                    R = nullptr;
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
