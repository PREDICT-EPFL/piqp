// This file is part of PIQP.
//
// Copyright (c) 2026 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_MULTISTAGE_PARTITION_HPP
#define PIQP_SPARSE_MULTISTAGE_PARTITION_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <vector>

namespace piqp
{

namespace sparse
{

// Partitioning of the multistage KKT system for the parallel factorization
// ========================================================================
//
// Problem
// -------
// The KKT matrix consists of a chain of M diagonal stage blocks 0, ..., M-1, where block i is coupled to
// block i+1 through an off-diagonal block (plus an arrow coupling all blocks to the global variable, which
// is ignored here). Block i is called decoupled if its off-diagonal block is zero, e.g., at the end of a
// scenario in scenario MPC.
//
// The blocks are split into at most P contiguous segments, which are factorized in parallel, one per thread.
// Two consecutive segments are either
//   * split freely, which is only possible after a decoupled block, or
//   * separated by a single separator block. The separator is permuted to the end of the matrix
//     and factorized sequentially in the reduced system after all segments have been factorized.
//     A separator has to be coupled to both neighbouring segments.
//
//   blocks:   [ 0 1 2 3 4 ][ 5 ][ 6 7 8 ] || [ 9 10 11 12 ]
//               segment 0   sep   seg. 1  free   segment 2
//
// Cost model
// ----------
// Each block i has a weight w_i (e.g. b_i^3 for a block of size b_i). Following the flop counts in
// https://arxiv.org/abs/2511.00946, the factorization of a segment S costs
//
//   cost(S) = c_S * sum_{i in S} w_i,   c_S = 7/3   if S is not preceded by a separator,
//                                       c_S = 19/3  if S is preceded by a separator,
//
// since a segment preceded by a separator gets coupled to it, which introduces a dense fill-in block that
// has to be propagated through the whole segment. Each separator j additionally costs 10/3 * w_j in the
// sequential reduced system. The partition minimizes the estimated factorization time
//
//   J = max_k cost(S_k) + 10/3 * sum_{separators j} w_j.
//
// Note that this model can trade off threads: if separators are too expensive, fewer segments are used.
//
// Algorithm
// ---------
// A direct dynamic program over (number of segments, last block, number of separators) costs O(P^2 M^2),
// which is too slow for long horizons. Instead, the problem is split into an outer search over the
// bottleneck cost T and an inner feasibility problem which is linear in M.
//
// 1. Feasibility for a fixed bound T: let h(k, j) be the minimum number of separators needed to cover the
//    blocks [0, j) with k segments, each with cost <= T. A segment [a, e) is appended to a state as
//      * free start:            from h(k, a) if a == 0 or block a-1 is decoupled, with factor 7/3,
//      * start after separator: from h(k, a-1) + 1 if block a-1 can be a separator, with factor 19/3.
//    For a fixed segment end e and factor c, the feasible starts c * W(a, e) <= T form an interval
//    [l_c(e), e-1], where W(a, e) = sum_{i=a}^{e-1} w_i is evaluated using prefix sums. Since l_c(e) is
//    non-decreasing in e, the minimum over the interval is a sliding window minimum, which is computed
//    with a monotone queue in amortized O(1). Hence, all h(k, .) for k = 1, ..., P are computed in O(P M),
//    and min_k h(k, M) is the minimum number of separators needed to achieve the bound T.
//
// 2. Outer search: for each separator budget s = 0, ..., P-1, a bisection finds the smallest bound T_s for
//    which s separators suffice. T_s is non-increasing in s, so the bisection for s starts from T_{s-1}.
//    The partition for T_s is reconstructed from back pointers stored during the feasibility check,
//    using the fewest segments achieving the fewest separators. The partition with the smallest J over
//    all budgets is returned.
//
// The overall complexity is O(P^2 M log(1/rel_tol)). If all weights are equal, the separator cost only
// depends on the number of separators and the result is optimal up to rel_tol. Otherwise, the separator
// cost depends on which blocks become separators, which the feasibility check does not distinguish, and
// the result is optimal up to the differences in the separator weights.

// Cost model of the parallel block-tridiagonal factorization, see https://arxiv.org/abs/2511.00946.
// All costs are per unit of block weight (e.g. b^3 for a block of size b).
struct MultistagePartitionCosts
{
    // factorization of a segment which is not preceded by a separator
    double segment = 7.0 / 3.0;
    // factorization of a segment which is preceded by a separator (includes the fill-in)
    double segment_after_separator = 19.0 / 3.0;
    // sequential factorization of a separator in the reduced system
    double separator = 10.0 / 3.0;
};

struct MultistagePartition
{
    // contiguous block indices of each segment
    std::vector<std::vector<std::size_t>> segments;
    // separators[k] is the separator block preceding segment k, or -1 if there is none
    std::vector<std::ptrdiff_t> separators;

    bool has_separator_before(std::size_t k) const { return separators[k] >= 0; }

    std::size_t separator_before(std::size_t k) const
    {
        assert(has_separator_before(k));
        return static_cast<std::size_t>(separators[k]);
    }
};

namespace detail
{

constexpr std::size_t partition_inf = (std::numeric_limits<std::size_t>::max)();

class MultistagePartitioner
{
private:
    struct Candidate
    {
        std::size_t start;
        std::size_t value;
    };

    // sliding window minimum over segment starts a in [l(e), e - 1] for segment [a, e),
    // candidates are stored in a monotone queue with increasing values
    struct Window
    {
        double c;
        std::size_t l = 0;
        std::vector<Candidate> q;
        std::size_t head = 0;
        std::size_t tail = 0;

        void reset(double c_, std::size_t capacity)
        {
            c = c_;
            l = 0;
            head = 0;
            tail = 0;
            q.resize(capacity);
        }

        bool empty() const { return head == tail; }

        const Candidate& front() const { return q[head]; }

        void push(std::size_t start, std::size_t value)
        {
            if (value == partition_inf) return;
            while (!empty() && q[tail - 1].value >= value) tail--;
            q[tail++] = {start, value};
        }

        void advance(const std::vector<double>& prefix, std::size_t e, double T)
        {
            while (l < e && c * (prefix[e] - prefix[l]) > T) l++;
            while (!empty() && q[head].start < l) head++;
        }
    };

    std::size_t M;
    std::size_t P;
    MultistagePartitionCosts costs;
    std::vector<double> prefix;
    std::vector<bool> free_start;
    std::vector<bool> separator_allowed;

    // h(k, j): minimum number of separators to cover blocks [0, j) with k segments of cost <= T
    std::vector<std::vector<std::size_t>> h;
    // start block and kind of the last segment of the optimal path to (k, j)
    std::vector<std::vector<std::size_t>> back_start;
    std::vector<std::vector<bool>> back_separator;

    Window free_window;
    Window separator_window;

public:
    MultistagePartitioner(const std::vector<double>& weights, const std::vector<bool>& decoupled,
                          std::size_t max_segments, const MultistagePartitionCosts& costs)
        : M(weights.size()), P(max_segments), costs(costs), prefix(M + 1, 0.0), free_start(M + 1, false), separator_allowed(M, false)
    {
        assert(decoupled.size() == M);
        for (std::size_t i = 0; i < M; i++) {
            prefix[i + 1] = prefix[i] + weights[i];
        }
        // a segment can start without separator at block a if block a - 1 is decoupled from block a
        free_start[0] = true;
        for (std::size_t a = 1; a <= M; a++) {
            free_start[a] = decoupled[a - 1];
        }
        // a separator has to be coupled to both neighbouring segments
        for (std::size_t j = 1; j + 1 < M; j++) {
            separator_allowed[j] = !decoupled[j - 1] && !decoupled[j];
        }
        h.assign(P + 1, std::vector<std::size_t>(M + 1, partition_inf));
        back_start.assign(P + 1, std::vector<std::size_t>(M + 1, 0));
        back_separator.assign(P + 1, std::vector<bool>(M + 1, false));
    }

    // Minimizes max_k segment_cost_k + separator cost.
    // Exact (up to rel_tol) if all weights are equal.
    MultistagePartition solve(double rel_tol)
    {
        MultistagePartition best;
        if (M == 0 || P == 0) return best;

        std::size_t max_separators = static_cast<std::size_t>(std::count(separator_allowed.begin(), separator_allowed.end(), true));
        max_separators = (std::min)(max_separators, P - 1);

        double best_cost = std::numeric_limits<double>::infinity();
        // a single segment is always feasible without separators
        double hi = costs.segment * prefix[M];
        for (std::size_t s = 0; s <= max_separators; s++) {
            // smallest bound T such that a partition with at most s separators exists,
            // hi is feasible for s since it is feasible for s - 1
            double lo = 0;
            while (hi - lo > rel_tol * hi) {
                double mid = 0.5 * (lo + hi);
                if (min_separators(mid) <= s) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }

            MultistagePartition partition = reconstruct(hi);
            double cost = evaluate(partition);
            if (cost < best_cost) {
                best_cost = cost;
                best = std::move(partition);
            }
        }

        return best;
    }

    double evaluate(const MultistagePartition& partition) const
    {
        double max_segment_cost = 0;
        double separator_cost = 0;
        for (std::size_t k = 0; k < partition.segments.size(); k++) {
            const auto& segment = partition.segments[k];
            double c = partition.has_separator_before(k) ? costs.segment_after_separator : costs.segment;
            max_segment_cost = (std::max)(max_segment_cost, c * (prefix[segment.back() + 1] - prefix[segment.front()]));
            if (partition.has_separator_before(k)) {
                std::size_t j = partition.separator_before(k);
                separator_cost += costs.separator * (prefix[j + 1] - prefix[j]);
            }
        }
        return max_segment_cost + separator_cost;
    }

private:
    // fills h and the back pointers for the bound T, returns min_k h(k, M)
    std::size_t min_separators(double T)
    {
        for (auto& h_k : h) {
            std::fill(h_k.begin(), h_k.end(), partition_inf);
        }
        h[0][0] = 0;

        std::size_t best = partition_inf;
        for (std::size_t k = 0; k < P; k++) {
            free_window.reset(costs.segment, M);
            separator_window.reset(costs.segment_after_separator, M);
            for (std::size_t e = 1; e <= M; e++) {
                // add segment start a = e - 1
                std::size_t a = e - 1;
                if (free_start[a]) {
                    free_window.push(a, h[k][a]);
                }
                if (a >= 1 && separator_allowed[a - 1] && h[k][a - 1] != partition_inf) {
                    separator_window.push(a, h[k][a - 1] + 1);
                }
                free_window.advance(prefix, e, T);
                separator_window.advance(prefix, e, T);

                std::size_t& h_next = h[k + 1][e];
                if (!free_window.empty()) {
                    h_next = free_window.front().value;
                    back_start[k + 1][e] = free_window.front().start;
                    back_separator[k + 1][e] = false;
                }
                if (!separator_window.empty() && separator_window.front().value < h_next) {
                    h_next = separator_window.front().value;
                    back_start[k + 1][e] = separator_window.front().start;
                    back_separator[k + 1][e] = true;
                }
            }
            best = (std::min)(best, h[k + 1][M]);
        }
        return best;
    }

    MultistagePartition reconstruct(double T)
    {
        std::size_t s = min_separators(T);
        assert(s != partition_inf);

        // use the fewest segments achieving the fewest separators
        std::size_t k = 1;
        while (h[k][M] != s) k++;

        MultistagePartition partition;
        partition.segments.resize(k);
        partition.separators.resize(k, -1);
        std::size_t e = M;
        while (k > 0) {
            std::size_t a = back_start[k][e];
            bool separator = back_separator[k][e];
            for (std::size_t i = a; i < e; i++) {
                partition.segments[k - 1].push_back(i);
            }
            if (separator) {
                partition.separators[k - 1] = static_cast<std::ptrdiff_t>(a - 1);
                e = a - 1;
            } else {
                e = a;
            }
            k--;
        }
        assert(e == 0);

        return partition;
    }
};

} // namespace detail

// Partitions a chain of M blocks into at most max_segments contiguous segments which are factorized in parallel.
// Consecutive segments are either split at a decoupled block (decoupled[i] == true if block i is not coupled
// to block i + 1), or are separated by a separator block, which is factorized sequentially afterwards.
// The partition minimizes the maximum segment cost plus the sequential separator cost according to costs.
// The result is optimal up to rel_tol if all weights are equal.
inline MultistagePartition partition_multistage(const std::vector<double>& weights, const std::vector<bool>& decoupled,
                                                std::size_t max_segments,
                                                const MultistagePartitionCosts& costs = MultistagePartitionCosts(),
                                                double rel_tol = 1e-6)
{
    detail::MultistagePartitioner partitioner(weights, decoupled, max_segments, costs);
    return partitioner.solve(rel_tol);
}

// Cost of a partition according to the cost model used in partition_multistage.
inline double multistage_partition_cost(const MultistagePartition& partition, const std::vector<double>& weights,
                                        const std::vector<bool>& decoupled,
                                        const MultistagePartitionCosts& costs = MultistagePartitionCosts())
{
    detail::MultistagePartitioner partitioner(weights, decoupled, 1, costs);
    return partitioner.evaluate(partition);
}

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_MULTISTAGE_PARTITION_HPP
