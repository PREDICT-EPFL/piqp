// This file is part of PIQP.
//
// Copyright (c) 2026 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

#include "piqp/sparse/multistage_partition.hpp"

#include "gtest/gtest.h"

using namespace piqp::sparse;

namespace
{

std::vector<bool> chain(std::size_t M)
{
    std::vector<bool> decoupled(M, false);
    decoupled[M - 1] = true;
    return decoupled;
}

std::vector<bool> scenarios(std::size_t num_scenarios, std::size_t scenario_length)
{
    std::vector<bool> decoupled(num_scenarios * scenario_length, false);
    for (std::size_t s = 0; s < num_scenarios; s++) {
        decoupled[(s + 1) * scenario_length - 1] = true;
    }
    return decoupled;
}

bool separator_allowed(const std::vector<bool>& decoupled, std::size_t j)
{
    return j >= 1 && j + 1 < decoupled.size() && !decoupled[j - 1] && !decoupled[j];
}

void check_valid(const MultistagePartition& partition, const std::vector<bool>& decoupled, std::size_t max_segments)
{
    const std::size_t M = decoupled.size();
    ASSERT_GE(partition.segments.size(), 1u);
    ASSERT_LE(partition.segments.size(), max_segments);
    ASSERT_EQ(partition.segments.size(), partition.separators.size());
    ASSERT_FALSE(partition.has_separator_before(0));

    std::size_t next = 0;
    for (std::size_t k = 0; k < partition.segments.size(); k++) {
        const auto& segment = partition.segments[k];
        ASSERT_FALSE(segment.empty());
        if (k > 0) {
            if (partition.has_separator_before(k)) {
                ASSERT_EQ(partition.separator_before(k), next);
                ASSERT_TRUE(separator_allowed(decoupled, next));
                next++;
            } else {
                ASSERT_TRUE(decoupled[next - 1]);
            }
        }
        for (std::size_t i : segment) {
            ASSERT_EQ(i, next);
            next++;
        }
    }
    ASSERT_EQ(next, M);
}

// recursively appends all valid segments starting at block start (preceded by separator, or -1)
// to the partial partition and updates best with the cost of each complete partition
void enumerate_partitions(const std::vector<double>& weights, const std::vector<bool>& decoupled, std::size_t max_segments,
                          std::size_t start, std::ptrdiff_t separator, MultistagePartition& partition, double& best)
{
    const std::size_t M = weights.size();
    if (partition.segments.size() == max_segments) return;
    for (std::size_t e = start + 1; e <= M; e++) {
        std::vector<std::size_t> segment;
        for (std::size_t i = start; i < e; i++) segment.push_back(i);
        partition.segments.push_back(segment);
        partition.separators.push_back(separator);
        if (e == M) {
            best = (std::min)(best, multistage_partition_cost(partition, weights, decoupled));
        } else {
            if (decoupled[e - 1]) {
                enumerate_partitions(weights, decoupled, max_segments, e, -1, partition, best);
            }
            if (separator_allowed(decoupled, e)) {
                enumerate_partitions(weights, decoupled, max_segments, e + 1, static_cast<std::ptrdiff_t>(e), partition, best);
            }
        }
        partition.segments.pop_back();
        partition.separators.pop_back();
    }
}

// enumerates all valid partitions and returns the minimal cost
double brute_force_cost(const std::vector<double>& weights, const std::vector<bool>& decoupled, std::size_t max_segments)
{
    double best = std::numeric_limits<double>::infinity();
    MultistagePartition partition;
    if (!weights.empty()) {
        enumerate_partitions(weights, decoupled, max_segments, 0, -1, partition, best);
    }
    return best;
}

std::vector<std::size_t> segment_sizes(const MultistagePartition& partition)
{
    std::vector<std::size_t> sizes;
    for (const auto& segment : partition.segments) sizes.push_back(segment.size());
    return sizes;
}

} // namespace

TEST(MultistagePartitionTest, SingleSegment)
{
    std::vector<double> weights(10, 1.0);
    MultistagePartition partition = partition_multistage(weights, chain(10), 1);
    check_valid(partition, chain(10), 1);
    ASSERT_EQ(segment_sizes(partition), std::vector<std::size_t>({10}));
}

TEST(MultistagePartitionTest, SingleBlock)
{
    std::vector<double> weights(1, 1.0);
    MultistagePartition partition = partition_multistage(weights, chain(1), 8);
    check_valid(partition, chain(1), 8);
    ASSERT_EQ(segment_sizes(partition), std::vector<std::size_t>({1}));
}

TEST(MultistagePartitionTest, ShortChainUsesFewerSegments)
{
    // separators are too expensive for such short chains
    for (std::size_t M = 1; M <= 4; M++) {
        std::vector<double> weights(M, 1.0);
        MultistagePartition partition = partition_multistage(weights, chain(M), 8);
        check_valid(partition, chain(M), 8);
        ASSERT_EQ(partition.segments.size(), 1u);
    }
}

TEST(MultistagePartitionTest, Chain)
{
    // first segment is longer to compensate for the fill-in in the other segments
    std::vector<double> weights(50, 1.0);
    MultistagePartition partition = partition_multistage(weights, chain(50), 4);
    check_valid(partition, chain(50), 4);
    ASSERT_EQ(segment_sizes(partition), std::vector<std::size_t>({23, 8, 8, 8}));
    ASSERT_EQ(partition.separators, std::vector<std::ptrdiff_t>({-1, 23, 32, 41}));
}

TEST(MultistagePartitionTest, Scenarios)
{
    // 25 scenarios are grouped into balanced segments of at most 5 scenarios without separators
    std::vector<bool> decoupled = scenarios(25, 4);
    std::vector<double> weights(decoupled.size(), 1.0);
    MultistagePartition partition = partition_multistage(weights, decoupled, 6);
    check_valid(partition, decoupled, 6);
    for (std::size_t k = 0; k < partition.segments.size(); k++) {
        ASSERT_FALSE(partition.has_separator_before(k));
        ASSERT_LE(partition.segments[k].size(), 5u * 4u);
        ASSERT_EQ(partition.segments[k].size() % 4, 0u);
    }
}

TEST(MultistagePartitionTest, ScenariosWithSeparators)
{
    // fewer scenarios than segments, scenarios get split further using separators
    std::vector<bool> decoupled = scenarios(2, 20);
    std::vector<double> weights(decoupled.size(), 1.0);
    MultistagePartition partition = partition_multistage(weights, decoupled, 4);
    check_valid(partition, decoupled, 4);
    ASSERT_EQ(segment_sizes(partition), std::vector<std::size_t>({14, 5, 14, 5}));
    ASSERT_EQ(partition.separators, std::vector<std::ptrdiff_t>({-1, 14, -1, 34}));
}

TEST(MultistagePartitionTest, RandomEqualWeightsMatchesBruteForce)
{
    std::mt19937 gen(42);
    for (int trial = 0; trial < 2000; trial++) {
        std::size_t M = std::uniform_int_distribution<std::size_t>(1, 14)(gen);
        std::size_t max_segments = std::uniform_int_distribution<std::size_t>(1, 6)(gen);
        double p_decoupled = std::uniform_real_distribution<double>(0.0, 0.5)(gen);
        std::vector<bool> decoupled(M);
        for (std::size_t i = 0; i < M; i++) decoupled[i] = std::bernoulli_distribution(p_decoupled)(gen);
        decoupled[M - 1] = true;
        std::vector<double> weights(M, 1.0);

        SCOPED_TRACE("trial " + std::to_string(trial));
        MultistagePartition partition = partition_multistage(weights, decoupled, max_segments);
        check_valid(partition, decoupled, max_segments);
        double cost = multistage_partition_cost(partition, weights, decoupled);
        double ref_cost = brute_force_cost(weights, decoupled, max_segments);
        ASSERT_LE(cost, ref_cost * (1 + 1e-5));
        ASSERT_GE(cost, ref_cost * (1 - 1e-12));
    }
}

TEST(MultistagePartitionTest, RandomWeightsBoundedByBruteForce)
{
    // for non-equal weights the separator cost depends on which blocks become separators,
    // hence the result is only optimal up to the difference in the separator weights
    MultistagePartitionCosts costs;
    std::mt19937 gen(7);
    for (int trial = 0; trial < 2000; trial++) {
        std::size_t M = std::uniform_int_distribution<std::size_t>(1, 14)(gen);
        std::size_t max_segments = std::uniform_int_distribution<std::size_t>(1, 6)(gen);
        double p_decoupled = std::uniform_real_distribution<double>(0.0, 0.5)(gen);
        std::vector<bool> decoupled(M);
        std::vector<double> weights(M);
        for (std::size_t i = 0; i < M; i++) {
            decoupled[i] = std::bernoulli_distribution(p_decoupled)(gen);
            weights[i] = std::uniform_real_distribution<double>(0.5, 2.0)(gen);
        }
        decoupled[M - 1] = true;
        double w_min = *std::min_element(weights.begin(), weights.end());
        double w_max = *std::max_element(weights.begin(), weights.end());

        SCOPED_TRACE("trial " + std::to_string(trial));
        MultistagePartition partition = partition_multistage(weights, decoupled, max_segments, costs);
        check_valid(partition, decoupled, max_segments);
        double cost = multistage_partition_cost(partition, weights, decoupled, costs);
        double ref_cost = brute_force_cost(weights, decoupled, max_segments);
        ASSERT_GE(cost, ref_cost * (1 - 1e-12));
        ASSERT_LE(cost, ref_cost * (1 + 1e-5) + costs.separator * (w_max - w_min) * static_cast<double>(max_segments - 1));
    }
}

TEST(MultistagePartitionTest, LargeProblem)
{
    std::vector<bool> decoupled = chain(2000);
    std::vector<double> weights(decoupled.size(), 1.0);
    MultistagePartition partition = partition_multistage(weights, decoupled, 16);
    check_valid(partition, decoupled, 16);
    ASSERT_EQ(partition.segments.size(), 16u);
}
