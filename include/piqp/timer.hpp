// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_TIMER_HPP
#define PIQP_TIMER_HPP

#include <chrono>

namespace piqp
{

template<typename T>
class Timer
{
protected:
    std::chrono::time_point<std::chrono::steady_clock> m_start;
    std::chrono::time_point<std::chrono::steady_clock> m_end;

public:
    void start() noexcept
    {
        m_start = std::chrono::steady_clock::now();
    }

    T stop() noexcept
    {
        m_end = std::chrono::steady_clock::now();
        return static_cast<T>(std::chrono::duration_cast<std::chrono::nanoseconds>(m_end - m_start).count()) * 1e-9;
    }
};

} // namespace piqp

#endif //PIQP_TIMER_HPP
