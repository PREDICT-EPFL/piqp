// This file is part of PIQP.
//
// Copyright (c) 2025 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_VARIABLES_HPP
#define PIQP_VARIABLES_HPP

#include "piqp/typedefs.hpp"

namespace piqp
{

template<typename T>
struct BasicVariables
{
    Vec<T> x;
    Vec<T> y;
    Vec<T> z;
    Vec<T> z_lb;
    Vec<T> z_ub;

    void resize(isize n, isize p, isize m)
    {
        x.resize(n);
        y.resize(p);
        z.resize(m);
        z_lb.resize(n);
        z_ub.resize(n);
    }

    bool allFinite()
    {
        return x.allFinite() && y.allFinite() && z.allFinite() && z_lb.allFinite() && z_ub.allFinite();
    }

    BasicVariables& operator+=(const BasicVariables& rhs)
    {
        x.array() += rhs.x.array();
        y.array() += rhs.y.array();
        z.array() += rhs.z.array();
        z_lb.array() += rhs.z_lb.array();
        z_ub.array() += rhs.z_ub.array();
        return *this;
    }

    friend void swap(BasicVariables& a, BasicVariables& b) noexcept
    {
        std::swap(a.x, b.x);
        std::swap(a.y, b.y);
        std::swap(a.z, b.z);
        std::swap(a.z_lb, b.z_lb);
        std::swap(a.z_ub, b.z_ub);
    }
};

template<typename T>
struct Variables : BasicVariables<T>
{
    Vec<T> s;
    Vec<T> s_lb;
    Vec<T> s_ub;

    void resize(isize n, isize p, isize m)
    {
        BasicVariables<T>::resize(n, p, m);
        s.resize(m);
        s_lb.resize(n);
        s_ub.resize(n);
    }

    bool allFinite()
    {
        return BasicVariables<T>::allFinite() && s.allFinite() && s_lb.allFinite() && s_ub.allFinite();
    }

    Variables& operator+=(const Variables& rhs)
    {
        BasicVariables<T>::operator+=(rhs);
        s.array() += rhs.s.array();
        s_lb.array() += rhs.s_lb.array();
        s_ub.array() += rhs.s_ub.array();
        return *this;
    }

    friend void swap(Variables& a, Variables& b) noexcept
    {
        std::swap(static_cast<BasicVariables<T>&>(a), static_cast<BasicVariables<T>&>(b));
        std::swap(a.s, b.s);
        std::swap(a.s_lb, b.s_lb);
        std::swap(a.s_ub, b.s_ub);
    }
};

} // namespace piqp

#endif //PIQP_VARIABLES_HPP
