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
    Vec<T> z_l;
    Vec<T> z_u;
    Vec<T> z_bl;
    Vec<T> z_bu;

    void resize(isize n, isize p, isize m)
    {
        x.resize(n);
        y.resize(p);
        z_l.resize(m);
        z_u.resize(m);
        z_bl.resize(n);
        z_bu.resize(n);
    }

    bool allFinite()
    {
        return x.allFinite() && y.allFinite() && z_l.allFinite() && z_u.allFinite()
            && z_bl.allFinite() && z_bu.allFinite();
    }

    BasicVariables& operator+=(const BasicVariables& rhs)
    {
        x.array() += rhs.x.array();
        y.array() += rhs.y.array();
        z_l.array() += rhs.z_l.array();
        z_u.array() += rhs.z_u.array();
        z_bl.array() += rhs.z_l.array();
        z_bu.array() += rhs.z_u.array();
        return *this;
    }

    friend void swap(BasicVariables& a, BasicVariables& b) noexcept
    {
        std::swap(a.x, b.x);
        std::swap(a.y, b.y);
        std::swap(a.z_l, b.z_l);
        std::swap(a.z_u, b.z_u);
        std::swap(a.z_bl, b.z_bl);
        std::swap(a.z_bu, b.z_bu);
    }
};

template<typename T>
struct Variables : BasicVariables<T>
{
    Vec<T> s_l;
    Vec<T> s_u;
    Vec<T> s_bl;
    Vec<T> s_bu;

    void resize(isize n, isize p, isize m)
    {
        BasicVariables<T>::resize(n, p, m);
        s_l.resize(m);
        s_u.resize(m);
        s_bl.resize(n);
        s_bu.resize(n);
    }

    bool allFinite()
    {
        return BasicVariables<T>::allFinite()
            && s_l.allFinite() && s_u.allFinite() && s_bl.allFinite() && s_bu.allFinite();
    }

    Variables& operator+=(const Variables& rhs)
    {
        BasicVariables<T>::operator+=(rhs);
        s_l.array() += rhs.s_l.array();
        s_u.array() += rhs.s_u.array();
        s_bl.array() += rhs.s_bl.array();
        s_bu.array() += rhs.s_bu.array();
        return *this;
    }

    friend void swap(Variables& a, Variables& b) noexcept
    {
        std::swap(static_cast<BasicVariables<T>&>(a), static_cast<BasicVariables<T>&>(b));
        std::swap(a.s_l, b.s_l);
        std::swap(a.s_u, b.s_u);
        std::swap(a.s_bl, b.s_bl);
        std::swap(a.s_bu, b.s_bu);
    }
};

} // namespace piqp

#endif //PIQP_VARIABLES_HPP
