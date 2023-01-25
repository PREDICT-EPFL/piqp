// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_UTILS_OPTIONAL_HPP
#define PIQP_UTILS_OPTIONAL_HPP

#include "piqp/fwd.hpp"
#include "piqp/utils/tl_optional.hpp"

namespace piqp
{

namespace detail {
// Source boost: https://www.boost.org/doc/libs/1_74_0/boost/none.hpp
// the trick here is to make instance defined once as a global but in a header
// file
template<typename T>
struct nullopt_instance
{
    static const T instance;
};
template<typename T>
const T nullopt_instance<T>::instance =
    T(tl::nullopt); // global, but because 'tis a template, no cpp file required
} // namespace detail
template<class T>
using optional = tl::optional<T>;
using nullopt_t = tl::nullopt_t;
constexpr nullopt_t nullopt = detail::nullopt_instance<tl::nullopt_t>::instance;

} // namespace piqp

#endif //PIQP_UTILS_OPTIONAL_HPP
