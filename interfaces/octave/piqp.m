% Copyright (c) 2024 Joshua Redstone
%
% This file is part of PIQP.
%
% This source code is licensed under the BSD 2-Clause License found in the
% LICENSE file in the root directory of this source tree.

## -*- texinfo -*-
## @deftypefn {} {[@var{x}, @var{obj}, @var{info}] =} piqp (@var{x0}, @var{Q}, @var{c}, @var{A}, @var{b}, @var{lb}, @var{ub}, @var{G}, @var{h})
## @deftypefnx {} {[@var{x}, @var{obj}, @var{info}] =} piqp (@dots{}, @var{options})
## Solve a quadratic program (QP) using the dense PIQP solver.
##
## Solve the quadratic program defined by
## @tex
## $$
##  \min_x {1 \over 2} x^T Q x + x^T c \\
##  \text {s.t.} \quad  Ax=b, \quad lb \leq x \leq ub, \quad  Gx \leq h
## $$
## @end tex
## @ifnottex
##
## @example
## @group
## min 0.5 x'*Q*x + x'*c
##  x
##
## s.t.  Ax=b, lb <= x <= ub,  Gx <= h
## @end group
## @end example
##
## @end ifnottex
## @noindent
##
## Any bound (@var{A}, @var{b}, @var{lb}, @var{ub}, @var{G}, @var{h},
## may be set to the empty matrix (@code{[]}) if not present.  The
## constraints @var{A} and @var{G} are matrices with each row representing
## a single constraint.  The other bounds are scalars or vectors depending on
## the number of constraints.
##
## @var{options} is a structure specifying additional parameters as defined in
## https://github.com/PREDICT-EPFL/piqp/blob/main/include/piqp/settings.hpp
## For example:   struct("verbose", true)
##
## On return, @var{result} is the raw result structure
## returned by __piqp__. The Result structure is defined in
## https://github.com/PREDICT-EPFL/piqp/blob/main/include/piqp/results.hpp
## For example, @code{result.x} is the location of the minimum
## and @code{result.info.primal_obj} is the value of the objective at the minimum
## and @code{resutl.info.status} is the status of the result, with 1 meaning success.
##
## @seealso{qp}
## @end deftypefn
##
function [result] = piqp (Q, c, A, b, lb, ub, G, h, opts)
  result = NaN;
  if nargin ==8
    opts = struct();
  elseif nargin == 9
    %% opts is specified
  else
    error("Wrong # args to piqp");
  endif
  n = size(Q, 1);
  neq = size(A, 1);
  nineq = size(G, 1);
  if !issquare(Q)
    error("Q must be square");
  endif
  if !all(size(c) == [n 1])
    error("c of wrong dimensions");
  endif
  if isempty(A)
    A = zeros(0,n);
  endif
  if size(A,2) != n
    error("A of wrong # of columns");
  endif
  if isempty(b)
    b = zeros(0,1);
  endif
  if !all(size(b) == [neq 1])
    error("b of wrong dimensions");
  endif
  if isempty(lb)
    lb = -Inf*ones(n,1);
  endif
  if !all(size(lb) == [n 1])
    error("lb of wrong dimensions");
  endif
  if isempty(ub)
    ub = Inf*ones(n,1);
  endif
  if !all(size(ub) == [n 1])
    error("ub of wrong dimensions");
  endif
  if isempty(G)
    G = zeros(0,n);
  endif
  if size(G,2) != n
    error("G of wrong # columns");
  endif
  if isempty(h)
    h = zeros(0,1);
  endif
  if !all(size(h) == [nineq 1])
    error("h of wrong dimensions");
  endif

  result = __piqp__(Q, c, A, b, G, h, lb, ub, opts);
endfunction
