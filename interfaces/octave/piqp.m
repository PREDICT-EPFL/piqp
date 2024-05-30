%% foo
%% opts are struct with fields listed in piqp distribution, piqp/include/piqp/settings.hpp. E.g., struct("verbose", true)
function [x, obj, INFO, lambda] = piqp (Q, c, A, b, lb, ub, rl, rA, ru, opts)
  save "/tmp/foo4a.dat" Q c A b rl rA ru lb ub
  x=[];
  obj=NaN;
  INFO.info = NaN;
  lambda = NaN;
  if nargin != 10
    error("Wrong # args to piqp");
  endif
  n = size(Q, 1);
  neq = size(A, 1);
  nineq = size(rA, 1);
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
    error("A of wrong # columns");
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
  if isempty(rl)
    rl = zeros(0,1);
  endif
  if !all(size(rl) == [nineq 1])
    error("rl of wrong dimensions");
  endif
  if isempty(rA)
    rA = zeros(0,n);
  endif
  if size(rA,2) != n
    error("rA of wrong # columns");
  endif
  if isempty(ru)
    ru = zeros(0,1);
  endif
  if !all(size(ru) == [nineq 1])
    error("ru of wrong dimensions");
  endif
  
  %% For rA*x >= rl, is equivalent to rA*x + s = rl and s <= 0
  %% so add constraints:
  %% rA*x + s = rl
  %% s <= 0
  %% First add slack variables:
  idxs = find(isfinite(rl));
  nidxs = numel(idxs);
  if nidxs > 0
    [nqr, nqc] = size(Q);
    Q = [ Q zeros(nqr, nidxs) ; zeros(nidxs, nqc + nidxs) ];
    c = [ c ; zeros(nidxs, 1) ];
    A = [ A zeros(size(A, 1), nidxs) ];
    %% b is fine
    lb = [ lb ; -Inf*ones(nidxs, 1) ];
    ub = [ ub ; zeros(nidxs, 1) ];  %% adding s <= 0 in one go
    preRA = rA;
    rA = [ rA zeros(size(rA,1), nidxs) ];
    %% ru is fine

    %% Now add constraints
    %% rA*x + s = rl
    A = [ A ; preRA(idxs,:) eye(nidxs) ];
    b = [ b ; rl(idxs) ];
  endif
  idxs = find(isfinite(ru));
  rA = rA(idxs,:);
  ru = ru(idxs);
  jasserteq(size(rA,1), size(ru, 1));
  


  save "/tmp/foo4b.dat" Q c A b rA ru lb ub
  rez = __piqp__(Q, c, A, b, rA, ru, lb, ub, opts);
  x = rez.x(1:n);
  obj = rez.info.primal_obj;
  if rez.info.status == 1
    INFO.info = 0
    %% Success. Make sure constraints are respected
    ttol = 1e-10;
    fullX = rez.x;
    jassert(!isempty(fullX));
    idxs = find(fullX < (lb - ttol));
    isbad = false;
    if !isempty(idxs)
      printf("LB violation:\n");
      lb(idxs)
      (x - lb)(idxs)
      isbad = true;
    endif
    idxs = find(fullX > (ub + ttol));
    if !isempty(idxs)
      printf("UB violation:\n");
      ub(idxs)
      (ub - x)(idxs)
      isbad = true;
    endif
    if ~isempty(A)
      dev = abs((A * fullX) - b);
      idxs = find(dev > ttol);
      if !isempty(idxs)
        printf("Ax=b violation:\n");
        fullX(idxs)
        (A*fullX)(idxs)
        b(idxs)
        dev(idxs)
        isbad = true;
      endif
    endif
    if ~isempty(rA)
      dev = (rA * fullX) - ru;
      idxs = find(dev > ttol);
      if !isempty(idxs)
        printf("rAx <= ru violation:\n");
        fullX(idxs)
        (rA*fullX)(idxs)
        ru(idxs)
        dev(idxs)
        isbad = true;
      endif
    endif
    if isbad
      jassert(false);
    endif
  else
    INFO.info = rez.info.status;
  endif
  INFO.rez = rez;
endfunction
