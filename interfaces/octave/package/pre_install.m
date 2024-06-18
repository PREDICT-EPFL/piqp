% This file is part of PIQP.
%
% Copyright (c) 2024 EPFL
%
% This source code is licensed under the BSD 2-Clause License found in the
% LICENSE file in the root directory of this source tree.

function pre_install (in)
    make_piqp();
    movefile('piqp_oct.oct', 'inst/piqp_oct.oct');
endfunction
