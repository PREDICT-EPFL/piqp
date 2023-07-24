% This file is part of PIQP.
%
% Copyright (c) 2023 EPFL
% Copyright (c) 2017 Bartolomeo Stellato
%
% This source code is licensed under the BSD 2-Clause License found in the
% LICENSE file in the root directory of this source tree.

classdef piqp < handle

    properties (SetAccess = private, Hidden = true)
        piqpMexHandle
        objectHandle % Handle to underlying C instance
    end

    methods
        %% Constructor - Create a new solver instance
        function this = piqp(varargin)
            % detect available instruction sets
            instructionSets = piqp_instruction_set_mex();
            % select correct mex file
            if instructionSets.avx512f
                this.piqpMexHandle = @piqp_avx512_mex;
            elseif instructionSets.avx2
                this.piqpMexHandle = @piqp_avx2_mex;
            else
                this.piqpMexHandle = @piqp_mex;
            end
            % Construct PIQP solver class
            this.objectHandle = this.piqpMexHandle('new', varargin{:});
        end

        %% Destructor - destroy the solver instance
        function delete(this)
            % Destroy PIQP solver class
            this.piqpMexHandle('delete', this.objectHandle);
        end
    end
end
