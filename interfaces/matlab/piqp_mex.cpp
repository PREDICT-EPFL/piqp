// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2017 Bartolomeo Stellato
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include "mex.h"
#include "piqp/piqp.hpp"

#define PIQP_MEX_SIGNATURE 0x271C1A7A

using DenseSolver = piqp::DenseSolver<double>;
using SparseSolver = piqp::SparseSolver<double, int>;

class piqp_mex_handle
{
public:
    explicit piqp_mex_handle(DenseSolver* ptr) : m_ptr(ptr), m_is_dense(true) { m_signature = PIQP_MEX_SIGNATURE; }
    explicit piqp_mex_handle(SparseSolver* ptr) : m_ptr(ptr), m_is_dense(false) { m_signature = PIQP_MEX_SIGNATURE; }
    bool isValid() const { return m_signature == PIQP_MEX_SIGNATURE; }
    bool isDense() const { return m_is_dense; }
    DenseSolver* as_dense_ptr() { return static_cast<DenseSolver*>(m_ptr); }
    SparseSolver* as_sparse_ptr() { return static_cast<SparseSolver*>(m_ptr); }

private:
    uint32_t m_signature;
    bool m_is_dense;
    void* m_ptr;
};

template<typename T>
inline mxArray* create_mex_handle(T* ptr)
{
    mexLock();
    mxArray* out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *((uint64_t*) mxGetData(out)) = reinterpret_cast<uint64_t>(new piqp_mex_handle(ptr));
    return out;
}

inline piqp_mex_handle* get_mex_handle(const mxArray* in)
{
    if (mxGetNumberOfElements(in) != 1 || mxGetClassID(in) != mxUINT64_CLASS || mxIsComplex(in)) {
        mexErrMsgTxt("Input must be a real uint64 scalar.");
    }
    auto *ptr = reinterpret_cast<piqp_mex_handle*>(*((uint64_t*) mxGetData(in)));
    if (!ptr->isValid()) {
        mexErrMsgTxt("Handle not valid.");
    }
    return ptr;
}

inline void destroy_mex_handle(const mxArray* in)
{
    delete get_mex_handle(in);
    mexUnlock();
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // Get the command string
    char cmd[64];
    if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd))) {
        mexErrMsgTxt("First input should be a command string less than 64 characters long.");
    }

    if (!strcmp("new", cmd)) {
        char backend[10];
        if (nlhs < 2) {
            strcpy(backend, "sparse");
            mexWarnMsgTxt("The sparse backend is automatically used. To get rid of this warning or use another backend, "
                          "provide the backend explicitly using pipq('dense') or piqp('sparse').");
        } else if (mxGetString(prhs[1], backend, sizeof(backend))) {
            mexErrMsgTxt("Second input should be string less than 10 characters long.");
        }

        if (!strcmp("dense", backend)) {
            plhs[0] = create_mex_handle(new DenseSolver());
        } else if (!strcmp("sparse", backend)) {
            plhs[0] = create_mex_handle(new SparseSolver());
        } else {
            mexErrMsgTxt("Second input must be 'dense' or 'sparse'.");
        }
        return;
    }

    // Check for a second input
    if (nrhs < 2) {
        mexErrMsgTxt("Second input should be a class instance handle.");
    }
    piqp_mex_handle* mex_handle = get_mex_handle(prhs[1]);

    // delete the object and its data
    if (!strcmp("delete", cmd)) {
        if (mex_handle->isDense()) {
            if (mex_handle->as_dense_ptr()) {
                delete mex_handle->as_dense_ptr();
            }
        } else {
            if (mex_handle->as_sparse_ptr()) {
                delete mex_handle->as_sparse_ptr();
            }
        }

        //clean up the handle object
        destroy_mex_handle(prhs[1]);
        // Warn if other commands were ignored
        if (nlhs != 0 || nrhs != 2) {
            mexWarnMsgTxt("Unexpected arguments ignored.");
        }
        return;
    }
}
