// This file is part of PIQP.
// It is a modification of MATio which is part of eigen-matio.
//
// Copyright (C) 2015 Michael Tesch, tesch1 (a) gmail com
// Copyright (C) 2024 EPFL
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.
#ifndef PIQP_UTILS_EIGEN_MATIO_HPP
#define PIQP_UTILS_EIGEN_MATIO_HPP

#include "matio.h"
#ifndef MATIO_VERSION
#define MATIO_VERSION (MATIO_MAJOR_VERSION* 100 + MATIO_MINOR_VERSION* 10 + MATIO_RELEASE_LEVEL)
#endif

#if MATIO_VERSION <= 150
#define MAT_COMPRESSION_NONE COMPRESSION_NONE
#define MAT_COMPRESSION_ZLIB COMPRESSION_ZLIB
typedef ComplexSplit mat_complex_split_t;
#endif

namespace Eigen {

namespace internal {

// defaults
template <typename Tp> struct matio_type;
template <typename Tp> struct matio_class;
template <typename Tp> struct matio_flag { static const matio_flags fid = (matio_flags)0; };
// specializations
//template <> struct matio_type<xxx>       { typedef xxx type; matio_types id = MAT_T_UNKNOWN;                   };
template <> struct matio_type<int8_t>      { typedef int8_t type;   static const matio_types tid = MAT_T_INT8;   };
template <> struct matio_type<uint8_t>     { typedef uint8_t type;  static const matio_types tid = MAT_T_UINT8;  };
template <> struct matio_type<int16_t>     { typedef int16_t type;  static const matio_types tid = MAT_T_INT16;  };
template <> struct matio_type<uint16_t>    { typedef uint16_t type; static const matio_types tid = MAT_T_UINT16; };
template <> struct matio_type<int32_t>     { typedef int32_t type;  static const matio_types tid = MAT_T_INT32;  };
template <> struct matio_type<uint32_t>    { typedef uint32_t type; static const matio_types tid = MAT_T_UINT32; };
template <> struct matio_type<float>        { typedef float type;    static const matio_types tid = MAT_T_SINGLE;  };
template <> struct matio_type<double>      { typedef double type;   static const matio_types tid = MAT_T_DOUBLE; };
template <> struct matio_type<long double> { typedef double type; static const matio_types tid = MAT_T_DOUBLE;   };
template <> struct matio_type<int64_t>     { typedef int64_t type;  static const matio_types tid = MAT_T_INT64;  };
template <> struct matio_type<uint64_t>    { typedef uint64_t type; static const matio_types tid = MAT_T_UINT64; };
//template <> struct matio_type<xxx>       { typedef xxx type; static const matio_types tid = MAT_T_MATRIX;      };
//template <> struct matio_type<xxx>       { typedef xxx type; static const matio_types tid = MAT_T_COMPRESSED;  };
//template <> struct matio_type<xxx>       { typedef xxx type; static const matio_types tid = MAT_T_UTF8;        };
//template <> struct matio_type<xxx>       { typedef xxx type; static const matio_types tid = MAT_T_UTF16;       };
//template <> struct matio_type<xxx>       { typedef xxx type; static const matio_types tid = MAT_T_UTF32;       };
//template <> struct matio_type<xxx>       { typedef xxx type; static const matio_types tid = MAT_T_STRING;      };
//template <> struct matio_type<xxx>       { typedef xxx type; static const matio_types tid = MAT_T_CELL;        };
//template <> struct matio_type<xxx>       { typedef xxx type; static const matio_types tid = MAT_T_STRUCT;      };
//template <> struct matio_type<xxx>       { typedef xxx type; static const matio_types tid = MAT_T_ARRAY;       };
//template <> struct matio_type<xxx>       { typedef xxx type; static const matio_types tid = MAT_T_FUNCTION;    };
template <typename Tp> struct matio_type<std::complex<Tp> >
{
  typedef typename matio_type<Tp>::type type;
  static const matio_types tid = matio_type<Tp>::tid;
};

//template <> struct matio_class<xxx>       { static const matio_classes cid = MAT_C_CELL;     };
//template <> struct matio_class<xxx>       { static const matio_classes cid = MAT_C_STRUCT;   };
//template <> struct matio_class<xxx>       { static const matio_classes cid = MAT_C_OBJECT;   };
template <> struct matio_class<char>        { static const matio_classes cid = MAT_C_CHAR;     };
//template <> struct matio_class<xxx>       { static const matio_classes cid = MAT_C_SPARSE;   };
template <> struct matio_class<float>        { static const matio_classes cid = MAT_C_SINGLE;   };
template <> struct matio_class<double>      { static const matio_classes cid = MAT_C_DOUBLE;   };
template <> struct matio_class<long double> { static const matio_classes cid = MAT_C_DOUBLE;   };
template <> struct matio_class<int8_t>      { static const matio_classes cid = MAT_C_INT8;     };
template <> struct matio_class<uint8_t>     { static const matio_classes cid = MAT_C_UINT8;    };
template <> struct matio_class<int16_t>     { static const matio_classes cid = MAT_C_INT16;    };
template <> struct matio_class<uint16_t>    { static const matio_classes cid = MAT_C_UINT16;   };
template <> struct matio_class<int32_t>     { static const matio_classes cid = MAT_C_INT32;    };
template <> struct matio_class<uint32_t>    { static const matio_classes cid = MAT_C_UINT32;   };
template <> struct matio_class<int64_t>     { static const matio_classes cid = MAT_C_INT64;    };
template <> struct matio_class<uint64_t>    { static const matio_classes cid = MAT_C_UINT64;   };
//template <> struct matio_class<xxx>       { static const matio_classes cid = MAT_C_FUNCTION; };
template <typename Tp> struct matio_class<std::complex<Tp> >
{
  static const matio_classes cid = matio_class<Tp>::cid;
};

template <typename Tp> struct matio_flag<std::complex<Tp> >
{
  static const matio_flags fid = MAT_F_COMPLEX;
};
//template <> struct matio_flag<xxx> { matio_flags fid = MAT_F_GLOBAL; };
//template <> struct matio_flag<xxx> { matio_flags fid = MAT_F_LOGICAL; };
//template <> struct matio_flag<xxx> { matio_flags fid = MAT_F_CLASS_T; };

// reverse map from id to basic type
template <int matio_type_id = -1> struct type_matio {};
template <int matio_class_id = -1> struct class_matio {};
template <> struct type_matio<MAT_T_INT8>   { typedef int8_t   type; };
template <> struct type_matio<MAT_T_UINT8>  { typedef uint8_t  type; };
template <> struct type_matio<MAT_T_INT16>  { typedef int16_t  type; };
template <> struct type_matio<MAT_T_UINT16> { typedef uint16_t type; };
template <> struct type_matio<MAT_T_INT32>  { typedef int32_t  type; };
template <> struct type_matio<MAT_T_UINT32> { typedef uint32_t type; };
template <> struct type_matio<MAT_T_SINGLE> { typedef float     type; };
template <> struct type_matio<MAT_T_DOUBLE> { typedef double   type; };
template <> struct type_matio<MAT_T_INT64>  { typedef int64_t  type; };
template <> struct type_matio<MAT_T_UINT64> { typedef uint64_t type; };

template <> struct class_matio<MAT_C_DOUBLE> { typedef double type; };

} // Eigen::internal::

class MatioFile {

private:
    mat_t* _file;
    mat_ft _ft;
    int _mode;
    bool _written;
    // \TODO - capture errors in std::string _errstr;
    // return them from: const std::string& lasterr();

public:
    MatioFile() : _file(nullptr), _ft(MAT_FT_DEFAULT), _mode(0), _written(false) {}

    MatioFile(const char* filename,
              int mode = MAT_ACC_RDWR,
              bool create = true,
              mat_ft fileversion = MAT_FT_DEFAULT,
              const char* header = "MatioFile")
            : _file(nullptr), _ft(MAT_FT_DEFAULT), _mode(0), _written(false)
    {
        open(filename, mode, create, fileversion, header);
    }

    ~MatioFile()
    {
        close();
    }

    mat_t* file() { return _file; }

    int open(const char* filename,
             int mode = MAT_ACC_RDWR,
             bool create = true,
             mat_ft fileversion = MAT_FT_DEFAULT,
             const char* header = "MatioFile")
    {
        if (_file) {
            close();
        }
        _file = Mat_Open(filename, mode);
        if (_file == nullptr && create && mode != MAT_ACC_RDONLY) {
#if MATIO_VERSION >= 150
            _file = Mat_CreateVer(filename, header, fileversion);
#else
            _file = Mat_Create(filename, header);
#endif
        }
        if (nullptr == _file) {
            std::cout << "MatioFile::open() unable to open " << filename << "/" << fileversion << "\n";
            return -1;
        }
        _mode = mode;
        _ft = fileversion;
        return 0;
    }

    void close()
    {
        if (_file) {
            Mat_Close(_file);
        }
        _file = nullptr;
    }

    // in case reading something after writing it (fflush / fsync would be better)
    int reopen()
    {
        if (_file) {
            std::string filename = Mat_GetFilename(_file);
            Mat_Close(_file);
            _file = Mat_Open(filename.c_str(), _mode);
            if (nullptr == _file) {
                std::cout << "MatioFile() unable to reopen '" << filename << "'\n";
                return -1;
            }
            _written = false;
        }
        return 0;
    }

    mat_t* getFile()
    {
        return _file;
    }

private:
    template<class Derived>
    int write_mat_impl(const char* matname, const DenseBase<Derived>& matrix, matio_compression compression)
    {
#if MATIO_VERSION >= 150
        size_t rows = static_cast<size_t>(matrix.rows());
        size_t cols = static_cast<size_t>(matrix.cols());
        size_t dims[2] = {rows, cols};
#else
        int rows = static_cast<int>(matrix.rows());
        int cols = static_cast<int>(matrix.cols());
        int dims[2] = {rows, cols};
#endif

        matio_types tid = internal::matio_type<typename Derived::Scalar>::tid;
        matio_classes cid = internal::matio_class<typename Derived::Scalar>::cid;
        matio_flags cxflag = internal::matio_flag<typename Derived::Scalar>::fid;

        void* data;
        typedef typename internal::matio_type<typename Derived::Scalar>::type mat_type;
        Matrix<mat_type, Dynamic, Dynamic> dst_re;
        Matrix<mat_type, Dynamic, Dynamic> dst_im;
        mat_complex_split_t cs;

        if (cxflag == MAT_F_COMPLEX) {
            dst_re.resize(matrix.rows(), matrix.cols());
            dst_re = matrix.derived().real().template cast<mat_type>();
            dst_im.resize(matrix.rows(), matrix.cols());
            dst_im = matrix.derived().imag().template cast<mat_type>();
            data = &cs;
            cs.Re = dst_re.data();
            cs.Im = dst_im.data();
        } else {
            dst_re.resize(matrix.rows(), matrix.cols());
            dst_re = matrix.derived().real().template cast<mat_type>();
            data = dst_re.data();
        }

        matvar_t* var = Mat_VarCreate(matname, cid, tid, 2, dims, data, cxflag);
        if (nullptr == var) {
            std::cout << "write_mat() unable to create matrix\n";
            return -1;
        }

        int status = Mat_VarWrite(_file, var, compression);
        if (status) {
            std::cout << "write_mat() unable to put variable '" << matname << "'\n";
        } else {
            _written = true;
        }

        Mat_VarFree(var);
        return status;
    }

    template<class Derived>
    int write_mat_impl(const char* matname, const SparseMatrixBase<Derived>& matrix, matio_compression compression)
    {
#if MATIO_VERSION >= 150
        size_t rows = static_cast<size_t>(matrix.rows());
        size_t cols = static_cast<size_t>(matrix.cols());
        size_t dims[2] = {rows, cols};
#else
        int rows = matrix.rows();
        int cols = matrix.cols();
        int dims[2] = {rows, cols};
#endif

        matio_types tid = internal::matio_type<typename Derived::Scalar>::tid;
        matio_classes cid = MAT_C_SPARSE;
        matio_flags cxflag = internal::matio_flag<typename Derived::Scalar>::fid;

        mat_sparse_t sparse;
        typedef typename internal::matio_type<typename Derived::Scalar>::type mat_type;
        SparseMatrix<typename Derived::Scalar, ColMajor, int> dst(matrix.rows(), matrix.cols());
        dst = matrix;
        dst.makeCompressed();
        Matrix<typename std::remove_reference<decltype(*sparse.ir)>::type, Dynamic, 1> dst_ir;
        Matrix<typename std::remove_reference<decltype(*sparse.jc)>::type, Dynamic, 1> dst_jc;
        Matrix<mat_type, Dynamic, 1> dst_re_val;
        Matrix<mat_type, Dynamic, 1> dst_im_val;
        mat_complex_split_t cs;

        mat_uint32_t nz = static_cast<mat_uint32_t>(dst.nonZeros());
        sparse.nzmax = nz;
        dst_ir.resize(dst.nonZeros());
        dst_ir = Map<Matrix<int, Dynamic, 1>>(dst.innerIndexPtr(), dst.nonZeros()).template cast<typename std::remove_reference<decltype(*sparse.ir)>::type>();
        sparse.ir = dst_ir.data();
        sparse.nir = nz;
        dst_jc.resize(dst.outerSize() + 1);
        dst_jc = Map<Matrix<int, Dynamic, 1>>(dst.outerIndexPtr(), dst.outerSize() + 1).template cast<typename std::remove_reference<decltype(*sparse.jc)>::type>();
        sparse.jc = dst_jc.data();
        sparse.njc = static_cast<mat_uint32_t>(dst.outerSize() + 1);
        sparse.ndata = nz;
        if (cxflag == MAT_F_COMPLEX) {
            dst_re_val.resize(dst.nonZeros());
            const Map<Matrix<typename Derived::Scalar, Dynamic, 1>> val_map(dst.valuePtr(), dst.nonZeros());
            dst_re_val = val_map.real().template cast<mat_type>();
            dst_im_val.resize(dst.nonZeros());
            dst_im_val = val_map.imag().template cast<mat_type>();
            sparse.data = &cs;
            cs.Re = dst_re_val.data();
            cs.Im = dst_im_val.data();
        } else {
            dst_re_val.resize(dst.nonZeros());
            const Map<Matrix<typename Derived::Scalar, Dynamic, 1>> val_map(dst.valuePtr(), dst.nonZeros());
            dst_re_val = val_map.real().template cast<mat_type>();
            sparse.data = dst_re_val.data();
        }

        matvar_t* var = Mat_VarCreate(matname, cid, tid, 2, dims, &sparse, cxflag);
        if (nullptr == var) {
            std::cout << "write_mat() unable to create matrix\n";
            return -1;
        }

        int status = Mat_VarWrite(_file, var, compression);
        if (status) {
            std::cout << "write_mat() unable to put variable '" << matname << "'\n";
        } else {
            _written = true;
        }

        Mat_VarFree(var);
        return status;
    }

public:
    template<class Derived>
    int write_mat(const char* matname, const Derived& matrix, matio_compression compression = MAT_COMPRESSION_NONE)
    {
        if (!_file || !matname) {
            std::cout << "MatioFile.write_mat() unable to write matrix '" << matname << "'\n";
            return -1;
        }

        Mat_VarDelete(_file, matname);
        return write_mat_impl(matname, matrix, compression);
    }

private:
    template<class data_t, class Derived, class Scalar>
    int matrix_from_var(PlainObjectBase<Derived>& matrix, matvar_t* var, const Scalar&)
    {
        Index rows = static_cast<Index>(var->dims[0]);
        Index cols = static_cast<Index>(var->dims[1]);
        Map<Matrix<data_t, Dynamic, Dynamic>> map((data_t*) var->data, rows, cols);
        matrix = map.template cast<Scalar>();
        return 0;
    }

    template<class data_t, class Derived, class Scalar>
    int matrix_from_var(PlainObjectBase<Derived>& matrix, matvar_t* var, const std::complex<Scalar>&)
    {
        Index rows = static_cast<Index>(var->dims[0]);
        Index cols = static_cast<Index>(var->dims[1]);
        mat_complex_split_t* cs = (mat_complex_split_t*) var->data;
        Map<Matrix<data_t, Dynamic, Dynamic>> map_re((data_t*) cs->Re, rows, cols);
        Map<Matrix<data_t, Dynamic, Dynamic>> map_im((data_t*) cs->Im, rows, cols);
        matrix.resize(var->dims[0], var->dims[1]);
        matrix.real() = map_re.template cast<Scalar>();
        matrix.imag() = map_im.template cast<Scalar>();
        return 0;
    }

    template<class data_t, class Derived, class Scalar>
    int matrix_from_var(SparseMatrixBase<Derived>& matrix, matvar_t* var, const Scalar&)
    {
        Index rows = static_cast<Index>(var->dims[0]);
        Index cols = static_cast<Index>(var->dims[1]);
        mat_sparse_t* sparse = (mat_sparse_t*) var->data;
        if (sparse->nir != sparse->ndata || sparse->njc != var->dims[1] + 1) {
            std::cout << "read_mat() wrong sparse format\n ";
            return -1;
        }
        Map<SparseMatrix<data_t, ColMajor, typename std::remove_reference<decltype(*sparse->ir)>::type> > map(rows, cols, sparse->ndata, sparse->jc, sparse->ir, (data_t*) sparse->data);
        matrix = map.template cast<Scalar>();
        return 0;
    }

    template<class data_t, class Derived, class Scalar>
    int matrix_from_var(SparseMatrixBase<Derived>& matrix, matvar_t* var, const std::complex<Scalar>&)
    {
        Index rows = static_cast<Index>(var->dims[0]);
        Index cols = static_cast<Index>(var->dims[1]);
        mat_sparse_t* sparse = (mat_sparse_t*) var->data;
        if (sparse->nir != sparse->ndata || sparse->njc != var->dims[1] + 1) {
            std::cout << "read_mat() wrong sparse format\n ";
            return -1;
        }
        mat_complex_split_t* cs = (mat_complex_split_t*) sparse->data;
        Map<Matrix<data_t, Dynamic, 1>> map_re((data_t*) cs->Re, sparse->ndata);
        Map<Matrix<data_t, Dynamic, 1>> map_im((data_t*) cs->Im, sparse->ndata);
        Matrix<Scalar, Dynamic, 1> tmp(sparse->ndata);
        tmp.real() = map_re.template cast<Scalar>();
        tmp.imag() = map_im.template cast<Scalar>();
        Map<SparseMatrix<Scalar, ColMajor, typename std::remove_reference<decltype(*sparse->ir)>::type> > map(rows, cols, sparse->ndata, sparse->jc, sparse->ir, tmp.data());
        matrix = map.template cast<Scalar>();
        return 0;
    }

public:
    template<class Derived>
    int read_mat(matvar_t* var, Derived& matrix)
    {
        if (nullptr == var) {
            std::cout << "null pointer\n";
            return -1;
        }

        if (var->rank != 2) {
            std::cout << "read_mat() can only read rank-2 matrices: '" << var->name << "':\n ";
            Mat_VarPrint(var, 0);
            return -1;
        }

        if (static_cast<bool>(var->isComplex) != static_cast<bool>(NumTraits<typename Derived::Scalar>::IsComplex)) {
            std::cout << "read_mat() complex / real matrix mismatch\n ";
            Mat_VarPrint(var, 0);
            return -1;
        }

#define MATIO_HANDLE_READ_TYPE(MAT_T_X)                                   \
        else if (var->data_type == MAT_T_X)                              \
        do {                                                             \
            typedef typename internal::type_matio<MAT_T_X>::type data_t; \
            typename Derived::Scalar ele_type{};                         \
            return matrix_from_var<data_t>(matrix, var, ele_type);       \
        } while (0)

        if (false) {}
        MATIO_HANDLE_READ_TYPE(MAT_T_INT8);
        MATIO_HANDLE_READ_TYPE(MAT_T_UINT8);
        MATIO_HANDLE_READ_TYPE(MAT_T_INT16);
        MATIO_HANDLE_READ_TYPE(MAT_T_UINT16);
        MATIO_HANDLE_READ_TYPE(MAT_T_INT32);
        MATIO_HANDLE_READ_TYPE(MAT_T_UINT32);
        MATIO_HANDLE_READ_TYPE(MAT_T_SINGLE);
        MATIO_HANDLE_READ_TYPE(MAT_T_DOUBLE);
        MATIO_HANDLE_READ_TYPE(MAT_T_INT64);
        MATIO_HANDLE_READ_TYPE(MAT_T_UINT64);
        else {
            std::cout << "read_mat() unrecognized matrix data_type '" << var->name << "':\n ";
            Mat_VarPrint(var, 0);
            return -1;
        }
#undef MATIO_HANDLE_READ_TYPE

        return 0;
    }


    template<class Derived>
    int read_mat(const char* matname, Derived& matrix)
    {
        if (_written) {
            reopen();
        }
        if (!_file || !matname) {
            std::cout << "MatioFile.read_mat() unable to read file for matrix '" << matname << "'\n";
            return -1;
        }

        matvar_t* var = Mat_VarRead(_file, matname);
        if (var) {
            int rez = read_mat(var, matrix);
            Mat_VarFree(var);
            return rez;
        } else {
            std::cout << "read_mat() unable to read matrix '" << matname << "'\n";
            return -1;
        }
    }
};

template<class Derived, matio_compression compression = MAT_COMPRESSION_NONE>
int write_mat(const char* filename, const char* matname, const Derived& matrix)
{
    MatioFile file(filename);
    return file.write_mat<Derived, compression>(matname, matrix);
}

template<class Derived>
int read_mat(const char* filename, const char* matname, Derived& matrix)
{
    MatioFile file(filename, MAT_ACC_RDONLY, false);
    return file.read_mat(matname, matrix);
}

}

#endif // PIQP_UTILS_EIGEN_MATIO_HPP
