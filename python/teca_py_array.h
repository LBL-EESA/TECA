#ifndef teca_py_array_h
#define teca_py_array_h

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>
#include <limits>
#include <cstdlib>

#include "teca_common.h"
#include "teca_variant_array.h"

namespace teca_py_array
{
/// cpp_tt -- traits class for working with PyArrayObject's
/**
cpp_tt::type -- get the C++ type given a numpy enum.

CODE -- numpy type enumeration
CPP_T -- corresponding C++ type
*/
template <int numpy_code> struct cpp_tt
{};

#define teca_py_array_cpp_tt_declare(CODE, CPP_T)   \
template <> struct cpp_tt<CODE>                     \
{                                                   \
    typedef CPP_T type;                             \
};
teca_py_array_cpp_tt_declare(NPY_INT8, char)
teca_py_array_cpp_tt_declare(NPY_INT16, short)
teca_py_array_cpp_tt_declare(NPY_INT32, int)
teca_py_array_cpp_tt_declare(NPY_INT64, long long)
teca_py_array_cpp_tt_declare(NPY_UINT8, unsigned char)
teca_py_array_cpp_tt_declare(NPY_UINT16, unsigned short)
teca_py_array_cpp_tt_declare(NPY_UINT32, unsigned int)
teca_py_array_cpp_tt_declare(NPY_UINT64, unsigned long long)
teca_py_array_cpp_tt_declare(NPY_FLOAT, float)
teca_py_array_cpp_tt_declare(NPY_DOUBLE, double)


/// numpy_tt - traits class for working with PyArrayObject's
/**
::code - get the numpy type enum given a C++ type.
::is_type - return true if the PyArrayObject has the given type

CODE -- numpy type enumeration
CPP_T -- corresponding C++ type
*/
template <typename cpp_t> struct numpy_tt
{};

#define teca_py_array_numpy_tt_declare(CODE, CPP_T) \
template <> struct numpy_tt<CPP_T>                  \
{                                                   \
    enum { code = CODE };                           \
    static bool is_type(PyArrayObject *arr)         \
    { return PyArray_TYPE(arr) == CODE; }           \
};
teca_py_array_numpy_tt_declare(NPY_INT8, char)
teca_py_array_numpy_tt_declare(NPY_INT16, short)
teca_py_array_numpy_tt_declare(NPY_INT32, int)
teca_py_array_numpy_tt_declare(NPY_LONG, long)
teca_py_array_numpy_tt_declare(NPY_INT64, long long)
teca_py_array_numpy_tt_declare(NPY_UINT8, unsigned char)
teca_py_array_numpy_tt_declare(NPY_UINT16, unsigned short)
teca_py_array_numpy_tt_declare(NPY_UINT32, unsigned int)
teca_py_array_numpy_tt_declare(NPY_ULONG, unsigned long)
teca_py_array_numpy_tt_declare(NPY_UINT64, unsigned long long)
teca_py_array_numpy_tt_declare(NPY_FLOAT, float)
teca_py_array_numpy_tt_declare(NPY_DOUBLE, double)


// CPP_T - array type to match
// OBJ - PyArrayObject* instance
// CODE - code to execute on match
#define TECA_PY_ARRAY_DISPATCH_CASE(CPP_T, OBJ, CODE)   \
    if (teca_py_array::numpy_tt<CPP_T>::is_type(OBJ))   \
    {                                                   \
        using AT = CPP_T;                               \
        CODE                                            \
    }

#define TECA_PY_ARRAY_DISPATCH(OBJ, CODE)                       \
    TECA_PY_ARRAY_DISPATCH_CASE(float, OBJ, CODE)               \
    TECA_PY_ARRAY_DISPATCH_CASE(double, OBJ, CODE)              \
    TECA_PY_ARRAY_DISPATCH_CASE(int, OBJ, CODE)                 \
    TECA_PY_ARRAY_DISPATCH_CASE(unsigned int, OBJ, CODE)        \
    TECA_PY_ARRAY_DISPATCH_CASE(long, OBJ, CODE)                \
    TECA_PY_ARRAY_DISPATCH_CASE(unsigned long, OBJ, CODE)       \
    TECA_PY_ARRAY_DISPATCH_CASE(long long, OBJ, CODE)           \
    TECA_PY_ARRAY_DISPATCH_CASE(unsigned long long, OBJ, CODE)  \
    TECA_PY_ARRAY_DISPATCH_CASE(char, OBJ, CODE)                \
    TECA_PY_ARRAY_DISPATCH_CASE(unsigned char, OBJ, CODE)       \
    TECA_PY_ARRAY_DISPATCH_CASE(short, OBJ, CODE)               \
    TECA_PY_ARRAY_DISPATCH_CASE(unsigned short, OBJ, CODE)



/// numpy_scalar_tt - traits class for working with PyArrayObject's
/**
::code - get the numpy type enum given a C++ type.
::is_type - return true if the PyArrayObject has the given type

CODE -- numpy type enumeration
STR_CODE -- string part of NumPy typename (see header files)
CPP_T -- corresponding C++ type
*/
template <typename cpp_t> struct numpy_scalar_tt
{
    enum { code = std::numeric_limits<int>::lowest() };
    static bool is_type(PyObject*){ return false; }
    static cpp_t value(PyObject*){ return cpp_t(); }
};

#define teca_py_array_numpy_scalar_tt_declare(CODE, STR_CODE, CPP_T)\
template <> struct numpy_scalar_tt<CPP_T>                           \
{                                                                   \
    enum { code = CODE };                                           \
                                                                    \
    static bool is_type(PyObject *obj)                              \
    { return PyArray_IsScalar(obj, STR_CODE); }                     \
                                                                    \
    static CPP_T value(PyObject *obj)                               \
    {                                                               \
        CPP_T tmp;                                                  \
        PyArray_ScalarAsCtype(obj, &tmp);                           \
        return tmp;                                                 \
    }                                                               \
};
teca_py_array_numpy_scalar_tt_declare(NPY_INT8, Int8, char)
teca_py_array_numpy_scalar_tt_declare(NPY_INT16, Int16, short)
teca_py_array_numpy_scalar_tt_declare(NPY_INT32, Int32, int)
teca_py_array_numpy_scalar_tt_declare(NPY_LONG, Int64, long)
teca_py_array_numpy_scalar_tt_declare(NPY_INT64, Int64, long long)
teca_py_array_numpy_scalar_tt_declare(NPY_UINT8, UInt8, unsigned char)
teca_py_array_numpy_scalar_tt_declare(NPY_UINT16, UInt16, unsigned short)
teca_py_array_numpy_scalar_tt_declare(NPY_UINT32, UInt32, unsigned int)
teca_py_array_numpy_scalar_tt_declare(NPY_UINT64, UInt64, unsigned long)
teca_py_array_numpy_scalar_tt_declare(NPY_ULONG, UInt64, unsigned long long)
teca_py_array_numpy_scalar_tt_declare(NPY_FLOAT, Float32, float)
teca_py_array_numpy_scalar_tt_declare(NPY_DOUBLE, Float64, double)

// CPP_T - array type to match
// OBJ - PyArrayObject* instance
// CODE - code to execute on match
#define TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(CPP_T, OBJ, CODE)   \
    if (teca_py_array::numpy_scalar_tt<CPP_T>::is_type(OBJ))   \
    {                                                          \
        using ST = CPP_T;                                      \
        CODE                                                   \
    }

#define TECA_PY_ARRAY_SCALAR_DISPATCH(OBJ, CODE)                       \
    TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(float, OBJ, CODE)               \
    TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(double, OBJ, CODE)              \
    TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(int, OBJ, CODE)                 \
    TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(unsigned int, OBJ, CODE)        \
    TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(long, OBJ, CODE)                \
    TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(unsigned long, OBJ, CODE)       \
    TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(long long, OBJ, CODE)           \
    TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(unsigned long long, OBJ, CODE)  \
    TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(char, OBJ, CODE)                \
    TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(unsigned char, OBJ, CODE)       \
    TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(short, OBJ, CODE)               \
    TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(unsigned short, OBJ, CODE)


// ****************************************************************************
bool append(teca_variant_array *varr, PyObject *obj)
{
    // numpy ndarray
    if (PyArray_Check(obj))
    {
        PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(obj);

        unsigned long n_elem = PyArray_SIZE(arr);
        if (!n_elem)
            return true;

        // append
        TEMPLATE_DISPATCH(teca_variant_array_impl, varr,
            TT *varrt = static_cast<TT*>(varr);
            varrt->reserve(n_elem);

            TECA_PY_ARRAY_DISPATCH(arr,
                NpyIter *it = NpyIter_New(arr, NPY_ITER_READONLY,
                        NPY_KEEPORDER, NPY_NO_CASTING, nullptr);
                NpyIter_IterNextFunc *next = NpyIter_GetIterNext(it, nullptr);
                AT **ptrptr = reinterpret_cast<AT**>(NpyIter_GetDataPtrArray(it));
                do
                {
                    varrt->append(**ptrptr);
                }
                while (next(it));
                NpyIter_Deallocate(it);
                return true;
                )
            )

        // unknown element type
        return false;
    }

    // numpy scalar
    if (PyArray_CheckScalar(obj))
    {
        TEMPLATE_DISPATCH(teca_variant_array_impl, varr,
            TT *varrt = static_cast<TT*>(varr);
            varrt->reserve(1);

            TECA_PY_ARRAY_SCALAR_DISPATCH(obj,
                varrt->append(teca_py_array::numpy_scalar_tt<ST>::value(obj));
                return true;
                )
            )

        // unknown element type
        return false;
    }

    // unknown conatiner type
    return false;
}

// ****************************************************************************
bool copy(teca_variant_array *varr, PyObject *obj)
{
    if (PyArray_Check(obj) || PyArray_CheckScalar(obj))
    {
        // numpy ndarray or scalar
        varr->resize(0);
        return teca_py_array::append(varr, obj);
    }

    // unknown object type
    return false;
}

// ****************************************************************************
bool set(teca_variant_array *varr, unsigned long i, PyObject *obj)
{
    // numpy scalar
    if (PyArray_CheckScalar(obj))
    {
        TEMPLATE_DISPATCH(teca_variant_array_impl, varr,
            TT *varrt = static_cast<TT*>(varr);
            TECA_PY_ARRAY_SCALAR_DISPATCH(obj,
                varrt->set(i, teca_py_array::numpy_scalar_tt<ST>::value(obj));
                return true;
                )
            )
    }
    return false;
}


// ****************************************************************************
p_teca_variant_array new_variant_array(PyObject *obj)
{
    // not an array
    if (!PyArray_Check(obj))
        return nullptr;

    PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(obj);

    // allocate and copy
    TECA_PY_ARRAY_DISPATCH(arr,
        size_t n_elem = PyArray_SIZE(arr);

        p_teca_variant_array_impl<AT> varrt
             = teca_variant_array_impl<AT>::New();
        varrt->reserve(n_elem);

        NpyIter *it = NpyIter_New(arr, NPY_ITER_READONLY,
                NPY_KEEPORDER, NPY_NO_CASTING, nullptr);
        NpyIter_IterNextFunc *next = NpyIter_GetIterNext(it, nullptr);
        AT **ptrptr = reinterpret_cast<AT**>(NpyIter_GetDataPtrArray(it));
        do
        {
            varrt->append(**ptrptr);
        }
        while (next(it));
        NpyIter_Deallocate(it);

        return varrt;
        )

    // unknown type
    return nullptr;
}

// ****************************************************************************
template <typename NT>
PyArrayObject *new_object(teca_variant_array_impl<NT> *varrt)
{
    // allocate a buffer
    npy_intp n_elem = varrt->size();
    size_t n_bytes = n_elem*sizeof(NT);
    NT *mem = static_cast<NT*>(malloc(n_bytes));
    if (!mem)
    {
        PyErr_Format(PyExc_RuntimeError,
            "failed to allocate %lu bytes", n_bytes);
        return nullptr;
    }

    // copy the data
    memcpy(mem, varrt->get(), n_bytes);

    // put the buffer in to a new numpy object
    PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNewFromData(1, &n_elem, numpy_tt<NT>::code, mem));
    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_OWNDATA);

    return arr;
}

// ****************************************************************************
PyArrayObject *new_object(teca_variant_array *varr)
{
    TEMPLATE_DISPATCH(teca_variant_array_impl, varr,
        TT *varrt = static_cast<TT*>(varr);
        return new_object(varrt);
        )
    return nullptr;
}
};

#endif
