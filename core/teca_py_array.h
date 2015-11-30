#ifndef teca_py_array_h
#define teca_py_array_h

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>
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

#define cpp_tt_declare(CODE, CPP_T) \
template <> struct cpp_tt<CODE>     \
{                                   \
    typedef CPP_T type;             \
};
cpp_tt_declare(NPY_BYTE, char)
cpp_tt_declare(NPY_INT32, int)
cpp_tt_declare(NPY_INT64, long long)
cpp_tt_declare(NPY_UBYTE, unsigned char)
cpp_tt_declare(NPY_UINT32, unsigned int)
cpp_tt_declare(NPY_UINT64, unsigned long long)
cpp_tt_declare(NPY_FLOAT, float)
cpp_tt_declare(NPY_DOUBLE, double)


/// numpy_tt -- traits class for working with PyArrayObject's
/**
numpy_tt::code -- get the numpy enum given a C++ type.

CODE -- numpy type enumeration
CPP_T -- corresponding C++ type
*/
template <typename cpp_t> struct numpy_tt
{};

#define numpy_tt_declare(CODE, CPP_T)   \
template <> struct numpy_tt<CPP_T>      \
{                                       \
    enum { code = CODE };               \
};
numpy_tt_declare(NPY_BYTE, char)
numpy_tt_declare(NPY_INT16, short)
numpy_tt_declare(NPY_INT32, int)
numpy_tt_declare(NPY_LONG, long)
numpy_tt_declare(NPY_INT64, long long)
numpy_tt_declare(NPY_UBYTE, unsigned char)
numpy_tt_declare(NPY_UINT16, unsigned short)
numpy_tt_declare(NPY_UINT32, unsigned int)
numpy_tt_declare(NPY_ULONG, unsigned long)
numpy_tt_declare(NPY_UINT64, unsigned long long)
numpy_tt_declare(NPY_FLOAT, float)
numpy_tt_declare(NPY_DOUBLE, double)

// ****************************************************************************
template <typename cpp_t>
bool is_type(PyArrayObject *arr)
{
    return PyArray_TYPE(arr) == numpy_tt<cpp_t>::code;
}

// ****************************************************************************
template <typename cpp_t>
bool append_t(p_teca_variant_array_impl<cpp_t> &varr, PyArrayObject *arr)
{
    size_t n_elem = PyArray_SIZE(arr);
    varr->resize(n_elem);

    PyObject *it = PyArray_IterNew(reinterpret_cast<PyObject*>(arr));
    for (size_t i = 0; i < n_elem; ++i)
    {
        varr->append(*static_cast<cpp_t*>(PyArray_ITER_DATA(it)));
        PyArray_ITER_NEXT(it);
    }
    Py_DECREF(it);

    return true;
}

// ****************************************************************************
template <typename cpp_t>
int append(p_teca_variant_array_impl<cpp_t> &varr, PyObject *obj)
{
    if (PyArray_Check(obj))
    {
        PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(obj);
        if ((is_type<int>(arr) && append_t<int>(varr, arr))
            || (is_type<float>(arr) && append_t<float>(varr, arr))
            || (is_type<double>(arr) && append_t<double>(varr, arr))
            || (is_type<char>(arr) && append_t<char>(varr, arr))
            || (is_type<long>(arr) && append_t<long>(varr, arr))
            || (is_type<long long>(arr) && append_t<long long>(varr, arr))
            || (is_type<unsigned char>(arr) && append_t<unsigned char>(varr, arr))
            || (is_type<unsigned int>(arr) && append_t<unsigned int>(varr, arr))
            || (is_type<unsigned long>(arr) && append_t<unsigned long>(varr, arr))
            || (is_type<unsigned long long>(arr) && append_t<unsigned long long>(varr, arr)))
        {
            // data was transfered!
            return true;
        }
    }
    // failed. probably user passed an unknown type
    return false;
}

// ****************************************************************************
template <typename cpp_t>
bool copy_t(p_teca_variant_array_impl<cpp_t> &varr, PyArrayObject *arr)
{
    size_t n_elem = PyArray_SIZE(arr);
    varr->resize(n_elem);

    PyObject *it = PyArray_IterNew(reinterpret_cast<PyObject*>(arr));
    for (size_t i = 0; i < n_elem; ++i)
    {
        varr->get(i) = *static_cast<cpp_t*>(PyArray_ITER_DATA(it));
        PyArray_ITER_NEXT(it);
    }
    Py_DECREF(it);

    return true;
}

// ****************************************************************************
template <typename cpp_t>
int copy(p_teca_variant_array_impl<cpp_t> &varr, PyObject *obj)
{
    if (PyArray_Check(obj))
    {
        PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(obj);
        if ((is_type<int>(arr) && copy_t<int>(varr, arr))
            || (is_type<float>(arr) && copy_t<float>(varr, arr))
            || (is_type<double>(arr) && copy_t<double>(varr, arr))
            || (is_type<char>(arr) && copy_t<char>(varr, arr))
            || (is_type<long>(arr) && copy_t<long>(varr, arr))
            || (is_type<long long>(arr) && copy_t<long long>(varr, arr))
            || (is_type<unsigned char>(arr) && copy_t<unsigned char>(varr, arr))
            || (is_type<unsigned int>(arr) && copy_t<unsigned int>(varr, arr))
            || (is_type<unsigned long>(arr) && copy_t<unsigned long>(varr, arr))
            || (is_type<unsigned long long>(arr) && copy_t<unsigned long long>(varr, arr)))
        {
            // copy was made!
            return true;
        }
    }
    // failed. probably user passed an unknown type
    return false;
}

// ****************************************************************************
template <typename cpp_t>
p_teca_variant_array new_copy_t(PyArrayObject *arr)
{
    size_t n_elem = PyArray_SIZE(arr);

    p_teca_variant_array_impl<cpp_t> varr
        = teca_variant_array_impl<cpp_t>::New(n_elem);

    PyObject *it = PyArray_IterNew(reinterpret_cast<PyObject*>(arr));
    for (size_t i = 0; i < n_elem; ++i)
    {
        varr->get(i) = *static_cast<cpp_t*>(PyArray_ITER_DATA(it));
        PyArray_ITER_NEXT(it);
    }
    Py_DECREF(it);

    return varr;
}

// ****************************************************************************
p_teca_variant_array new_copy(PyObject *obj)
{
    if (PyArray_Check(obj))
    {
        PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(obj);
        p_teca_variant_array varr;
        if ((is_type<int>(arr) && (varr = new_copy_t<int>(arr)))
            || (is_type<float>(arr) && (varr = new_copy_t<float>(arr)))
            || (is_type<double>(arr) && (varr = new_copy_t<double>(arr)))
            || (is_type<char>(arr) && (varr = new_copy_t<char>(arr)))
            || (is_type<long>(arr) && (varr = new_copy_t<long>(arr)))
            || (is_type<long long>(arr) && (varr = new_copy_t<long long>(arr)))
            || (is_type<unsigned char>(arr) && (varr = new_copy_t<unsigned char>(arr)))
            || (is_type<unsigned int>(arr) && (varr = new_copy_t<unsigned int>(arr)))
            || (is_type<unsigned long>(arr) && (varr = new_copy_t<unsigned long>(arr)))
            || (is_type<unsigned long long>(arr) && (varr = new_copy_t<unsigned long long>(arr))))
        {
            return varr;
        }
    }
    return nullptr;
}

// ****************************************************************************
template<typename cpp_t>
PyArrayObject *new_copy(teca_variant_array_impl<cpp_t> *varr)
{
    // allocate a buffer
    npy_intp n_elem = varr->size();
    size_t n_bytes = n_elem*sizeof(cpp_t);
    cpp_t *mem = static_cast<cpp_t*>(malloc(n_bytes));
    if (!mem)
    {
        TECA_ERROR("malloc failed to allocate " << n_bytes << " bytes")
        abort();
        return NULL;
    }

    // copy the data
    memcpy(mem, varr->get(), n_bytes);

    // transfer the buffer to a new numpy object
    npy_intp dim[1] = {n_elem};
    PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNewFromData(1, dim, numpy_tt<cpp_t>::code, mem));
    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_OWNDATA);

    return arr;
}
};

#endif
