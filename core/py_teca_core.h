#ifndef py_teca_core_h
#define py_teca_core_h

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>
#include <cstdlib>

#include "teca_common.h"
#include "teca_variant_array.h"

/* traits classes for working with PyArrayObject's
 *
 * use numpy_cpp_tt to get the C++ type given a
 * numpy enum. use cpp_numpy_tt to get the enum
 * given a C++ type.
 *
 * CODE -- numpy type enumeration
 * CPP_T -- corresponding C++ type
 */
template <int numpy_code> struct numpy_cpp_tt
{};
#define numpy_cpp_tt_declare(CODE, CPP_T)   \
template <> struct numpy_cpp_tt<CODE>       \
{                                           \
    typedef CPP_T type;                     \
};
numpy_cpp_tt_declare(NPY_BYTE, char)
numpy_cpp_tt_declare(NPY_INT32, int)
numpy_cpp_tt_declare(NPY_INT64, long long)
numpy_cpp_tt_declare(NPY_UBYTE, unsigned char)
numpy_cpp_tt_declare(NPY_UINT32, unsigned int)
numpy_cpp_tt_declare(NPY_UINT64, unsigned long long)
numpy_cpp_tt_declare(NPY_FLOAT, float)
numpy_cpp_tt_declare(NPY_DOUBLE, double)


template <typename cpp_t> struct cpp_numpy_tt
{};
#define cpp_numpy_tt_declare(CODE, CPP_T)   \
template <> struct cpp_numpy_tt<CPP_T>      \
{                                           \
    enum { code = CODE };                   \
};
cpp_numpy_tt_declare(NPY_BYTE, char)
cpp_numpy_tt_declare(NPY_INT16, short)
cpp_numpy_tt_declare(NPY_INT32, int)
cpp_numpy_tt_declare(NPY_LONG, long)
cpp_numpy_tt_declare(NPY_INT64, long long)
cpp_numpy_tt_declare(NPY_UBYTE, unsigned char)
cpp_numpy_tt_declare(NPY_UINT16, unsigned short)
cpp_numpy_tt_declare(NPY_UINT32, unsigned int)
cpp_numpy_tt_declare(NPY_ULONG, unsigned long)
cpp_numpy_tt_declare(NPY_UINT64, unsigned long long)
cpp_numpy_tt_declare(NPY_FLOAT, float)
cpp_numpy_tt_declare(NPY_DOUBLE, double)

// ****************************************************************************
template <typename cpp_t>
bool array_is_type(PyArrayObject *arr)
{
    return
    PyArray_TYPE(arr) == cpp_numpy_tt<cpp_t>::code;
    /*array->descr->type_enum
        == cpp_numpy_tt<cpp_t>::code;*/
}

// ****************************************************************************
template <typename cpp_t>
p_teca_variant_array new_from_array_t(PyArrayObject *arr)
{
    size_t n_elem = PyArray_SIZE(arr);
    if (n_elem)
    {
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
    return nullptr;
}

// ****************************************************************************
p_teca_variant_array new_from_array(PyObject *obj)
{
    if (PyArray_Check(obj))
    {
        PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(obj);
        p_teca_variant_array varr;
        if ((array_is_type<int>(arr) && (varr = new_from_array_t<int>(arr)))
            || (array_is_type<float>(arr) && (varr = new_from_array_t<float>(arr)))
            || (array_is_type<double>(arr) && (varr = new_from_array_t<double>(arr)))
            || (array_is_type<char>(arr) && (varr = new_from_array_t<char>(arr)))
            || (array_is_type<long>(arr) && (varr = new_from_array_t<long>(arr)))
            || (array_is_type<long long>(arr) && (varr = new_from_array_t<long long>(arr)))
            || (array_is_type<unsigned char>(arr) && (varr = new_from_array_t<unsigned char>(arr)))
            || (array_is_type<unsigned int>(arr) && (varr = new_from_array_t<unsigned int>(arr)))
            || (array_is_type<unsigned long>(arr) && (varr = new_from_array_t<unsigned long>(arr)))
            || (array_is_type<unsigned long long>(arr) && (varr = new_from_array_t<unsigned long long>(arr))))
        {
            return varr;
        }
    }
    return nullptr;
}

// ****************************************************************************
template<typename cpp_t>
PyArrayObject *new_from_variant_array(teca_variant_array_impl<cpp_t> *varr)
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
        PyArray_SimpleNewFromData(1, dim, cpp_numpy_tt<cpp_t>::code, mem));
    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_OWNDATA);

    return arr;
}






/* traits classes for working with PyObject's
 *
 * PY_T -- C-name of python type
 * CPP_T -- underlying type needed to store it on the C++ side
 * PY_CHECK -- function that verifies the PyObject is this type
 * PY_AS_CPP -- function that converts to the C++ type
 * CPP_AS_PY -- function that converts from the C++ type
 */
template <typename py_t> struct py_object_cpp_tt
{};
#define py_object_cpp_tt_declare(PY_T, CPP_T, PY_CHECK, PY_AS_CPP)  \
template <> struct py_object_cpp_tt<PY_T>                           \
{                                                                   \
    typedef CPP_T cpp_t;                                         \
    static bool is_type(PyObject *obj) { return PY_CHECK(obj); }    \
    static cpp_t value(PyObject *obj) { return PY_AS_CPP(obj); } \
};
py_object_cpp_tt_declare(int, long, PyInt_Check, PyInt_AsLong)
py_object_cpp_tt_declare(long, long, PyLong_Check, PyLong_AsLong)
py_object_cpp_tt_declare(float, double, PyFloat_Check, PyFloat_AsDouble)
py_object_cpp_tt_declare(char*, std::string, PyString_Check, PyString_AsString)
py_object_cpp_tt_declare(bool, int, PyBool_Check, PyInt_AsLong)

template <typename cpp_t> struct cpp_py_object_tt
{};
#define cpp_py_object_tt_declare(CPP_T, CPP_AS_PY)                  \
template <> struct cpp_py_object_tt<CPP_T>                          \
{ static PyObject *new_object(CPP_T val) { return CPP_AS_PY(val); } };
cpp_py_object_tt_declare(char, PyInt_FromLong)
cpp_py_object_tt_declare(short, PyInt_FromLong)
cpp_py_object_tt_declare(int, PyInt_FromLong)
cpp_py_object_tt_declare(long, PyInt_FromLong)
cpp_py_object_tt_declare(long long, PyInt_FromSsize_t)
cpp_py_object_tt_declare(unsigned char, PyInt_FromSize_t)
cpp_py_object_tt_declare(unsigned short, PyInt_FromSize_t)
cpp_py_object_tt_declare(unsigned int, PyInt_FromSize_t)
cpp_py_object_tt_declare(unsigned long, PyInt_FromSize_t)
cpp_py_object_tt_declare(unsigned long long, PyInt_FromSize_t)
cpp_py_object_tt_declare(float, PyFloat_FromDouble)
cpp_py_object_tt_declare(double, PyFloat_FromDouble)
template <> struct cpp_py_object_tt<std::string>
{
    static PyObject *new_object(const std::string &s)
    { return PyString_FromString(s.c_str()); }
};

// ****************************************************************************
template <typename py_t>
bool sequence_is_type(PyObject *seq)
{
    int n_items = static_cast<int>(PySequence_Size(seq));
    for (int i = 0; i < n_items; ++i)
        if (!py_object_cpp_tt<py_t>::is_type(PySequence_GetItem(seq, i)))
            return false;
    return true;
}

// ****************************************************************************
template <typename py_t>
p_teca_variant_array new_from_sequence_t(PyObject *seq)
{
    int n_items = static_cast<int>(PySequence_Size(seq));
    if (n_items)
    {
        p_teca_variant_array_impl<typename py_object_cpp_tt<py_t>::cpp_t> va
            = teca_variant_array_impl<typename py_object_cpp_tt<py_t>::cpp_t>::New(n_items);

        for (int i = 0; i < n_items; ++i)
            va->get(i) = py_object_cpp_tt<py_t>::value(PySequence_GetItem(seq, i));

        return va;
    }
    return nullptr;
}

// ****************************************************************************
p_teca_variant_array new_from_sequence(PyObject *seq)
{
    if (PySequence_Check(seq))
    {
        if (PySequence_Size(seq))
        {
            p_teca_variant_array va;
            if ((sequence_is_type<int>(seq) && (va = new_from_sequence_t<int>(seq)))
              || (sequence_is_type<float>(seq) && (va = new_from_sequence_t<float>(seq)))
              || (sequence_is_type<long>(seq) && (va = new_from_sequence_t<long>(seq)))
              || (sequence_is_type<char*>(seq) && (va = new_from_sequence_t<char*>(seq)))
              || (sequence_is_type<bool>(seq) && (va = new_from_sequence_t<bool>(seq))))
            {
                return va;
            }
        }
        TECA_ERROR("Failed to transfer the sequence. "
            "Sequences must be non-empty and have homogenious type.")
    }
    return nullptr;
}

// ****************************************************************************
template <typename py_t>
p_teca_variant_array new_from_object_t(PyObject *obj)
{
    p_teca_variant_array_impl<typename py_object_cpp_tt<py_t>::cpp_t> varr
        = teca_variant_array_impl<typename py_object_cpp_tt<py_t>::cpp_t>::New(1);

    varr->get(0) = py_object_cpp_tt<py_t>::value(obj);

    return varr;
}

// ****************************************************************************
p_teca_variant_array new_from_object(PyObject *obj)
{
    if (py_object_cpp_tt<bool>::is_type(obj))
        return new_from_object_t<bool>(obj);
    else
    if (py_object_cpp_tt<int>::is_type(obj))
        return new_from_object_t<int>(obj);
    else
    if (py_object_cpp_tt<long>::is_type(obj))
        return new_from_object_t<long>(obj);
    else
    if (py_object_cpp_tt<float>::is_type(obj))
        return new_from_object_t<float>(obj);
    else
    if (py_object_cpp_tt<char*>::is_type(obj))
        return new_from_object_t<char*>(obj);
    return nullptr;
}

#endif
