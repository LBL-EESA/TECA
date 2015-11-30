#ifndef teca_py_object_h
#define teca_py_object_h

#include "teca_common.h"
#include "teca_variant_array.h"
#include <Python.h>

namespace teca_py_object
{
/// cpp_tt, traits class for working with PyObject's
/**
if know the Python type(PY_T) then this class gives you:

cpp_tt<PY_T>::type -- C++ type for a PyObject of subtype PY_T
cpp_tt<PY_T>::is_type -- identifies given PyObject as subtype of PY_T
cpp_tt<PY_T>::value -- convert given PyObject to its C++ type
*/
template <typename py_t> struct cpp_tt
{};

/*
PY_T -- C-name of python type
CPP_T -- underlying type needed to store it on the C++ side
PY_CHECK -- function that verifies the PyObject is this type
PY_AS_CPP -- function that converts to the C++ type */
#define teca_py_array_cpp_tt_declare(PY_T, CPP_T, PY_CHECK, PY_AS_CPP)  \
template <> struct cpp_tt<PY_T>                                         \
{                                                                       \
    typedef CPP_T type;                                                 \
    static bool is_type(PyObject *obj) { return PY_CHECK(obj); }        \
    static type value(PyObject *obj) { return PY_AS_CPP(obj); }         \
};
teca_py_array_cpp_tt_declare(int, long, PyInt_Check, PyInt_AsLong)
teca_py_array_cpp_tt_declare(long, long, PyLong_Check, PyLong_AsLong)
teca_py_array_cpp_tt_declare(float, double, PyFloat_Check, PyFloat_AsDouble)
teca_py_array_cpp_tt_declare(char*, std::string, PyString_Check, PyString_AsString)
teca_py_array_cpp_tt_declare(bool, int, PyBool_Check, PyInt_AsLong)

/// py_tt, traits class for working with PyObject's
/**
if you know the C++ type(CPP_T) then this class gives you:

py_tt<CPP_T>::new_object -- copy construct the corespoonding PyObject
*/
template <typename type> struct py_tt
{};

/**
CPP_T -- underlying type needed to store it on the C++ side
CPP_AS_PY -- function that converts from the C++ type */
#define teca_py_array_py_tt_declare(CPP_T, CPP_AS_PY)               \
template <> struct py_tt<CPP_T>                                     \
{ static PyObject *new_object(CPP_T val) { return CPP_AS_PY(val); } };
teca_py_array_py_tt_declare(char, PyInt_FromLong)
teca_py_array_py_tt_declare(short, PyInt_FromLong)
teca_py_array_py_tt_declare(int, PyInt_FromLong)
teca_py_array_py_tt_declare(long, PyInt_FromLong)
teca_py_array_py_tt_declare(long long, PyInt_FromSsize_t)
teca_py_array_py_tt_declare(unsigned char, PyInt_FromSize_t)
teca_py_array_py_tt_declare(unsigned short, PyInt_FromSize_t)
teca_py_array_py_tt_declare(unsigned int, PyInt_FromSize_t)
teca_py_array_py_tt_declare(unsigned long, PyInt_FromSize_t)
teca_py_array_py_tt_declare(unsigned long long, PyInt_FromSize_t)
teca_py_array_py_tt_declare(float, PyFloat_FromDouble)
teca_py_array_py_tt_declare(double, PyFloat_FromDouble)
// strings are a special case
template <> struct py_tt<std::string>
{
    static PyObject *new_object(const std::string &s)
    { return PyString_FromString(s.c_str()); }
};
// TODO -- special case for teca_metadata

// ****************************************************************************
template <typename py_t>
p_teca_variant_array new_copy_t(PyObject *obj)
{
    p_teca_variant_array_impl<typename cpp_tt<py_t>::type> varr
        = teca_variant_array_impl<typename cpp_tt<py_t>::type>::New(1);

    varr->get(0) = cpp_tt<py_t>::value(obj);

    return varr;
}

// ****************************************************************************
p_teca_variant_array new_copy(PyObject *obj)
{
    if (cpp_tt<bool>::is_type(obj))
        return new_copy_t<bool>(obj);
    else
    if (cpp_tt<int>::is_type(obj))
        return new_copy_t<int>(obj);
    else
    if (cpp_tt<long>::is_type(obj))
        return new_copy_t<long>(obj);
    else
    if (cpp_tt<float>::is_type(obj))
        return new_copy_t<float>(obj);
    else
    if (cpp_tt<char*>::is_type(obj))
        return new_copy_t<char*>(obj);

    // the object does not have one of the supported sub-object
    // types. it's necessarilly not an error.
    return nullptr;
}
};

#endif
