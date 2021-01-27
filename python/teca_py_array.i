%{
#include "teca_py_integer.h"
#include "teca_py_array.h"
#include <iostream>
#include <typeinfo>
%}

/* define the following to generate some test code (See
test/python/test_typemaps.py) has a substantial impact on
perfromance so best leave it off unless debugging */
/*#define TECA_DEBUG_TYPES*/

%fragment("teca_py_array_scalar", "header")
{
/* the following helper check and conversion functions need to handle
both Python types and NumPy types. SWIG's overload precedence is used
to select the largest compatible type */

/* tell if we should convert to a C floating point value */
template<typename num_t>
int teca_py_array_scalar_check_float(PyObject *obj)
{
    int ret = 0;
    if (PyFloat_Check(obj) || PyLongCheck(obj) ||
#if TECA_PYTHON_VERSION == 2
        PyInt_Check(obj) ||
#endif
        teca_py_array::numpy_scalar_tt<num_t>::is_type(obj))
    {
        ret = 1;
    }
    return ret;
}

/* convert to C floating point value */
template <typename num_t>
int teca_py_array_scalar_convert_float(PyObject *obj, num_t &res)
{
    res = num_t();

    /* check floating point types first, then fall back to integers */
    if (PyFloat_Check(obj))
    {
        res = PyFloat_AsDouble(obj);
        return 0;
    }
    else if (teca_py_array::numpy_scalar_tt<float>::is_type(obj))
    {
        res = teca_py_array::numpy_scalar_tt<float>::value(obj);
        return 0;
    }
    else if (teca_py_array::numpy_scalar_tt<double>::is_type(obj))
    {
        res = teca_py_array::numpy_scalar_tt<double>::value(obj);
        return 0;
    }
    else if (PyLongCheck(obj))
    {
        res = PyLong_AsDouble(obj);
        return 0;
    }
#if TECA_PYTHON_VERSION == 2
    else if (PyInt_Check(obj))
    {
        res = PyInt_AsDouble(obj);
        return 0;
    }
#endif
    else if (teca_py_array::numpy_scalar_tt<char>::is_type(obj))
    {
        res = teca_py_array::numpy_scalar_tt<char>::value(obj);
        return 0;
    }
    else if (teca_py_array::numpy_scalar_tt<unsigned char>::is_type(obj))
    {
        res = teca_py_array::numpy_scalar_tt<unsigned char>::value(obj);
        return 0;
    }
    else if (teca_py_array::numpy_scalar_tt<short>::is_type(obj))
    {
        res = teca_py_array::numpy_scalar_tt<short>::value(obj);
        return 0;
    }
    else if (teca_py_array::numpy_scalar_tt<unsigned short>::is_type(obj))
    {
        res = teca_py_array::numpy_scalar_tt<unsigned short>::value(obj);
        return 0;
    }
    else if (teca_py_array::numpy_scalar_tt<int>::is_type(obj))
    {
        res = teca_py_array::numpy_scalar_tt<int>::value(obj);
        return 0;
    }
    else if (teca_py_array::numpy_scalar_tt<unsigned int>::is_type(obj))
    {
        res = teca_py_array::numpy_scalar_tt<unsigned int>::value(obj);
        return 0;
    }
    else if (teca_py_array::numpy_scalar_tt<long long>::is_type(obj))
    {
        res = teca_py_array::numpy_scalar_tt<long long>::value(obj);
        return 0;
    }
    else if (teca_py_array::numpy_scalar_tt<unsigned long long>::is_type(obj))
    {
        res = teca_py_array::numpy_scalar_tt<unsigned long long>::value(obj);
        return 0;
    }

    return -1;
}

/* tell if we should convert to a C integer value */
template<typename num_t>
int teca_py_array_scalar_check_int(PyObject *obj)
{
    int ret = 0;
    if (PyLongCheck(obj) ||
#if TECA_PYTHON_VERSION == 2
        PyInt_Check(obj) ||
#endif
    teca_py_array::numpy_scalar_tt<num_t>::is_type(obj))
    {
        ret = 1;
    }
    return ret;
}

/* convert to C integer value */
template <typename num_t>
int teca_py_array_scalar_convert_int(PyObject *obj, num_t &res)
{
    res = num_t();

    if (PyLong_Check(obj))
    {
        res = PyLong_AsLong(obj);
        return 0;
    }
#if TECA_PYTHON_VERSION == 2
    else if (PyInt_Check(obj))
    {
        res = PyInt_AsLong(obj);
        return 0;
    }
#endif
    else if (teca_py_array::numpy_scalar_tt<char>::is_type(obj))
    {
        res = teca_py_array::numpy_scalar_tt<char>::value(obj);
        return 0;
    }
    else if (teca_py_array::numpy_scalar_tt<unsigned char>::is_type(obj))
    {
        res = teca_py_array::numpy_scalar_tt<unsigned char>::value(obj);
        return 0;
    }
    else if (teca_py_array::numpy_scalar_tt<short>::is_type(obj))
    {
        res = teca_py_array::numpy_scalar_tt<short>::value(obj);
        return 0;
    }
    else if (teca_py_array::numpy_scalar_tt<unsigned short>::is_type(obj))
    {
        res = teca_py_array::numpy_scalar_tt<unsigned short>::value(obj);
        return 0;
    }
    else if (teca_py_array::numpy_scalar_tt<int>::is_type(obj))
    {
        res = teca_py_array::numpy_scalar_tt<int>::value(obj);
        return 0;
    }
    else if (teca_py_array::numpy_scalar_tt<unsigned int>::is_type(obj))
    {
        res = teca_py_array::numpy_scalar_tt<unsigned int>::value(obj);
        return 0;
    }
    else if (teca_py_array::numpy_scalar_tt<long long>::is_type(obj))
    {
        res = teca_py_array::numpy_scalar_tt<long long>::value(obj);
        return 0;
    }
    else if (teca_py_array::numpy_scalar_tt<unsigned long long>::is_type(obj))
    {
        res = teca_py_array::numpy_scalar_tt<unsigned long long>::value(obj);
        return 0;
    }

    /* no automatic conversion from floating point types,
    this will force the3 caller to cast upstream and should
    prevent some potentially confusing issues */

    return -1;
}
}

/* the following type checks are used to identify the Python or
NumPy scalar type that should be used to convert to the C++ type.
The precednece is set such that the largest type wins, because
for example it would not make sense to convert Python integer
to a C++ char */
%typemap(typecheck, precedence=SWIG_TYPECHECK_DOUBLE,
         fragment="teca_py_array_scalar") float
{
    $1 = teca_py_array_scalar_check_float<float>($input);
#ifdef TECA_DEBUG_TYPES
    std::cerr << "type " << ($1 ? "is " : "is NOT ")
        << typeid($1_ltype).name() << sizeof($1_ltype) << std::endl;
#endif
}

%typemap(typecheck, precedence=SWIG_TYPECHECK_FLOAT,
         fragment="teca_py_array_scalar") double
{
    $1 = teca_py_array_scalar_check_float<double>($input);
#ifdef TECA_DEBUG_TYPES
    std::cerr << "type " << ($1 ? "is " : "is NOT ")
        << typeid($1_ltype).name() << sizeof($1_ltype) << std::endl;
#endif
}

%typemap(typecheck, precedence=SWIG_TYPECHECK_INT8,
         fragment="teca_py_array_scalar") long, long long,
                                          unsigned long,
                                          unsigned long long
{
    $1 = teca_py_array_scalar_check_int<$1_ltype>($input);
#ifdef TECA_DEBUG_TYPES
    std::cerr << "type " << ($1 ? "is " : "is NOT ")
        << typeid($1_ltype).name() << sizeof($1_ltype) << std::endl;
#endif
}

%typemap(typecheck, precedence=SWIG_TYPECHECK_INT16,
         fragment="teca_py_array_scalar") int, unsigned int
{
    $1 = teca_py_array_scalar_check_int<$1_ltype>($input);
#ifdef TECA_DEBUG_TYPES
    std::cerr << "type " << ($1 ? "is " : "is NOT ")
        << typeid($1_ltype).name() << sizeof($1_ltype) << std::endl;
#endif
}

%typemap(typecheck, precedence=SWIG_TYPECHECK_INT32,
         fragment="teca_py_array_scalar") short, unsigned short
{
    $1 = teca_py_array_scalar_check_int<$1_ltype>($input);
#ifdef TECA_DEBUG_TYPES
    std::cerr << "type " << ($1 ? "is " : "is NOT ")
        << typeid($1_ltype).name() << sizeof($1_ltype) << std::endl;
#endif
}

%typemap(typecheck, precedence=SWIG_TYPECHECK_INT64,
         fragment="teca_py_array_scalar") char, unsigned char
{
    $1 = teca_py_array_scalar_check_int<$1_ltype>($input);
#ifdef TECA_DEBUG_TYPES
    std::cerr << "type " << ($1 ? "is " : "is NOT ")
        << typeid($1_ltype).name() << sizeof($1_ltype) << std::endl;
#endif
}

%typemap(in, fragment="teca_py_array_scalar") float, double
{
    if (teca_py_array_scalar_convert_float<$1_ltype>($input, $1))
    {
        TECA_PY_ERROR(PyExc_TypeError,
            << "in method \"$symname\" failed to convert "
            << $input->ob_type->tp_name << " to $1_type at argument $argnum "
            << "\"$1_type $1_name\". An explicit cast may be necessary.")
        SWIG_fail;
    }
#ifdef TECA_DEBUG_TYPES
    std::cerr << "converted to " << typeid($1_ltype).name()
        << sizeof($1_ltype) << " " << $1 << std::endl;
#endif
}

%typemap(in, fragment="teca_py_array_scalar") char, short, int, long,
                                              long long, unsigned char,
                                              unsigned short, unsigned int,
                                              unsigned long, unsigned long long
{
    if (teca_py_array_scalar_convert_int<$1_ltype>($input, $1))
    {
        TECA_PY_ERROR(PyExc_TypeError,
            << "in method \"$symname\" failed to convert "
            << $input->ob_type->tp_name << " to $1_type at argument $argnum "
            << "\"$1_type $1_name\". An explicit cast may be necessary.")
        SWIG_fail;
    }
#ifdef TECA_DEBUG_TYPES
    std::cerr << "converted to "  << typeid($1_ltype).name()
        << sizeof($1_ltype) << " " <<  $1 << std::endl;
#endif
}

#if defined(TECA_DEBUG_TYPES)
%include "teca_py_array_test.i"
#endif
