#ifndef teca_py_iterator_h
#define teca_py_iterator_h

/// @file

#include "teca_config.h"
#include "teca_common.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_py_object.h"
#include "teca_py_string.h"
#include <Python.h>

/** this macro is used to build up dispatchers
 * PYT - type tag idnetifying the PyObject
 * ITER - PySequence* instance
 * CODE - code to execute on match
 * ST - typedef coresponding to matching tag
 */
#define TECA_PY_ITERATOR_DISPATCH_CASE(PYT, ITER, CODE) \
    if (teca_py_iterator::is_type<PYT>(ITER))           \
    {                                                   \
        using ST = PYT;                                 \
        CODE                                            \
    }

/// the macro dispatches for all the Python types
#define TECA_PY_ITERATOR_DISPATCH(ITER, CODE)               \
    TECA_PY_ITERATOR_DISPATCH_CASE(bool, ITER, CODE)        \
    else TECA_PY_ITERATOR_DISPATCH_CASE(int, ITER, CODE)    \
    else TECA_PY_ITERATOR_DISPATCH_CASE(float, ITER, CODE)  \
    else TECA_PY_ITERATOR_DISPATCH_CASE(char*, ITER, CODE)  \
    else TECA_PY_ITERATOR_DISPATCH_CASE(long, ITER, CODE)

/// this one just the numeric types
#define TECA_PY_ITERATOR_DISPATCH_NUM(ITER, CODE)           \
    TECA_PY_ITERATOR_DISPATCH_CASE(int, ITER, CODE)         \
    else TECA_PY_ITERATOR_DISPATCH_CASE(float, ITER, CODE)  \
    else TECA_PY_ITERATOR_DISPATCH_CASE(long, ITER, CODE)

/// this one just strings
#define TECA_PY_ITERATOR_DISPATCH_STR(ITER, CODE)   \
    TECA_PY_ITERATOR_DISPATCH_CASE(char*, ITER, CODE)

/// Codes for interfacing to Python iterators
namespace teca_py_iterator
{

/// Returns true if the object is iterable.
TECA_EXPORT
bool is_iterable(PyObject *obj)
{
    PyObject *iter = nullptr;
    if (PyStringCheck(obj) || !(iter = PyObject_GetIter(obj)))
    {
        PyErr_Clear();
        return false;
    }
    Py_DECREF(iter);
    return true;
}


/// Returns true if all of the elements of the container match the templated type.
template <typename py_t>
TECA_EXPORT
bool is_type(PyObject *obj)
{
    // all items must have same type and it must match
    // the requested type
    PyObject *iter = PyObject_GetIter(obj);
    PyObject *item = nullptr;
    size_t i = 0;
    while ((item = PyIter_Next(iter)))
    {
        if (!teca_py_object::cpp_tt<py_t>::is_type(item))
        {
            if (i)
            {
                TECA_PY_ERROR(PyExc_TypeError, "mixed types are not "
                    "supported. Failed at element " <<  i)
            }
            return false;
        }
        Py_DECREF(item);
        ++i;
    }
    Py_DECREF(iter);
    // type matches
    return true;
}

/// Appends values from the object to the variant array
TECA_EXPORT
bool append(teca_variant_array *va, PyObject *obj)
{
    // not a iterator
    if (!is_iterable(obj))
        return false;

    // append numeric types
    VARIANT_ARRAY_DISPATCH(va,
        TT *vat = static_cast<TT*>(va);
        TECA_PY_ITERATOR_DISPATCH_NUM(obj,
            PyObject *iter = PyObject_GetIter(obj);
            PyObject *item = nullptr;
            while((item = PyIter_Next(iter)))
            {
                vat->append(teca_py_object::cpp_tt<ST>::value(item));
                Py_DECREF(item);
            }
            Py_DECREF(iter);
            return true;
            )
        )

    // append strings
    else VARIANT_ARRAY_DISPATCH_CASE(
        std::string, va,
        TT *vat = static_cast<TT*>(va);
        TECA_PY_ITERATOR_DISPATCH_STR(obj,
            PyObject *iter = PyObject_GetIter(obj);
            PyObject *item = nullptr;
            while((item = PyIter_Next(iter)))
            {
                vat->append(teca_py_object::cpp_tt<ST>::value(item));
                Py_DECREF(item);
            }
            Py_DECREF(iter);
            return true;
            )
        )

    // unknown type
    return false;
}

/// Copies values from the object into the variant array
TECA_EXPORT
bool copy(teca_variant_array *va, PyObject *obj)
{
    // not a iterator
    if (!is_iterable(obj))
        return false;

    // copy numeric types
    VARIANT_ARRAY_DISPATCH(va,
        TT *vat = static_cast<TT*>(va);
        TECA_PY_ITERATOR_DISPATCH_NUM(obj,
            vat->clear();
            PyObject *iter = PyObject_GetIter(obj);
            PyObject *item = nullptr;
            while((item = PyIter_Next(iter)))
            {
                vat->append(teca_py_object::cpp_tt<ST>::value(item));
                Py_DECREF(item);
            }
            Py_DECREF(iter);
            return true;
            )
        )

    // copy strings
    else VARIANT_ARRAY_DISPATCH_CASE(
        std::string, va,
        TT *vat = static_cast<TT*>(va);
        TECA_PY_ITERATOR_DISPATCH_STR(obj,
            vat->clear();
            PyObject *iter = PyObject_GetIter(obj);
            PyObject *item = nullptr;
            while((item = PyIter_Next(iter)))
            {
                vat->append(teca_py_object::cpp_tt<ST>::value(item));
                Py_DECREF(item);
            }
            Py_DECREF(iter);
            return true;
            )
        )

    // unknown type
    return false;
}

/// Cretaes a new variant array initialized with a copy of the object.
TECA_EXPORT
p_teca_variant_array new_variant_array(PyObject *obj)
{
    // not a iterator
    if (!is_iterable(obj))
        return nullptr;

    // copy into a new array
    TECA_PY_ITERATOR_DISPATCH(obj,

        p_teca_variant_array_impl<typename teca_py_object::cpp_tt<ST>::type> vat
            = teca_variant_array_impl<typename teca_py_object::cpp_tt<ST>::type>::New();

        PyObject *iter = PyObject_GetIter(obj);
        PyObject *item = nullptr;
        while((item = PyIter_Next(iter)))
        {
            vat->append(teca_py_object::cpp_tt<ST>::value(item));
            Py_DECREF(item);
        }
        Py_DECREF(iter);

        return vat;
        )

    // unknown type
    return nullptr;
}
};

#endif
