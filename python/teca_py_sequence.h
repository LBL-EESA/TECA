#ifndef teca_py_sequence_h
#define teca_py_sequence_h

/// @file

#include <Python.h>

#include "teca_config.h"
#include "teca_common.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_py_object.h"
#include "teca_py_string.h"

/// @cond
// this macro is used to build up dispatchers
// PYT - type tag idnetifying the PyObject
// SEQ - PySequence* instance
// CODE - code to execute on match
// ST - typedef coresponding to matching tag
#define TECA_PY_SEQUENCE_DISPATCH_CASE(PYT, SEQ, CODE)  \
    if (teca_py_sequence::is_type<PYT>(SEQ))            \
    {                                                   \
        using ST = PYT;                                 \
        CODE                                            \
    }

// the macro dispatches for all the Python types
#define TECA_PY_SEQUENCE_DISPATCH(SEQ, CODE)                \
    TECA_PY_SEQUENCE_DISPATCH_CASE(bool, SEQ, CODE)         \
    else TECA_PY_SEQUENCE_DISPATCH_CASE(int, SEQ, CODE)     \
    else TECA_PY_SEQUENCE_DISPATCH_CASE(float, SEQ, CODE)   \
    else TECA_PY_SEQUENCE_DISPATCH_CASE(char*, SEQ, CODE)   \
    else TECA_PY_SEQUENCE_DISPATCH_CASE(long, SEQ, CODE)

// this one just the numeric types
#define TECA_PY_SEQUENCE_DISPATCH_NUM(SEQ, CODE)            \
    TECA_PY_SEQUENCE_DISPATCH_CASE(int, SEQ, CODE)          \
    else TECA_PY_SEQUENCE_DISPATCH_CASE(float, SEQ, CODE)   \
    else TECA_PY_SEQUENCE_DISPATCH_CASE(long, SEQ, CODE)

// this one just strings
#define TECA_PY_SEQUENCE_DISPATCH_STR(SEQ, CODE)    \
    TECA_PY_SEQUENCE_DISPATCH_CASE(char*, SEQ, CODE)

/// @endcond

/// Codes for interfacing to Python sequences
namespace teca_py_sequence
{

/** @brief Returns true if all the elements in the sequence have the same type
 * as the template argument.
 */
template <typename py_t>
TECA_EXPORT
bool is_type(PyObject *seq)
{
    // nothing to do
    long n_items = PySequence_Size(seq);
    if (n_items < 1)
        return false;

    // all items must have same type and it must match
    // the requested type
    for (long i = 0; i < n_items; ++i)
    {
        if (!teca_py_object::cpp_tt<py_t>::is_type(PySequence_GetItem(seq, i)))
        {
            if (i)
            {
                TECA_PY_ERROR(PyExc_TypeError, "Sequences with mixed types "
                    " are not supported. Failed at element " <<  i)
            }
            return false;
        }
    }

    // sequence type matches
    return true;
}

/// Appends values from the sequence into the variant array.
TECA_EXPORT
bool append(teca_variant_array *va, PyObject *seq)
{
    // not a sequence
    if (!PySequence_Check(seq) || PyStringCheck(seq))
        return false;

    // nothing to do
    long n_items = PySequence_Size(seq);
    if (!n_items)
        return true;

    // append number objects
    VARIANT_ARRAY_DISPATCH(va,
        TT *vat = static_cast<TT*>(va);
        TECA_PY_SEQUENCE_DISPATCH_NUM(seq,
            for (long i = 0; i < n_items; ++i)
            {
                vat->append(teca_py_object::cpp_tt<ST>::value(
                    PySequence_GetItem(seq, i)));
            }
            return true;
            )
        )
    // append strings
    else VARIANT_ARRAY_DISPATCH_CASE(std::string, va,
        TT *vat = static_cast<TT*>(va);
        TECA_PY_SEQUENCE_DISPATCH_STR(seq,
            for (long i = 0; i < n_items; ++i)
            {
                vat->append(teca_py_object::cpp_tt<ST>::value(
                    PySequence_GetItem(seq, i)));
            }
            return true;
            )
        )

    // unknown type
    return false;
}

/// Copies the values from the sequence into the variant array.
TECA_EXPORT
bool copy(teca_variant_array *va, PyObject *seq)
{
    // not a sequence
    if (!PySequence_Check(seq) || PyStringCheck(seq))
        return false;

    // nothing to do
    long n_items = PySequence_Size(seq);
    if (!n_items)
        return true;

    // copy numeric types
    VARIANT_ARRAY_DISPATCH(va,
        TT *vat = static_cast<TT*>(va);
        TECA_PY_SEQUENCE_DISPATCH_NUM(seq,
            vat->resize(n_items);
            for (long i = 0; i < n_items; ++i)
            {
                vat->set(i, teca_py_object::cpp_tt<ST>::value(
                    PySequence_GetItem(seq, i)));
            }
            return true;
            )
        )

    // copy strings
    else VARIANT_ARRAY_DISPATCH_CASE(std::string, va,
        TT *vat = static_cast<TT*>(va);
        TECA_PY_SEQUENCE_DISPATCH_STR(seq,
            vat->resize(n_items);
            for (long i = 0; i < n_items; ++i)
            {
                vat->set(i, teca_py_object::cpp_tt<ST>::value(
                    PySequence_GetItem(seq, i)));
            }
            return true;
            )
        )

    // unknown type
    return false;
}

/// Returns a new variant array initialized with a copy of the sequence.
TECA_EXPORT
p_teca_variant_array new_variant_array(PyObject *seq)
{
    // not a sequence
    if (!PySequence_Check(seq) || PyStringCheck(seq))
        return nullptr;

    // nothing to do
    long n_items = PySequence_Size(seq);
    if (!n_items)
        return nullptr;

    // copy into a new array
    TECA_PY_SEQUENCE_DISPATCH(seq,
        long n_items = PySequence_Size(seq);

        p_teca_variant_array_impl<typename teca_py_object::cpp_tt<ST>::type> vat
            = teca_variant_array_impl<typename teca_py_object::cpp_tt<ST>::type>::New(n_items);

        for (long i = 0; i < n_items; ++i)
        {
            vat->set(i, teca_py_object::cpp_tt<ST>::value(
                PySequence_GetItem(seq, i)));
        }

        return vat;
        )

    // unknown type
    return nullptr;
}

/// Returns a list initialized with a copy of the variant array.
template<typename NT>
TECA_EXPORT
PyObject *new_object(const teca_variant_array_impl<NT> *va)
{
    unsigned long n_elem = va->size();
    PyObject *list = PyList_New(n_elem);
    auto spva = va->get_host_accessible();
    if (!va->host_accessible()) va->synchronize();
    const NT *pva = spva.get();
    for (unsigned long i = 0; i < n_elem; ++i)
        PyList_SetItem(list, i, teca_py_object::py_tt<NT>::new_object(pva[i]));
    return list;
}

/// Returns a list initialized with a copy of the variant array.
TECA_EXPORT
PyObject *new_object(const_p_teca_variant_array va)
{
    VARIANT_ARRAY_DISPATCH(va.get(),
        return teca_py_sequence::new_object(static_cast<const TT*>(va.get()));
        )
    else VARIANT_ARRAY_DISPATCH_CASE(std::string, va.get(),
        return teca_py_sequence::new_object(static_cast<const TT*>(va.get()));
        )

    TECA_PY_ERROR(PyExc_TypeError,
        "Failed to convert teca_variant_array to PyList")
    return nullptr;
}
};

#endif
