#ifndef teca_py_sequence_h
#define teca_py_sequence_h

#include "teca_common.h"
#include "teca_variant_array.h"
#include "teca_py_object.h"
#include <Python.h>

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


namespace teca_py_sequence
{
// ****************************************************************************
template <typename py_t>
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
                TECA_ERROR("Sequences with mixed types are not supported. "
                    " Failed at element " <<  i)
            }
            return false;
        }
    }

    // sequence type matches
    return true;
}

// ****************************************************************************
bool append(teca_variant_array *va, PyObject *seq)
{
    // not a sequence
    if (!PySequence_Check(seq) || PyString_Check(seq))
        return false;

    // nothing to do
    long n_items = PySequence_Size(seq);
    if (!n_items)
        return true;

    // append number objects
    TEMPLATE_DISPATCH(teca_variant_array_impl, va,
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
    else TEMPLATE_DISPATCH_CASE(teca_variant_array_impl,
        std::string, va,
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

// ****************************************************************************
bool copy(teca_variant_array *va, PyObject *seq)
{
    // not a sequence
    if (!PySequence_Check(seq) || PyString_Check(seq))
        return false;

    // nothing to do
    long n_items = PySequence_Size(seq);
    if (!n_items)
        return true;

    // copy numeric types
    TEMPLATE_DISPATCH(teca_variant_array_impl, va,
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
    else TEMPLATE_DISPATCH_CASE(teca_variant_array_impl,
        std::string, va,
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

// ****************************************************************************
p_teca_variant_array new_variant_array(PyObject *seq)
{
    // not a sequence
    if (!PySequence_Check(seq) || PyString_Check(seq))
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
};

#endif
