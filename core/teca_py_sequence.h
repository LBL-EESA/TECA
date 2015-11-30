#ifndef teca_py_sequence_h
#define teca_py_sequence_h

#include "teca_common.h"
#include "teca_variant_array.h"
#include "teca_py_object.h"
#include <Python.h>

namespace teca_py_sequence
{
// ****************************************************************************
template <typename py_t>
bool is_type(PyObject *seq)
{
    using py_object_tt = teca_py_object::cpp_tt<py_t>;

    int n_items = static_cast<int>(PySequence_Size(seq));
    if (n_items)
    {
        // all items in the sequence need to have the same type
        for (int i = 0; i < n_items; ++i)
        {
            if (!py_object_tt::is_type(PySequence_GetItem(seq, i)))
                return false;
        }
        return true;
    }
    // need at least one item to determine the sequence's type.
    return false;
}

// ****************************************************************************
template <typename py_t>
bool append_t(
    p_teca_variant_array_impl<typename teca_py_object::cpp_tt<py_t>::type> &va,
    PyObject *seq)
{
    using py_object_tt = teca_py_object::cpp_tt<py_t>;

    int n_items = static_cast<int>(PySequence_Size(seq));
    va->resize(n_items);

    for (int i = 0; i < n_items; ++i)
        va->append(py_object_tt::value(PySequence_GetItem(seq, i)));

    return true;
}

// ****************************************************************************
template <typename py_t>
bool append(
    p_teca_variant_array_impl<typename teca_py_object::cpp_tt<py_t>::type> &va,
    PyObject *seq)
{
    if (PySequence_Check(seq))
    {
        // nothing to do and not an error.
        if (!PySequence_Size(seq))
            return true;

        if ((is_type<int>(seq) && append_t<int>(va, seq))
          || (is_type<float>(seq) && append_t<float>(va, seq))
          || (is_type<long>(seq) && append_t<long>(va, seq))
          || (is_type<char*>(seq) && append_t<char*>(va, seq))
          || (is_type<bool>(seq) && append_t<bool>(va, seq)))
        {
            // made the copy
            return true;
        }
        // failed. probably user is passing in an
        // unsupported type or a container with mixed types.
        TECA_ERROR("Failed to transfer the sequence. "
            "Sequences must have homogenious type.")
    }
    return false;
}

// ****************************************************************************
template <typename py_t>
bool copy_t(
    p_teca_variant_array_impl<typename teca_py_object::cpp_tt<py_t>::type> &va,
    PyObject *seq)
{
    using py_object_tt = teca_py_object::cpp_tt<py_t>;

    int n_items = static_cast<int>(PySequence_Size(seq));
    va->resize(n_items);

    for (int i = 0; i < n_items; ++i)
        va->get(i) = py_object_tt::value(PySequence_GetItem(seq, i));

    return true;
}

// ****************************************************************************
template <typename py_t>
bool copy(
    p_teca_variant_array_impl<typename teca_py_object::cpp_tt<py_t>::type> &va,
    PyObject *seq)
{
    if (PySequence_Check(seq))
    {
        // nothing to do and not an error.
        if (!PySequence_Size(seq))
            return true;

        if ((is_type<int>(seq) && copy_t<int>(va, seq))
          || (is_type<float>(seq) && copy_t<float>(va, seq))
          || (is_type<long>(seq) && copy_t<long>(va, seq))
          || (is_type<char*>(seq) && copy_t<char*>(va, seq))
          || (is_type<bool>(seq) && copy_t<bool>(va, seq)))
        {
            // made the copy
            return true;
        }
        // failed. probably user is passing in an
        // unsupported type or a container with mixed types.
        TECA_ERROR("Failed to transfer the sequence. "
            "Sequence must have homogenious type.")
    }
    return false;
}

// ****************************************************************************
template <typename py_t>
p_teca_variant_array new_copy_t(PyObject *seq)
{
    using py_object_tt = teca_py_object::cpp_tt<py_t>;

    int n_items = static_cast<int>(PySequence_Size(seq));

    p_teca_variant_array_impl<typename py_object_tt::type> va
        = teca_variant_array_impl<typename py_object_tt::type>::New(n_items);

    for (int i = 0; i < n_items; ++i)
        va->get(i) = py_object_tt::value(PySequence_GetItem(seq, i));

    return va;
}

// ****************************************************************************
p_teca_variant_array new_copy(PyObject *seq)
{
    if (PySequence_Check(seq))
    {
        p_teca_variant_array va;
        if ((is_type<int>(seq) && (va = new_copy_t<int>(seq)))
          || (is_type<float>(seq) && (va = new_copy_t<float>(seq)))
          || (is_type<long>(seq) && (va = new_copy_t<long>(seq)))
          || (is_type<char*>(seq) && (va = new_copy_t<char*>(seq)))
          || (is_type<bool>(seq) && (va = new_copy_t<bool>(seq))))
        {
            return va;
        }
        // failed to make the copy. probably user is passing in an
        // unsupported type or a container with mixed types.
        TECA_ERROR("Failed to transfer the sequence. "
            "Sequences must be non-empty and have homogenious type.")
    }
    return nullptr;
}
};

#endif
