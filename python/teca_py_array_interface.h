#ifndef teca_py_array_interface_h
#define teca_py_array_interface_h

#include "Python.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_common.h"
#include "teca_py_gil_state.h"

#include <hamr_python_deleter.h>
#include <hamr_stream.h>

#define ARRAY_INTERFACE_DISPATCH_CASE(_type, _size, _cpp_type, _typestr, _code) \
if ((_typestr[0] != '>') && (_typestr[1] == _type) && (_typestr[2] == _size))   \
{                                                                               \
    using NT = _cpp_type;                                                       \
                                                                                \
    _code                                                                       \
}

#define ARRAY_INTERFACE_DISPATCH(_typestr, _code)                               \
ARRAY_INTERFACE_DISPATCH_CASE('f', '8', double,  _typestr, _code)               \
else ARRAY_INTERFACE_DISPATCH_CASE('f', '4', float,  _typestr, _code)           \
else ARRAY_INTERFACE_DISPATCH_CASE('i', '8', long,  _typestr, _code)            \
else ARRAY_INTERFACE_DISPATCH_CASE('i', '4', int,  _typestr, _code)             \
else ARRAY_INTERFACE_DISPATCH_CASE('i', '2', short,  _typestr, _code)           \
else ARRAY_INTERFACE_DISPATCH_CASE('i', '1', char,  _typestr, _code)            \
else ARRAY_INTERFACE_DISPATCH_CASE('u', '8', unsigned long,  _typestr, _code)   \
else ARRAY_INTERFACE_DISPATCH_CASE('u', '4', unsigned int,  _typestr, _code)    \
else ARRAY_INTERFACE_DISPATCH_CASE('u', '2', unsigned short,  _typestr, _code)  \
else ARRAY_INTERFACE_DISPATCH_CASE('u', '1', unsigned char,  _typestr, _code)



namespace teca_py_array_interface
{

/** Look for the array interface protocol in passed object. */
int has_cuda_array_interface(PyObject *obj)
{
    if (PyObject_HasAttrString(obj, "__cuda_array_interface__"))
    {
        return 1;
    }
   return 0;
}

/** Look for the array interface protocol in passed object. */
int has_numpy_array_interface(PyObject *obj)
{
    if (PyObject_HasAttrString(obj, "__array_interface__"))
    {
        return 1;
    }
   return 0;
}

/** Look for the array interface protocol in passed object. if found parse the
 * dictionary
 *
 * @param[in] obj the obvject to look for the array interfacfe protocol in
 * @param[out] has_array_interface set to true if the object contained the
 *                                 array interface protocol
 * @param[out] alloc the allocator to use with the shared data
 * @param[out] typestr the type string describing the shared data
 * @param[out] n_elem the size of the array
 * @param[out] the pointer to the shared data as an integer
 *
 * @returns 0 if successful. The call will succeed if the object does not have
 * the array interface protocol and will fail if the object has the protocol
 * but implemented it in an unexpected way.  Use the has_array_interface
 * argument to differentiate if needed.
 */
int parse_array_interface(PyObject *obj, int &has_array_interface,
    teca_variant_array::allocator &alloc, const char *&ctypestr,
    size_t &n_elem, size_t &intptr, size_t &istream)
{
    /* get the array interface protocol dictionary. the dictionary will
     * have the same keys regardless if it is the Numba CUDA array
     * interface or the Numpy array interface */
    PyObject *aint = nullptr;
    if (PyObject_HasAttrString(obj, "__cuda_array_interface__"))
    {
        has_array_interface = 1;
        aint = PyObject_GetAttrString(obj, "__cuda_array_interface__");
        alloc = teca_variant_array::allocator::cuda_async;
    }
    else if (PyObject_HasAttrString(obj, "__array_interface__"))
    {
        has_array_interface = 1;
        aint = PyObject_GetAttrString(obj, "__array_interface__");
        alloc = teca_variant_array::allocator::malloc;
    }
    else
    {
        /* the object does not implement the array interface protocol */
        has_array_interface = 0;
        return 0;
    }

    if (!PyDict_Check(aint))
    {
        TECA_ERROR("Failed to parse the array interface protocol dictionary."
            " The object is not a dictonary")
        return -1;
    }

    /* get the shape and compute the size */
    PyObject *shape = nullptr;
    if ((shape = PyDict_GetItemString(aint, "shape")) == nullptr)
    {
        TECA_ERROR("Failed to parse the array interface protocol dictionary."
            " Missing key \"shape\" required by the array interface protocol")
        return -1;
    }

    if (!PySequence_Check(shape))
    {
        TECA_ERROR("Failed to parse the array interface protocol dictionary."
            " The item containing the \"shape\" specification does not support"
            " the sequence protocol")
        return -1;
    }

    n_elem = 1;
    Py_ssize_t nd = PySequence_Size(shape);
    for (Py_ssize_t i = 0; i < nd; ++i)
    {
        PyObject *dimi = PySequence_GetItem(shape, i);
        if (!PyLong_Check(dimi))
        {
            TECA_ERROR("Failed to parse the array interface protocol dictionary."
                " Element" << i << " of the \"shape\" specification is not an int")
            return -1;
        }
        n_elem *= PyLong_AsLong(dimi);
    }

    /* get the pointer to the shared data */
    PyObject *data = nullptr;
    if ((data = PyDict_GetItemString(aint, "data")) == nullptr)
    {
        TECA_ERROR("Failed to parse the array interface protocol dictionary."
            " Missing key \"data\" required by the array interface protocol")
        return -1;
    }

    PyObject *data_0 = nullptr;
    if (!PyTuple_Check(data) ||
        ((data_0 = PyTuple_GetItem(data, 0)) == nullptr) ||
        !PyLong_Check(data_0))
    {
        TECA_ERROR("Failed to parse the array interface protocol dictionary."
            " Failed to get element 0 from the \"data\" specification as an"
            " integer")
        return -1;
    }

    intptr = PyLong_AsSize_t(data_0);

    /* get the typestr and parse to determine the type of data being passed
     * and from this create the buffer using the pointer */
    PyObject *typestr = nullptr;
    ctypestr = nullptr;
    if (((typestr = PyDict_GetItemString(aint, "typestr")) == nullptr) ||
        ((ctypestr = PyUnicode_AsUTF8(typestr)) == nullptr))
    {
        TECA_ERROR("Failed to parse the array interface protocol dictionary."
            " Missing key \"typestr\" required by the array interface protocol")
        return -1;
    }

    /* get the stream from the Numba CUDA array interface */
    PyObject *stream = nullptr;
    if ((stream = PyDict_GetItemString(aint, "stream")) != nullptr)
    {
        istream = PyLong_AsSize_t(stream);
    }
    else
    {
        istream = hamr::stream();
    }

    return 0;
}

/* a constructor enabling zero-copy from Python */
p_teca_variant_array new_variant_array(PyObject *obj)
{
    // check for and parse the array interface protocol
    int has_interface = 0;
    teca_variant_array::allocator alloc = teca_variant_array::allocator::none;
    const char *ctypestr = nullptr;
    size_t n_elem = 0;
    size_t intptr = 0;
    size_t istream = 0;

    if (parse_array_interface(obj, has_interface,
        alloc, ctypestr, n_elem, intptr, istream) || !has_interface)
        return nullptr;

    // if the data is using a different stream than we are, synchronize here
#if defined(TECA_HAS_CUDA)
    if (istream != ((size_t)cudaStreamPerThread))
    {
        cudaStreamSynchronize((cudaStream_t)istream);
    }
#endif

    // zero-copy construct the teca_variant_array passing the pointer
    // to the shared data. the deleter holds a reference to the object
    // while the variant array is using the pointer.
    p_teca_variant_array varr;

    ARRAY_INTERFACE_DISPATCH(ctypestr,
        NT *ptr = (NT*)intptr;

        varr = teca_variant_array_impl<NT>::New(n_elem, ptr,
            alloc, -1, hamr::python_deleter(ptr, n_elem, obj));
        )
    else
    {
        TECA_ERROR("Failed to construct a teca_variant_array using the"
            " array interface protocols. The type specification \""
            << ctypestr << "\" is not suported.")
        return nullptr;
    }

#ifdef TECA_DEBUG
    if (varr)
    {
        TECA_STATUS("Created teca_variant_array from the array interface"
            " protocol." << "data[0] = " << intptr << " typestr = "
            << ctypestr)
    }
#endif

    return varr;
}

/// Copy values from the object into variant array.
TECA_EXPORT
bool copy(teca_variant_array *varr, PyObject *obj)
{
    if (has_cuda_array_interface(obj))
    {
        // zero-copy construct an array
        p_teca_variant_array other = teca_py_array_interface::new_variant_array(obj);
        if (!other)
            return false;

        // assign. this api respects the location of the data, and will only
        // move if neccessary
        varr->assign(other);

        return true;
    }

    // unknown object type
    return false;
}

/// Copy values from the object into variant array.
TECA_EXPORT
bool append(teca_variant_array *varr, PyObject *obj)
{
    if (has_cuda_array_interface(obj))
    {
        // zero-copy construct an array
        p_teca_variant_array other = teca_py_array_interface::new_variant_array(obj);
        if (!other)
            return false;

        // append. this api respects the location of the data, and will only
        // move if neccessary
        varr->append(other);

        return true;
    }

    // unknown object type
    return false;
}

}
#endif
