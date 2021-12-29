#ifndef teca_py_array_h
#define teca_py_array_h

/// @file

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>
#include <limits>
#include <cstdlib>

#include "teca_config.h"
#include "teca_common.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_py_common.h"

/// Codes for interfacing with numpy arrays
namespace teca_py_array
{
/// @cond
/// cpp_tt -- traits class for working with PyArrayObject's
/**
cpp_tt::type -- get the C++ type given a numpy enum.

CODE -- numpy type enumeration
CPP_T -- corresponding C++ type
*/
template <int numpy_code> struct TECA_EXPORT cpp_tt
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
template <typename cpp_t> struct TECA_EXPORT numpy_tt
{};

#define teca_py_array_numpy_tt_declare(CODE, KIND, CPP_T)   \
template <> struct numpy_tt<CPP_T>                          \
{                                                           \
    enum { code = CODE };                                   \
                                                            \
    static bool is_type(PyArrayObject *arr)                 \
    { return PyArray_TYPE(arr) == CODE; }                   \
                                                            \
    static constexpr char typekind()                        \
    { return KIND; }                                        \
                                                            \
    static constexpr int itemsize()                         \
    { return sizeof(CPP_T); }                               \
};
teca_py_array_numpy_tt_declare(NPY_INT8, 'i', char)
teca_py_array_numpy_tt_declare(NPY_INT16, 'i', short)
teca_py_array_numpy_tt_declare(NPY_INT32, 'i', int)
teca_py_array_numpy_tt_declare(NPY_LONG, 'i', long)
teca_py_array_numpy_tt_declare(NPY_INT64, 'i', long long)
teca_py_array_numpy_tt_declare(NPY_UINT8, 'u', unsigned char)
teca_py_array_numpy_tt_declare(NPY_UINT16, 'u', unsigned short)
teca_py_array_numpy_tt_declare(NPY_UINT32, 'u', unsigned int)
teca_py_array_numpy_tt_declare(NPY_ULONG, 'u', unsigned long)
teca_py_array_numpy_tt_declare(NPY_UINT64, 'u', unsigned long long)
teca_py_array_numpy_tt_declare(NPY_FLOAT, 'f', float)
teca_py_array_numpy_tt_declare(NPY_DOUBLE, 'f', double)

// CPP_T - array type to match
// OBJ - PyArrayObject* instance
// CODE - code to execute on match
#define TECA_PY_ARRAY_DISPATCH_CASE(CPP_T, OBJ, CODE)   \
    if (teca_py_array::numpy_tt<CPP_T>::is_type(OBJ))   \
    {                                                   \
        using AT = CPP_T;                               \
        CODE                                            \
    }
/// @endcond


/** @brief A dispatch macro that executes the code in CODE based
 *  on the run time determined type of OBJ.
 *
 *  @details The following alias is available for determining the actual
 *  type within the CODE section: `using AT = CPP_T;`
 *
 */
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


/// @cond
/// numpy_scalar_tt - traits class for working with PyArrayObject's
/**
::code - get the numpy type enum given a C++ type.
::is_type - return true if the PyArrayObject has the given type

CODE -- numpy type enumeration
STR_CODE -- string part of NumPy typename (see header files)
CPP_T -- corresponding C++ type
*/
template <typename cpp_t> struct TECA_EXPORT numpy_scalar_tt
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

#define TECA_PY_ARRAY_SCALAR_DISPATCH_FP(OBJ, CODE)                         \
    TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(float, OBJ, CODE)                    \
    else TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(double, OBJ, CODE)

#define TECA_PY_ARRAY_SCALAR_DISPATCH_I8(OBJ, CODE)                         \
    TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(char, OBJ, CODE)                     \
    else TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(unsigned char, OBJ, CODE)

#define TECA_PY_ARRAY_SCALAR_DISPATCH_I16(OBJ, CODE)                        \
    TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(short, OBJ, CODE)                    \
    else TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(unsigned short, OBJ, CODE)

#if 0
#define TECA_PY_ARRAY_SCALAR_DISPATCH_I32(OBJ, CODE)                        \
    TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(int, OBJ, CODE)                      \
    else TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(unsigned int, OBJ, CODE)        \
    else TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(long, OBJ, CODE)                \
    else TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(unsigned long, OBJ, CODE)

#define TECA_PY_ARRAY_SCALAR_DISPATCH_I64(OBJ, CODE)                        \
    TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(long long, OBJ, CODE)                \
    else TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(unsigned long long, OBJ, CODE)
#else
#define TECA_PY_ARRAY_SCALAR_DISPATCH_I32(OBJ, CODE)                        \
    TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(int, OBJ, CODE)                      \
    else TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(unsigned int, OBJ, CODE)        \
    else TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(long, OBJ, CODE)                \
    else TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(unsigned long, OBJ, CODE)

#define TECA_PY_ARRAY_SCALAR_DISPATCH_I64(OBJ, CODE)                        \
    TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(long long, OBJ, CODE)                \
    else TECA_PY_ARRAY_SCALAR_DISPATCH_CASE(unsigned long long, OBJ, CODE)
#endif

#define TECA_PY_ARRAY_SCALAR_DISPATCH_I(OBJ, CODE)                          \
    TECA_PY_ARRAY_SCALAR_DISPATCH_I8(OBJ, CODE)                             \
    else TECA_PY_ARRAY_SCALAR_DISPATCH_I16(OBJ, CODE)                       \
    else TECA_PY_ARRAY_SCALAR_DISPATCH_I32(OBJ, CODE)                       \
    else TECA_PY_ARRAY_SCALAR_DISPATCH_I64(OBJ, CODE)

#define TECA_PY_ARRAY_SCALAR_DISPATCH(OBJ, CODE)                            \
    TECA_PY_ARRAY_SCALAR_DISPATCH_FP(OBJ, CODE)                             \
    else TECA_PY_ARRAY_SCALAR_DISPATCH_I(OBJ, CODE)
/// @endcond


/// Append values from the object to the variant array.
TECA_EXPORT
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

/// Copy values from the object into variant array.
TECA_EXPORT
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

/// Set i'th element of the variant array to the value of the object.
TECA_EXPORT
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

/// Construct a new variant array and initialize it with a copy of the object.
TECA_EXPORT
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

/// Construct a new numpy array initialized with the contents of the variant array.
template <typename NT>
TECA_EXPORT
PyArrayObject *new_object(teca_variant_array_impl<NT> *varrt)
{
    // allocate a buffer
    npy_intp n_elem = varrt->size();
    size_t n_bytes = n_elem*sizeof(NT);
    NT *mem = static_cast<NT*>(malloc(n_bytes));
    if (!mem)
    {
        TECA_PY_ERROR(PyExc_RuntimeError,
            "failed to allocate " << n_bytes << " bytes")
        return nullptr;
    }

    // copy the data
    auto spvarrt = varrt->get_cpu_accessible();
    NT *pvarrt = spvarrt.get();
    memcpy(mem, pvarrt, n_bytes);

    // put the buffer in to a new numpy object
    PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNewFromData(1, &n_elem, numpy_tt<NT>::code, mem));
    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_OWNDATA);

    return arr;
}

/// Construct a new numpy array initialized with the contents of the variant array.
TECA_EXPORT
PyArrayObject *new_object(teca_variant_array *varr)
{
    TEMPLATE_DISPATCH(teca_variant_array_impl, varr,
        TT *varrt = static_cast<TT*>(varr);
        return new_object(varrt);
        )
    return nullptr;
}

/** Delete an instance of Numpy ArrayInterface structure created by
 * ::new_array_interface
 */
template <typename NT>
TECA_EXPORT
void delete_array_interface(PyObject *cap)
{
    std::cerr << "==== delete_array_interface ====" << std::endl;

    // relerase our vreference to the data
    std::shared_ptr<NT> *ptr = (std::shared_ptr<NT>*)PyCapsule_GetContext(cap);
    ptr->reset();
    delete ptr;

    //(*ptr) = nullptr;

    // free the ArrayIntergface
    PyArrayInterface *nai = (PyArrayInterface*)
        PyCapsule_GetPointer(cap, nullptr);

    free(nai->shape);
    free(nai);

    std::cerr << "cap = " << cap << " nai = "  << nai
        << std::endl;
}

/** Creates an instance of Numpy ArrayInterface structure pointing to data
 * from the passed ::teca_variant_array
 */
// **************************************************************************
TECA_EXPORT
PyObject *new_array_interface(teca_variant_array *varr)
{
    std::cerr << "==== new_array_interface ====" << std::endl;

    TEMPLATE_DISPATCH(teca_variant_array_impl, varr,
        TT *varrt = static_cast<TT*>(varr);

        // get a pointer to the data. this will ensure that the passed data lives
        // at least as long as the ArrayInterface structure iteself. However,
        // Numpy takes a reference to the PyObject providing the ArrayInterface
        // and it is that reference will keep the data alive.
        std::shared_ptr<NT> *ptr = new std::shared_ptr<NT>();
        (*ptr) = varrt->get_cpu_accessible();

        // calculate the shape and stride
        npy_intp nx = varrt->size();
        npy_intp ny = 1;
        npy_intp nz = 1;

        int nd = 1 + (ny > 1 ? 1 : 0) + (nz > 1 ? 1 : 0);

        npy_intp *tmp = (npy_intp*)malloc(2*nd*sizeof(npy_intp));
        npy_intp *shape = tmp;
        npy_intp *stride = tmp + nd;

        int q = 0;
        if (nz > 1)
        {
            shape[q] = nz;
            stride[q] = nx*ny*sizeof(NT);
            ++q;
        }

        if (ny > 1)
        {
            shape[q] = ny;
            stride[q] = nx*sizeof(NT);
            ++q;
        }

        shape[q] = nx;
        stride[q] = sizeof(NT);

        // construct the array interface
        PyArrayInterface *nai = (PyArrayInterface*)
            malloc(sizeof(PyArrayInterface));

        memset(nai, 0, sizeof(PyArrayInterface));

        nai->two = 2;
        nai->nd = nd;
        nai->typekind = numpy_tt<NT>::typekind();
        nai->itemsize = sizeof(NT);
        nai->shape = shape;
        nai->strides = stride;
        nai->data = (*ptr).get();
        nai->flags = NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ALIGNED |
            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE;

        // package into a capsule
        PyObject *cap = PyCapsule_New(nai, nullptr, delete_array_interface<NT>);

        // save the shared pointer to the data which keeps it alive as long
        // as the capsule.
        PyCapsule_SetContext(cap, ptr);

        std::cerr << "cap = " << cap << " nai = "  << nai
            << " ptr = " << (long long)((*ptr).get()) << std::endl;

        return cap;
        )
    TECA_PY_ERROR(PyExc_RuntimeError, "Invalid array type")
    return nullptr;
}

}
#endif
