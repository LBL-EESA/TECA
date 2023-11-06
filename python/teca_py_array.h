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

#include "teca_py_gil_state.h"
#include "teca_py_array_interface.h"

#include <hamr_python_deleter.h>

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

#define teca_py_array_numpy_tt_declare(CODE, CPP_T) \
template <> struct numpy_tt<CPP_T>                  \
{                                                   \
    enum { code = CODE };                           \
    static bool is_type(PyArrayObject *arr)         \
    { return PyArray_TYPE(arr) == CODE; }           \
};
teca_py_array_numpy_tt_declare(NPY_INT8, char)
teca_py_array_numpy_tt_declare(NPY_INT16, short)
teca_py_array_numpy_tt_declare(NPY_INT32, int)
teca_py_array_numpy_tt_declare(NPY_LONG, long)
teca_py_array_numpy_tt_declare(NPY_INT64, long long)
teca_py_array_numpy_tt_declare(NPY_UINT8, unsigned char)
teca_py_array_numpy_tt_declare(NPY_UINT16, unsigned short)
teca_py_array_numpy_tt_declare(NPY_UINT32, unsigned int)
teca_py_array_numpy_tt_declare(NPY_ULONG, unsigned long)
teca_py_array_numpy_tt_declare(NPY_UINT64, unsigned long long)
teca_py_array_numpy_tt_declare(NPY_FLOAT, float)
teca_py_array_numpy_tt_declare(NPY_DOUBLE, double)

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
        VARIANT_ARRAY_DISPATCH( varr,
            TT *varrt = static_cast<TT*>(varr);
            varrt->reserve(n_elem);
            TECA_PY_ARRAY_DISPATCH(arr,
                if ((PyArray_CHKFLAGS(arr, NPY_ARRAY_C_CONTIGUOUS) ||
                    PyArray_CHKFLAGS(arr, NPY_ARRAY_F_CONTIGUOUS)) &&
                    PyArray_CHKFLAGS(arr, NPY_ARRAY_ALIGNED))
                {
                    // the data is continuous, batch transfer
                    AT *ptr = (AT*)PyArray_DATA(arr);
                    varrt->append(ptr, 0, n_elem);
                }
                else
                {
                    // not continuous, send element by element
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
                }
                return true;
                )
            )

        // unknown element type
        return false;
    }

    // numpy scalar
    if (PyArray_CheckScalar(obj))
    {
        VARIANT_ARRAY_DISPATCH( varr,
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
        VARIANT_ARRAY_DISPATCH( varr,
            TT *varrt = static_cast<TT*>(varr);
            TECA_PY_ARRAY_SCALAR_DISPATCH(obj,
                varrt->set(i, teca_py_array::numpy_scalar_tt<ST>::value(obj));
                return true;
                )
            )
    }
    return false;
}

/** Construct a new variant array and initialize it from the object. The data
 * is transferred by zero-copy operation if possible and if not then a deep copy
 * is made. One can force a deep copy. Returns a new variant array if
 * successful and a nullptr otherwise.
 */
TECA_EXPORT
p_teca_variant_array new_variant_array(PyObject *obj, bool deep_copy = false)
{
    if (PyArray_Check(obj))
    {
        // numpy array
        PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(obj);

        // verify that zero-copy is possible, force a deep copy if it is not
        if (!deep_copy &&
            !((PyArray_CHKFLAGS(arr, NPY_ARRAY_C_CONTIGUOUS) ||
            PyArray_CHKFLAGS(arr, NPY_ARRAY_F_CONTIGUOUS)) &&
            PyArray_CHKFLAGS(arr, NPY_ARRAY_ALIGNED)))
            deep_copy = true;

        TECA_PY_ARRAY_DISPATCH(arr,
            size_t n_elem = PyArray_SIZE(arr);

            p_teca_variant_array_impl<AT> varrt;

            if (deep_copy)
            {
#if defined(TECA_DEBUG)
                std::cerr << "teca_py_array::new_variant_array deep copy" << std::endl;
#endif
                // make a deep copy of the data
                varrt = teca_variant_array_impl<AT>::New();
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
            }
            else
            {
#if defined(TECA_DEBUG)
                std::cerr << "teca_py_array::new_variant_array zero copy" << std::endl;
#endif
                // pass by zero-copy and hold a reference to obj
                AT *ptr = (AT*)PyArray_DATA(arr);
                varrt = teca_variant_array_impl<AT>::New(n_elem,
                    ptr, teca_variant_array::allocator::malloc,
                    -1, hamr::python_deleter(ptr, n_elem, obj));
            }

            return varrt;
            )
    }
    else if (PyArray_CheckScalar(obj))
    {
        // numpy scalar
        TECA_PY_ARRAY_SCALAR_DISPATCH(obj,
            return teca_variant_array_impl<ST>::New(1,
                teca_py_array::numpy_scalar_tt<ST>::value(obj));
            )
    }
    // unknown type
    return nullptr;
}

/// a destructor used with the following funciton
template <typename NT>
void variant_array_capsule_destrcutor(PyObject *cap)
{
    teca_py_gil_state gil;

    using ptr_t = std::shared_ptr<NT>;
    ptr_t *ptr = (ptr_t*)PyCapsule_GetPointer(cap, nullptr);

#if defined(TECA_DEBUG)
    std::cerr << "variant_array_capsule_dsestructor "
        << size_t(ptr->get()) << std::endl;
#endif

    delete ptr;
}

/// Construct a new numpy array initialized with the contents of the variant array.
template <typename NT>
TECA_EXPORT
PyArrayObject *new_object(teca_variant_array_impl<NT> *varrt, bool deep_copy = false)
{
    PyArrayObject *obj = nullptr;

    if (deep_copy)
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

        // deep copy the data
        auto spvarrt = varrt->get_host_accessible();

        if (!varrt->host_accessible())
            varrt->synchronize();

        const NT *pvarrt = spvarrt.get();
        memcpy(mem, pvarrt, n_bytes);

        // put the buffer in to a new numpy object
        obj = reinterpret_cast<PyArrayObject*>(
            PyArray_SimpleNewFromData(1, &n_elem, numpy_tt<NT>::code, mem));
        PyArray_ENABLEFLAGS(obj, NPY_ARRAY_OWNDATA);

#if defined(TECA_DEBUG)
        std::cerr << "new_object (deep copy)" << std::endl;
#endif
    }
    else
    {
        using ptr_t = std::shared_ptr<const NT>;

        // get a pointer to the data
        ptr_t *ptr = new ptr_t(std::move(varrt->get_host_accessible()));

        // create a new numpy array passing the pointer to the data to share
        npy_intp n_elem = varrt->size();

        obj = reinterpret_cast<PyArrayObject*>(
            PyArray_SimpleNewFromData(1, &n_elem, numpy_tt<NT>::code, (void*)ptr->get()));

        // package in a capsule. this holds the reference to keep it alive while
        // Numpy is using it.
        PyObject *cap = PyCapsule_New(ptr, nullptr, variant_array_capsule_destrcutor<const NT>);

        PyArray_SetBaseObject(obj, cap);

        PyArray_ENABLEFLAGS(obj, NPY_ARRAY_NOTSWAPPED |
            NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);

        if (!varrt->host_accessible())
            varrt->synchronize();

#if defined(TECA_DEBUG)
        std::cerr << "new_object (zero copy) " << size_t(ptr->get()) << std::endl;
#endif
    }

    return obj;
}

/// Construct a new numpy array initialized with the contents of the variant array.
TECA_EXPORT
PyArrayObject *new_object(teca_variant_array *varr, bool deep_copy = false)
{
    VARIANT_ARRAY_DISPATCH(varr,
        TT *varrt = static_cast<TT*>(varr);
        return new_object(varrt);
        )
    return nullptr;
}

};

#endif
