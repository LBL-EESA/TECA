#ifndef teca_py_object_h
#define teca_py_object_h

#include "teca_common.h"
#include "teca_variant_array.h"
#include "teca_py_string.h"
#include "teca_py_integer.h"
#include <Python.h>

namespace teca_py_object
{
/// teca_py_object::cpp_tt, A traits class for working with PyObject's
/**
if know the Python type tag then this class gives you:

::type -- C++ type that can hold the value of the PyObject.
::is_type -- returns true if the given PyObject has this type
::value -- convert given PyObject to its C++ type

Python type tags and their coresponding PyObject's are:
int --> PyInt/PyLong, long --> PyInt/PyLong, bool --> PyBool,
float --> PyFloat, char* --> PyUnicode/PyString
*/
template <typename py_t> struct cpp_tt
{};

/*
PY_T -- C-name of python type
CPP_T -- underlying type needed to store it on the C++ side
PY_CHECK -- function that verifies the PyObject is this type
PY_AS_CPP -- function that converts to the C++ type */
#define teca_py_object_cpp_tt_declare(PY_T, CPP_T, PY_CHECK, PY_AS_CPP) \
template <> struct cpp_tt<PY_T>                                         \
{                                                                       \
    typedef CPP_T type;                                                 \
    static bool is_type(PyObject *obj) { return PY_CHECK(obj); }        \
    static type value(PyObject *obj) { return PY_AS_CPP(obj); }         \
};
teca_py_object_cpp_tt_declare(int, long, PyIntegerCheck, PyIntegerToCInt)
teca_py_object_cpp_tt_declare(long, long, PyLongCheck, PyLongToCLong)
teca_py_object_cpp_tt_declare(float, double, PyFloat_Check, PyFloat_AsDouble)
teca_py_object_cpp_tt_declare(char*, std::string, PyStringCheck, PyStringToCString)
teca_py_object_cpp_tt_declare(bool, int, PyBool_Check, PyIntegerToCInt)

/// py_tt, traits class for working with PyObject's
/**
if you know the C++ type then this class gives you:

::tag -- Use this in teca_py_object::cpp_t to find
         the PyObject indentification and conversion
         methods. see example below.

::new_object -- copy construct a new PyObject

here is an example of looking up the PyObject conversion
function(value) from a known C++ type (float).

float val = cpp_tt<py_tt<float>::tag>::value(obj);

py_tt is used to take a C++ type and lookup the Python type
tag. Then the type tag is used to lookup the function.
*/
template <typename type> struct py_tt
{};

/**
CPP_T -- underlying type needed to store it on the C++ side
CPP_AS_PY -- function that converts from the C++ type */
#define teca_py_object_py_tt_declare(CPP_T, PY_T, CPP_AS_PY)\
template <> struct py_tt<CPP_T>                             \
{                                                           \
    typedef PY_T tag;                                       \
    static PyObject *new_object(CPP_T val)                  \
    { return CPP_AS_PY(val); }                              \
};
teca_py_object_py_tt_declare(char, int, CIntToPyInteger)
teca_py_object_py_tt_declare(short, int, CIntToPyInteger)
teca_py_object_py_tt_declare(int, int, CIntToPyInteger)
teca_py_object_py_tt_declare(long, int, CIntToPyInteger)
teca_py_object_py_tt_declare(long long, int, CIntLLToPyInteger)
teca_py_object_py_tt_declare(unsigned char, int, CIntUToPyInteger)
teca_py_object_py_tt_declare(unsigned short, int, CIntUToPyInteger)
teca_py_object_py_tt_declare(unsigned int, int, CIntUToPyInteger)
teca_py_object_py_tt_declare(unsigned long, int, CIntUToPyInteger)
teca_py_object_py_tt_declare(unsigned long long, int, CIntULLToPyInteger)
teca_py_object_py_tt_declare(float, float, PyFloat_FromDouble)
teca_py_object_py_tt_declare(double, float, PyFloat_FromDouble)
// strings are a special case
template <> struct py_tt<std::string>
{
    typedef char* tag;
    static PyObject *new_object(const std::string &s)
    { return CStringToPyString(s.c_str()); }
};
// TODO -- special case for teca_metadata

// dispatch macro.
// OBJ -- PyObject* instance
// CODE -- code block to execute on match
// OT -- a typedef to the match type available in
//       the code block
#define TECA_PY_OBJECT_DISPATCH_CASE(CPP_T, PY_OBJ, CODE)   \
    if (teca_py_object::cpp_tt<CPP_T>::is_type(PY_OBJ))     \
    {                                                       \
        using OT = CPP_T;                                   \
        CODE                                                \
    }

#define TECA_PY_OBJECT_DISPATCH(PY_OBJ, CODE)               \
    TECA_PY_OBJECT_DISPATCH_CASE(int, PY_OBJ, CODE)         \
    else TECA_PY_OBJECT_DISPATCH_CASE(float, PY_OBJ, CODE)  \
    else TECA_PY_OBJECT_DISPATCH_CASE(char*, PY_OBJ, CODE)  \
    else TECA_PY_OBJECT_DISPATCH_CASE(long, PY_OBJ, CODE)

// without string
#define TECA_PY_OBJECT_DISPATCH_NUM(PY_OBJ, CODE)           \
    TECA_PY_OBJECT_DISPATCH_CASE(int, PY_OBJ, CODE)         \
    else TECA_PY_OBJECT_DISPATCH_CASE(float, PY_OBJ, CODE)  \
    else TECA_PY_OBJECT_DISPATCH_CASE(long, PY_OBJ, CODE)

// just string
#define TECA_PY_OBJECT_DISPATCH_STR(PY_OBJ, CODE)           \
    TECA_PY_OBJECT_DISPATCH_CASE(char*, PY_OBJ, CODE)

// ****************************************************************************
p_teca_variant_array new_variant_array(PyObject *obj)
{
    TECA_PY_OBJECT_DISPATCH(obj,

        p_teca_variant_array_impl<typename cpp_tt<OT>::type> varr
            = teca_variant_array_impl<typename cpp_tt<OT>::type>::New(1);

        varr->set(0, cpp_tt<OT>::value(obj));

        return varr;
        )

    return nullptr;
}

// ****************************************************************************
bool copy(teca_variant_array *varr, PyObject *obj)
{
    TEMPLATE_DISPATCH(teca_variant_array_impl, varr,
        TT *varrt = static_cast<TT*>(varr);
        TECA_PY_OBJECT_DISPATCH_NUM(obj,
            varrt->resize(1);
            varrt->set(0, cpp_tt<OT>::value(obj));
            return true;
            )
        )
    else TEMPLATE_DISPATCH_CASE(teca_variant_array_impl,
        std::string, varr,
        TT *varrt = static_cast<TT*>(varr);
        TECA_PY_OBJECT_DISPATCH_STR(obj,
            varrt->resize(1);
            varrt->set(0, cpp_tt<OT>::value(obj));
            return true;
            )
        )

    return false;
}

// ****************************************************************************
bool set(teca_variant_array *varr, unsigned long i, PyObject *obj)
{
    TEMPLATE_DISPATCH(teca_variant_array_impl, varr,
        TT *varrt = static_cast<TT*>(varr);
        TECA_PY_OBJECT_DISPATCH_NUM(obj,
            varrt->set(i, cpp_tt<OT>::value(obj));
            return true;
            )
        )
    else TEMPLATE_DISPATCH_CASE(teca_variant_array_impl,
        std::string, varr,
        TT *varrt = static_cast<TT*>(varr);
        TECA_PY_OBJECT_DISPATCH_STR(obj,
            varrt->set(i, cpp_tt<OT>::value(obj));
            return true;
            )
        )

    return false;
}

// ****************************************************************************
bool append(teca_variant_array *varr, PyObject *obj)
{
    TEMPLATE_DISPATCH(teca_variant_array_impl, varr,
        TT *varrt = static_cast<TT*>(varr);
        TECA_PY_OBJECT_DISPATCH_NUM(obj,
            varrt->append(static_cast<NT>(cpp_tt<OT>::value(obj)));
            return true;
            )
        )
    else TEMPLATE_DISPATCH_CASE(teca_variant_array_impl,
        std::string, varr,
        TT *varrt = static_cast<TT*>(varr);
        TECA_PY_OBJECT_DISPATCH_STR(obj,
            varrt->append(static_cast<NT>(cpp_tt<OT>::value(obj)));
            return true;
            )
         )
    return false;
}

// container that keeps a reference to a PyObject
class teca_py_object_ptr
{
public:
    teca_py_object_ptr() : m_obj(nullptr) {}

    teca_py_object_ptr(PyObject *obj)
        : m_obj(obj) { Py_XINCREF(m_obj); }

     virtual ~teca_py_object_ptr() { Py_XDECREF(m_obj); }

     teca_py_object_ptr(teca_py_object_ptr &&o)
        : m_obj(o.m_obj) { o.m_obj = nullptr; }

     teca_py_object_ptr &operator=(teca_py_object_ptr &&o)
     {
         PyObject *tmp = m_obj;
         m_obj = o.m_obj;
         o.m_obj = tmp;
         return *this;
     }

     teca_py_object_ptr(const teca_py_object_ptr &o)
        : m_obj(o.m_obj) { Py_XINCREF(m_obj); }

     teca_py_object_ptr &operator=(const teca_py_object_ptr &o)
     {
         Py_XINCREF(o.m_obj);
         Py_XDECREF(m_obj);
         m_obj = o.m_obj;
         return *this;
     }

     explicit operator bool () const
     { return m_obj != nullptr; }

    PyObject *get_object(){ return m_obj; }

    virtual void set_object(PyObject *obj)
    {
        Py_XINCREF(obj);
        Py_XDECREF(m_obj);
        m_obj = obj;
    }

private:
    PyObject *m_obj;
};

class teca_py_callable : public teca_py_object_ptr
{
public:
    teca_py_callable() : teca_py_object_ptr() {}

    virtual ~teca_py_callable()
    { this->teca_py_object_ptr::set_object(nullptr); }

    teca_py_callable(PyObject *f) : teca_py_object_ptr()
    { this->teca_py_callable::set_object(f); }

    teca_py_callable(const teca_py_callable &&o)
        : teca_py_object_ptr(std::move(o)) {}

    teca_py_callable &operator=(const teca_py_callable &&o)
    {
        this->teca_py_object_ptr::operator=(std::move(o));
        return *this;
    }

    teca_py_callable(const teca_py_callable &o)
        : teca_py_object_ptr(o) {}

    teca_py_callable &operator=(const teca_py_callable &o)
    {
        this->teca_py_object_ptr::operator=(o);
        return *this;
    }

    virtual void set_object(PyObject *f)
    {
        if (PyCallable_Check(f))
            this->teca_py_object_ptr::set_object(f);
        else
            PyErr_Format(PyExc_TypeError,
                "object is not callable");
    }
};

};

#endif
