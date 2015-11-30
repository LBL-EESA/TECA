%define MDOC
"TECA core module

The core module contains the pipeline and executive
as well as metadata object, variant array and abstract
datasets.
"
%enddef

%module (docstring=MDOC) py_teca_core

%{
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL  PyArray_API_py_teca_core
#include <numpy/arrayobject.h>
#include <sstream>
#include <vector>
#include <Python.h>
#include "teca_metadata.h"
#include "teca_variant_array.h"
#include "teca_py_object.h"
#include "teca_py_sequence.h"
#include "teca_py_array.h"
%}

%init %{
import_array();
%}

/***************************************************************************/
%include "std_string.i"
%ignore teca_metadata::teca_metadata(teca_metadata &&);
%ignore teca_metadata::operator=;
%ignore operator<(const teca_metadata &, const teca_metadata &);
%ignore operator&(const teca_metadata &, const teca_metadata &);
%ignore operator==(const teca_metadata &, const teca_metadata &);
%ignore operator!=(const teca_metadata &, const teca_metadata &);
%ignore teca_metadata::insert;
%ignore teca_metadata::set; /* use __setitem__ instead */
%ignore teca_metadata::get; /* use __getitem__ instead */
%ignore teca_metadata::resize;
%ignore teca_metadata::to_stream; /* TODO */
%ignore teca_metadata::from_stream; /* TODO */
%include "teca_metadata.h"
%extend teca_metadata
{
    PY_TECA_STR

    /* pythonic insert: md['name'] = value */
    void __setitem__(const std::string &name, PyObject *value)
    {
        /* the order matters here because strings are sequences
           and we dont want to treat them that way. */
        p_teca_variant_array varr;
        if ((varr = teca_py_object::new_copy(value))
            || (varr = teca_py_array::new_copy(value))
            || (varr = teca_py_sequence::new_copy(value)))
            /* TODO copy in teca_metadata */
        {
            self->insert(name, varr);
            return;
        }
        PyErr_Format(PyExc_RuntimeError,
            "failed to insert %s", name.c_str());
    }

    /* pythonic lookup: md['name'] */
    PyObject *__getitem__(const std::string &name)
    {
        p_teca_variant_array varr = self->get(name);
        if (varr)
        {
            TEMPLATE_DISPATCH(teca_variant_array_impl,
                varr.get(),
                TT *varrt = static_cast<TT*>(varr.get());
                size_t n_elem = varrt->size();
                if (n_elem == 1)
                {
                    return teca_py_object::py_tt<NT>::new_object(varrt->get(0));
                }
                else
                if (n_elem > 1)
                {
                    return
                    reinterpret_cast<PyObject*>(teca_py_array::new_copy(varrt));
                }
                )
            else TEMPLATE_DISPATCH_CASE(const teca_variant_array_impl,
                std::string, varr.get(),
                TT *varrt = static_cast<TT*>(varr.get());
                size_t n_elem = varrt->size();
                if (n_elem == 1)
                {
                    return teca_py_object::py_tt<NT>::new_object(varrt->get(0));
                }
                else
                if (n_elem > 1)
                {
                    PyObject *list = PyList_New(n_elem);
                    for (size_t i = 0; i < n_elem; ++i)
                    {
                        PyList_SET_ITEM(list, i,
                            teca_py_object::py_tt<NT>::new_object(varrt->get(i)));
                    }
                    return list;
                }
                )
            else TEMPLATE_DISPATCH_CASE(const teca_variant_array_impl,
                teca_metadata, varr.get(),
                TT *varrt = static_cast<TT*>(varr.get());
                size_t n_elem = varrt->size();
                if (n_elem == 1)
                {
                    teca_metadata *md = new teca_metadata(varrt->get(0));
                    return SWIG_NewPointerObj(SWIG_as_voidptr(md),
                         SWIGTYPE_p_teca_metadata, SWIG_POINTER_NEW|0);
                }
                else
                if (n_elem > 1)
                {
                    PyObject *list = PyList_New(n_elem);
                    for (size_t i = 0; i < n_elem; ++i)
                    {
                        teca_metadata *md = new teca_metadata(varrt->get(i));
                        PyList_SET_ITEM(list, i,
                            SWIG_NewPointerObj(SWIG_as_voidptr(md),
                                SWIGTYPE_p_teca_metadata, SWIG_POINTER_NEW|0));
                    }
                    return list;
                }
                )
        }
        return PyErr_Format(PyExc_KeyError,
            "key \"%s\" not found", name.c_str());
    }
}


/*
 * teca_variant_array
 *
%include <std_shared_ptr.i>
%shared_ptr(teca_variant_array)
%shared_ptr(teca_variant_array_impl<char>)
%shared_ptr(teca_variant_array_impl<int>)
%shared_ptr(teca_variant_array_impl<long long>)
%shared_ptr(teca_variant_array_impl<float>)
%shared_ptr(teca_variant_array_impl<double>)
%inline{
typedef std::shared_ptr<teca_variant_array> p_teca_variant_array;
}
%include "teca_common.h"
%include "teca_shared_object.h"
%include "teca_variant_array_fwd.h"
%ignore teca_variant_array::operator=;
%include "teca_variant_array.h"
%template(teca_variant_array_char) teca_variant_array_impl<char>;
%template(teca_variant_array_int) teca_variant_array_impl<int>;
%template(teca_variant_array_int64) teca_variant_array_impl<long long>;
%template(teca_variant_array_float) teca_variant_array_impl<float>;
%template(teca_variant_array_double) teca_variant_array_impl<double>;
%extend teca_variant_array
{
    const char *__str__()
    {
        static std::ostringstream oss;
        oss.str("");
        self->to_stream(oss);
        return oss.str().c_str();
    }
}*/
