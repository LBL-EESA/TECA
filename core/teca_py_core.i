%define TECA_PY_CORE_DOC
"TECA core module

The core module contains the pipeline and executive
as well as metadata object, variant array and abstract
datasets.
"
%enddef
%module (docstring=TECA_PY_CORE_DOC) teca_py_core

%{
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL  PyArray_API_teca_py_core
#include <numpy/arrayobject.h>
#include <sstream>
#include <vector>
#include <Python.h>

#include "teca_algorithm_executive.h"
#include "teca_time_step_executive.h"
#include "teca_metadata.h"
#include "teca_algorithm.h"
#include "teca_threaded_algorithm.h"
#include "teca_temporal_reduction.h"
#include "teca_variant_array.h"

#include "teca_py_object.h"
#include "teca_py_sequence.h"
#include "teca_py_array.h"
%}

%init %{
import_array();
%}

%include "teca_py_common.i"
%include "teca_py_shared_ptr.i"
%include <std_string.i>
%include <std_vector.i>
%include <std_pair.i>

%template(std_vector_char) std::vector<char>;
%template(std_vector_uchar) std::vector<unsigned char>;
%template(std_vector_int) std::vector<int>;
%template(std_vector_uint) std::vector<unsigned int>;
%template(std_vector_long_long) std::vector<long long>;
%template(std_vector_ulong_long) std::vector<unsigned long long>;
%template(std_vector_float) std::vector<float>;
%template(std_vector_double) std::vector<double>;
%template(std_vector_string) std::vector<std::string>;



/***************************************************************************
 metadata
 ***************************************************************************/
%ignore teca_metadata::teca_metadata(teca_metadata &&);
%ignore teca_metadata::operator=;
%ignore operator<(const teca_metadata &, const teca_metadata &);
%ignore operator&(const teca_metadata &, const teca_metadata &);
%ignore operator==(const teca_metadata &, const teca_metadata &);
%ignore operator!=(const teca_metadata &, const teca_metadata &);
%ignore teca_metadata::insert;
%ignore teca_metadata::set; /* use __setitem__ instead */
%ignore teca_metadata::get; /* use __getitem__ instead */
%include "teca_metadata.h"
%extend teca_metadata
{
    PY_TECA_STR()

    /* pythonic insert: md['name'] = value */
    void __setitem__(const std::string &name, PyObject *value)
    {
        p_teca_variant_array varr;
        if ((varr = teca_py_object::new_variant_array(value))
            || (varr = teca_py_array::new_variant_array(value))
            || (varr = teca_py_sequence::new_variant_array(value)))
        {
            self->insert(name, varr);
            return;
        }

        PyErr_Format(PyExc_TypeError,
            "Failed to convert value for key \"%s\"", name.c_str());
    }

    /* pythonic lookup: md['name'] */
    PyObject *__getitem__(const std::string &name)
    {
        p_teca_variant_array varr = self->get(name);
        if (!varr)
        {
            PyErr_Format(PyExc_KeyError,
                "key \"%s\" not found", name.c_str());
            return nullptr;
        }

        size_t n_elem = varr->size();
        if (n_elem == 1)
        {
            TEMPLATE_DISPATCH(teca_variant_array_impl,
                varr.get(),
                TT *varrt = static_cast<TT*>(varr.get());
                return teca_py_object::py_tt<NT>::new_object(varrt->get(0));
                )
            else TEMPLATE_DISPATCH_CASE(const teca_variant_array_impl,
                std::string, varr.get(),
                TT *varrt = static_cast<TT*>(varr.get());
                return teca_py_object::py_tt<NT>::new_object(varrt->get(0));
                )
            else TEMPLATE_DISPATCH_CASE(const teca_variant_array_impl,
                teca_metadata, varr.get(),
                TT *varrt = static_cast<TT*>(varr.get());
                teca_metadata *md = new teca_metadata(varrt->get(0));
                return SWIG_NewPointerObj(SWIG_as_voidptr(md),
                     SWIGTYPE_p_teca_metadata, SWIG_POINTER_NEW|0);
                )
        }
        else if (n_elem > 1)
        {
            TEMPLATE_DISPATCH(teca_variant_array_impl,
                varr.get(),
                TT *varrt = static_cast<TT*>(varr.get());
                return reinterpret_cast<PyObject*>(
                    teca_py_array::new_object(varrt));
                )
            else TEMPLATE_DISPATCH_CASE(const teca_variant_array_impl,
                std::string, varr.get(),
                TT *varrt = static_cast<TT*>(varr.get());
                PyObject *list = PyList_New(n_elem);
                for (size_t i = 0; i < n_elem; ++i)
                {
                    PyList_SET_ITEM(list, i,
                        teca_py_object::py_tt<NT>::new_object(varrt->get(i)));
                }
                return list;
                )
            else TEMPLATE_DISPATCH_CASE(const teca_variant_array_impl,
                teca_metadata, varr.get(),
                TT *varrt = static_cast<TT*>(varr.get());
                PyObject *list = PyList_New(n_elem);
                for (size_t i = 0; i < n_elem; ++i)
                {
                    teca_metadata *md = new teca_metadata(varrt->get(i));
                    PyList_SET_ITEM(list, i,
                        SWIG_NewPointerObj(SWIG_as_voidptr(md),
                            SWIGTYPE_p_teca_metadata, SWIG_POINTER_NEW|0));
                }
                return list;
                )
        }

        return PyErr_Format(PyExc_TypeError,
            "Failed to convert value for key \"%s\"", name.c_str());
    }
}

/***************************************************************************
 algorithm_executive
 ***************************************************************************/
%ignore teca_algorithm_executive::shared_from_this;
%ignore std::enable_shared_from_this<teca_algorithm_executive>;
%shared_ptr(std::enable_shared_from_this<teca_algorithm_executive>)
%shared_ptr(teca_algorithm_executive)
class teca_algorithm_executive;
%template(teca_algorithm_executive_base) std::enable_shared_from_this<teca_algorithm_executive>;
%ignore teca_algorithm_executive::operator=;
%include "teca_common.h"
%include "teca_shared_object.h"
%include "teca_algorithm_executive_fwd.h"
%include "teca_algorithm_executive.h"

/***************************************************************************
 time_step_executive
 ***************************************************************************/
%ignore teca_time_step_executive::shared_from_this;
%shared_ptr(teca_time_step_executive)
%ignore teca_time_step_executive::operator=;
%include "teca_time_step_executive.h"

/***************************************************************************
 algorithm
 ***************************************************************************/
%ignore teca_algorithm::shared_from_this;
%ignore std::enable_shared_from_this<teca_algorithm>;
%shared_ptr(std::enable_shared_from_this<teca_algorithm>)
%shared_ptr(teca_algorithm)
class teca_algorithm;
%template(teca_algorithm_base) std::enable_shared_from_this<teca_algorithm>;
typedef std::pair<std::shared_ptr<teca_algorithm>, unsigned int> teca_algorithm_output_port;
%template(teca_output_port_type) std::pair<std::shared_ptr<teca_algorithm>, unsigned int>;
%include "teca_common.h"
%include "teca_shared_object.h"
%include "teca_algorithm_fwd.h"
%include "teca_program_options.h"
%include "teca_algorithm.h"

/***************************************************************************
 threaded_algorithm
 ***************************************************************************/
%ignore teca_threaded_algorithm::shared_from_this;
%shared_ptr(teca_threaded_algorithm)
%ignore teca_threaded_algorithm::operator=;
%include "teca_threaded_algorithm.h"

/***************************************************************************
 temporal_reduction
 ***************************************************************************/
%ignore teca_temporal_reduction::shared_from_this;
%shared_ptr(teca_temporal_reduction)
%ignore teca_temporal_reduction::operator=;
%include "teca_temporal_reduction.h"

/***************************************************************************
 variant_array
 ***************************************************************************/
%ignore teca_variant_array::shared_from_this;
%ignore std::enable_shared_from_this<teca_variant_array>;
%shared_ptr(std::enable_shared_from_this<teca_variant_array>)
%shared_ptr(teca_variant_array)
%shared_ptr(teca_variant_array_impl<double>)
%shared_ptr(teca_variant_array_impl<float>)
%shared_ptr(teca_variant_array_impl<char>)
%shared_ptr(teca_variant_array_impl<int>)
%shared_ptr(teca_variant_array_impl<long long>)
%shared_ptr(teca_variant_array_impl<unsigned char>)
%shared_ptr(teca_variant_array_impl<unsigned int>)
%shared_ptr(teca_variant_array_impl<unsigned long long>)
%shared_ptr(teca_variant_array_impl<std::string>)
class teca_variant_array;
%template(teca_variant_array_base) std::enable_shared_from_this<teca_variant_array>;
%include "teca_common.h"
%include "teca_shared_object.h"
%include "teca_variant_array_fwd.h"
%ignore teca_variant_array::operator=;
%ignore teca_variant_array_factory;
%ignore teca_variant_array::append(const teca_variant_array &other);
%ignore teca_variant_array::append(const const_p_teca_variant_array &other);
%ignore copy(const teca_variant_array &other);
%include "teca_variant_array.h"
%template(teca_double_array) teca_variant_array_impl<double>;
%template(teca_float_array) teca_variant_array_impl<float>;
%template(teca_int_array) teca_variant_array_impl<char>;
%template(teca_char_array) teca_variant_array_impl<int>;
%template(teca_long_long_array) teca_variant_array_impl<long long>;
%template(teca_unsigned_int_array) teca_variant_array_impl<unsigned char>;
%template(teca_unsigned_char_array) teca_variant_array_impl<unsigned int>;
%template(teca_unsigned_long_long_array) teca_variant_array_impl<unsigned long long>;
%extend teca_variant_array
{
    PY_TECA_STR()

    void __setitem__(unsigned long i, PyObject *value)
    {
        if (teca_py_object::set(self, i, value))
            return;

        PyErr_Format(PyExc_TypeError,
            "failed to set value at index %lu", i);
    }

    PyObject *__getitem__(unsigned long i)
    {
        TEMPLATE_DISPATCH(teca_variant_array_impl, self,
            TT *varrt = static_cast<TT*>(self);
            return teca_py_object::py_tt<NT>::new_object(varrt->get(i));
            )
        else TEMPLATE_DISPATCH_CASE(teca_variant_array_impl,
            std::string, self,
            TT *varrt = static_cast<TT*>(self);
            return teca_py_object::py_tt<NT>::new_object(varrt->get(i));
            )

        PyErr_Format(PyExc_TypeError,
            "failed to set value at index %lu", i);
        return nullptr;
    }

    PyObject *as_array()
    {
        return reinterpret_cast<PyObject*>(
            teca_py_array::new_object(self));
    }

    void append(PyObject *obj)
    {
        if (teca_py_object::append(self, obj)
            || teca_py_array::append(self, obj)
            || teca_py_sequence::append(self, obj))
            return;

        PyErr_Format(PyExc_TypeError,
            "Failed to convert value");
    }

    void copy(PyObject *obj)
    {
        if (teca_py_object::copy(self, obj)
            || teca_py_array::copy(self, obj)
            || teca_py_sequence::copy(self, obj))
            return;

        PyErr_Format(PyExc_TypeError,
            "Failed to convert value");
    }
}
