%{
#include <vector>

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
#include "teca_py_iterator.h"
#include "teca_py_gil_state.h"
%}

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
    static
    p_teca_variant_array New(PyObject *obj)
    {
        teca_py_gil_state gil;

        p_teca_variant_array varr;
        if ((varr = teca_py_object::new_variant_array(obj))
            || (varr = teca_py_array::new_variant_array(obj))
            || (varr = teca_py_sequence::new_variant_array(obj))
            || (varr = teca_py_iterator::new_variant_array(obj)))
            return varr;

        PyErr_Format(PyExc_TypeError, "Failed to convert value");
        return nullptr;
    }

    TECA_PY_STR()

    unsigned long __len__()
    { return self->size(); }

    void __setitem__(unsigned long i, PyObject *value)
    {
        teca_py_gil_state gil;

#ifndef NDEBUG
        if (i >= self->size())
        {
            PyErr_Format(PyExc_IndexError,
                "index %lu is out of bounds in teca_variant_array "
                " with size %lu", i, self->size());
            return;
        }
#endif
        if (teca_py_object::set(self, i, value))
            return;

        PyErr_Format(PyExc_TypeError,
            "failed to set value at index %lu", i);
    }

    PyObject *__getitem__(unsigned long i)
    {
        teca_py_gil_state gil;

        if (i >= self->size())
        {
            PyErr_Format(PyExc_IndexError,
                "index %lu is out of bounds in teca_variant_array "
                " with size %lu", i, self->size());
            return nullptr;
        }

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
            "failed to get value at index %lu", i);
        return nullptr;
    }

    PyObject *as_array()
    {
        teca_py_gil_state gil;

        return reinterpret_cast<PyObject*>(
            teca_py_array::new_object(self));
    }

    void append(PyObject *obj)
    {
        teca_py_gil_state gil;

        if (teca_py_object::append(self, obj)
            || teca_py_array::append(self, obj)
            || teca_py_sequence::append(self, obj))
            return;

        PyErr_Format(PyExc_TypeError,
            "Failed to convert value");
    }

    void copy(PyObject *obj)
    {
        teca_py_gil_state gil;

        if (teca_py_object::copy(self, obj)
            || teca_py_array::copy(self, obj)
            || teca_py_sequence::copy(self, obj))
            return;

        PyErr_Format(PyExc_TypeError,
            "Failed to convert value");
    }
}
TECA_PY_DYNAMIC_VARIANT_ARRAY_CAST(double, double)
TECA_PY_DYNAMIC_VARIANT_ARRAY_CAST(float, float)
TECA_PY_DYNAMIC_VARIANT_ARRAY_CAST(char, char)
TECA_PY_DYNAMIC_VARIANT_ARRAY_CAST(int, int)
TECA_PY_DYNAMIC_VARIANT_ARRAY_CAST(long long, long_long)
TECA_PY_DYNAMIC_VARIANT_ARRAY_CAST(unsigned char, unsigned_char)
TECA_PY_DYNAMIC_VARIANT_ARRAY_CAST(unsigned int, unsigned_int)
TECA_PY_DYNAMIC_VARIANT_ARRAY_CAST(unsigned long long, unsigned_long_long)

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
    TECA_PY_STR()

    /* md['name'] = value */
    void __setitem__(const std::string &name, PyObject *value)
    {
        teca_py_gil_state gil;

        p_teca_variant_array varr;
        if ((varr = teca_py_object::new_variant_array(value))
            || (varr = teca_py_array::new_variant_array(value))
            || (varr = teca_py_sequence::new_variant_array(value))
            || (varr = teca_py_iterator::new_variant_array(value)))
        {
            self->insert(name, varr);
            return;
        }
        PyErr_Format(PyExc_TypeError,
            "Failed to convert value for key \"%s\"", name.c_str());
    }

    /* return md['name'] */
    PyObject *__getitem__(const std::string &name)
    {
        teca_py_gil_state gil;

        p_teca_variant_array varr = self->get(name);
        if (!varr)
        {
            PyErr_Format(PyExc_KeyError,
                "key \"%s\" not found", name.c_str());
            return nullptr;
        }

        size_t n_elem = varr->size();
        if (n_elem < 1)
        {
            return PyList_New(0);
        }
        else if (n_elem == 1)
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
                return SWIG_NewPointerObj(new teca_metadata(varrt->get(0)),
                     SWIGTYPE_p_teca_metadata, SWIG_POINTER_OWN);
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
                    PyList_SET_ITEM(list, i,
                        SWIG_NewPointerObj(new teca_metadata(varrt->get(i)),
                            SWIGTYPE_p_teca_metadata, SWIG_POINTER_OWN));
                }
                return list;
                )
        }

        return PyErr_Format(PyExc_TypeError,
            "Failed to convert value for key \"%s\"", name.c_str());
    }

    void append(const std::string &name, PyObject *obj)
    {
        teca_py_gil_state gil;

        teca_variant_array *varr = self->get(name).get();
        if (!varr)
        {
            PyErr_Format(PyExc_KeyError,
                "key \"%s\" not found", name.c_str());
            return;
        }

        if (teca_py_object::append(varr, obj)
            || teca_py_array::append(varr, obj)
            || teca_py_sequence::append(varr, obj)
            || teca_py_iterator::append(varr, obj))
            return;

        PyErr_Format(PyExc_TypeError,
            "Failed to convert value");
    }
}
%template(std_vector_metadata) std::vector<teca_metadata>;

/***************************************************************************
 dataset
 ***************************************************************************/
%ignore teca_dataset::shared_from_this;
%ignore std::enable_shared_from_this<teca_dataset>;
%shared_ptr(std::enable_shared_from_this<teca_dataset>)
%shared_ptr(teca_dataset)
class teca_dataset;
%template(teca_dataset_base) std::enable_shared_from_this<teca_dataset>;
%ignore teca_dataset::operator=;
%include "teca_dataset_fwd.h"
%include "teca_dataset.h"
%extend teca_dataset
{
    TECA_PY_STR()
}
%template(std_vector_dataset) std::vector<std::shared_ptr<teca_dataset>>;

/***************************************************************************/
%define TECA_PY_DATASET_METADATA(_type, _name)
    void set_## _name(PyObject *obj)
    {
        teca_py_gil_state gil;

        TECA_PY_OBJECT_DISPATCH_NUM(obj,
            self->set_ ## _name(teca_py_object::cpp_tt<
                teca_py_object::py_tt<_type>::tag>::value(obj));
            return;
            )

        PyErr_Format(PyExc_TypeError,
            "Failed to set property \"%s\"", #_name);
    }

    PyObject *get_## _name()
    {
        teca_py_gil_state gil;

        _type val;
        self->get_ ## _name(val);

        return teca_py_object::py_tt<_type>::new_object(val);
    }
%enddef

/***************************************************************************/
%define TECA_PY_DATASET_VECTOR_METADATA(_type, _name)
    PyObject *get_## _name()
    {
        teca_py_gil_state gil;

        p_teca_variant_array_impl<_type> varr =
             teca_variant_array_impl<_type>::New();

        self->get_## _name(varr);

        PyObject *list = teca_py_sequence::new_object(varr);
        if (!list)
        {
            PyErr_Format(PyExc_TypeError,
                "Failed to get property \"%s\"", # _name);
        }

        return list;
    }

    void set_## _name(PyObject *array)
    {
        teca_py_gil_state gil;

        p_teca_variant_array varr;
        if ((varr = teca_py_array::new_variant_array(array))
            || (varr = teca_py_sequence::new_variant_array(array))
            || (varr = teca_py_iterator::new_variant_array(array)))
        {
            self->set_## _name(varr);
            return;
        }

        PyErr_Format(PyExc_TypeError,
            "Failed to set property \"%s\"", # _name);
    }
%enddef

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

/***************************************************************************/
%define TECA_PY_ALGORITHM_PROPERTY(_type, _name)
    void set_## _name(PyObject *obj)
    {
        teca_py_gil_state gil;

        TECA_PY_OBJECT_DISPATCH_NUM(obj,
            self->set_ ## _name(teca_py_object::cpp_tt<
                teca_py_object::py_tt<_type>::tag>::value(obj));
            return;
            )

        PyErr_Format(PyExc_TypeError,
            "Failed to set property \"%s\"", #_name);
    }

    PyObject *get_## _name()
    {
        teca_py_gil_state gil;

        _type val;
        self->get_ ## _name(val);

        return teca_py_object::py_tt<_type>::new_object(val);
    }
%enddef

/***************************************************************************/
%define TECA_PY_ALGORITHM_VECTOR_PROPERTY(_type, _name)
    PyObject *get_## _name ##s()
    {
        teca_py_gil_state gil;

        p_teca_variant_array_impl<_type> varr =
             teca_variant_array_impl<_type>::New();

        self->get_## _name ##s(varr);

        PyObject *list = teca_py_sequence::new_object(varr);
        if (!list)
        {
            PyErr_Format(PyExc_TypeError,
                "Failed to get property \"%s\"", # _name);
        }

        return list;
    }

    void set_## _name ##s(PyObject *array)
    {
        teca_py_gil_state gil;

        p_teca_variant_array varr;
        if ((varr = teca_py_array::new_variant_array(array))
            || (varr = teca_py_sequence::new_variant_array(array))
            || (varr = teca_py_iterator::new_variant_array(array)))
        {
            self->set_## _name ##s(varr);
            return;
        }

        PyErr_Format(PyExc_TypeError,
            "Failed to set property \"%s\"", # _name);
    }
%enddef

/***************************************************************************
 threaded_algorithm
 ***************************************************************************/
%ignore teca_threaded_algorithm::shared_from_this;
%shared_ptr(teca_threaded_algorithm)
%ignore teca_threaded_algorithm::operator=;
%include "teca_threaded_algorithm_fwd.h"
%include "teca_threaded_algorithm.h"

/***************************************************************************
 temporal_reduction
 ***************************************************************************/
%ignore teca_temporal_reduction::shared_from_this;
%shared_ptr(teca_temporal_reduction)
%ignore teca_temporal_reduction::operator=;
%include "teca_temporal_reduction_fwd.h"
%include "teca_temporal_reduction.h"
