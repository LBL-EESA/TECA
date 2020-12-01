%{
#include <vector>

#include "calcalcs.h"
#include "teca_algorithm_executive.h"
#include "teca_index_executive.h"
#include "teca_metadata.h"
#include "teca_algorithm.h"
#include "teca_threaded_algorithm.h"
#include "teca_thread_util.h"
#include "teca_index_reduce.h"
#include "teca_variant_array.h"
#include "teca_binary_stream.h"
#include "teca_parallel_id.h"
#include "teca_profiler.h"
#include "teca_programmable_algorithm.h"
#include "teca_programmable_reduce.h"
#include "teca_threaded_programmable_algorithm.h"
#include "teca_system_util.h"

#include "teca_py_object.h"
#include "teca_py_sequence.h"
#include "teca_py_array.h"
#include "teca_py_iterator.h"
#include "teca_py_algorithm.h"
#include "teca_py_gil_state.h"
%}

/***************************************************************************
 profiler
 ***************************************************************************/
%include "teca_profiler.h"

/***************************************************************************
 parallel_id
 ***************************************************************************/
%ignore operator<<(std::ostream &, const teca_parallel_id&);
%include "teca_parallel_id.h"
%extend teca_parallel_id
{
    PyObject *__str__()
    {
        (void)self;
        teca_py_gil_state gil;
        std::ostringstream oss;
        oss << teca_parallel_id();
        return CStringToPyString(oss.str().c_str());
    }
}

/***************************************************************************
 binary_stream
 ***************************************************************************/
%ignore teca_binary_stream::teca_binary_stream(teca_binary_stream &&);
%ignore teca_binary_stream::operator=;
%ignore teca_binary_stream::get_data;
%rename teca_binary_stream::get_data_py get_data;
%include "teca_binary_stream.h"
%extend teca_binary_stream
{
    PyObject *get_data_py()
    {
        teca_py_gil_state gil;

        // allocate a buffer
        npy_intp n_bytes = self->size();
        char *mem = static_cast<char*>(malloc(n_bytes));
        if (!mem)
        {
            PyErr_Format(PyExc_RuntimeError,
                "failed to allocate %lu bytes", n_bytes);
            return nullptr;
        }

        // copy the data
        memcpy(mem, self->get_data(), n_bytes);

        // put the buffer in to a new numpy object
        PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(
            PyArray_SimpleNewFromData(1, &n_bytes, NPY_BYTE, mem));
        PyArray_ENABLEFLAGS(arr, NPY_ARRAY_OWNDATA);

        return reinterpret_cast<PyObject*>(arr);
    }

    PyObject *set_data(PyObject *obj)
    {
        // not an array
        if (!PyArray_Check(obj))
        {
            PyErr_Format(PyExc_RuntimeError,
                "Object is not a numpy array");
            return nullptr;
        }

        PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(obj);

        if (PyArray_TYPE(arr) != NPY_BYTE)
        {
            PyErr_Format(PyExc_RuntimeError,
                "Array is not NPY_BYTE");
            return nullptr;
        }

        NpyIter *it = NpyIter_New(arr, NPY_ITER_READONLY,
                NPY_KEEPORDER, NPY_NO_CASTING, nullptr);
        NpyIter_IterNextFunc *next = NpyIter_GetIterNext(it, nullptr);
        char **ptrptr = reinterpret_cast<char**>(NpyIter_GetDataPtrArray(it));
        do
        {
            self->pack(**ptrptr);
        }
        while (next(it));
        NpyIter_Deallocate(it);

        Py_INCREF(Py_None);
        return Py_None;
    }
}

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
%shared_ptr(teca_variant_array_impl<long>)
%shared_ptr(teca_variant_array_impl<long long>)
%shared_ptr(teca_variant_array_impl<unsigned char>)
%shared_ptr(teca_variant_array_impl<unsigned int>)
%shared_ptr(teca_variant_array_impl<unsigned long>)
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
%ignore teca_variant_array::get;
%ignore teca_variant_array::set;
%ignore teca_variant_array::swap;
%ignore teca_variant_array::equal;
%include "teca_variant_array.h"
// named variant arrays
%template(teca_float_array) teca_variant_array_impl<float>;
%template(teca_double_array) teca_variant_array_impl<double>;
%template(teca_char_array) teca_variant_array_impl<char>;
%template(teca_int_array) teca_variant_array_impl<int>;
%template(teca_long_array) teca_variant_array_impl<long>;
%template(teca_long_long_array) teca_variant_array_impl<long long>;
%template(teca_unsigned_char_array) teca_variant_array_impl<unsigned char>;
%template(teca_unsigned_int_array) teca_variant_array_impl<unsigned int>;
%template(teca_unsigned_long_array) teca_variant_array_impl<unsigned long>;
%template(teca_unsigned_long_long_array) teca_variant_array_impl<unsigned long long>;
// named variant array traits
%template(teca_float_array_code) teca_variant_array_code<float>;
%template(teca_double_array_code) teca_variant_array_code<double>;
%template(teca_char_array_code) teca_variant_array_code<char>;
%template(teca_int_array_code) teca_variant_array_code<int>;
%template(teca_long_array_code) teca_variant_array_code<long>;
%template(teca_long_long_array_code) teca_variant_array_code<long long>;
%template(teca_unsigned_char_array_code) teca_variant_array_code<unsigned char>;
%template(teca_unsigned_int_array_code) teca_variant_array_code<unsigned int>;
%template(teca_unsigned_long_array_code) teca_variant_array_code<unsigned long>;
%template(teca_unsigned_long_long_array_code) teca_variant_array_code<unsigned long long>;
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

        TECA_PY_ERROR_NOW(PyExc_TypeError, "Failed to convert value")
        return nullptr;
    }

    TECA_PY_STR()

    unsigned long __len__()
    { return self->size(); }

    PyObject *__setitem__(unsigned long i, PyObject *value)
    {
        teca_py_gil_state gil;

        if (i >= self->size())
        {
            TECA_PY_ERROR(PyExc_IndexError,
                "index " << i << " is out of bounds in teca_variant_array "
                " with size " << self->size())
            return nullptr;
        }

        Py_INCREF(Py_None);

        if (teca_py_array::set(self, i, value) ||
            teca_py_object::set(self, i, value))
            return Py_None;

        TECA_PY_ERROR(PyExc_TypeError,
            "failed to set value at index " <<  i)

        Py_DECREF(Py_None);
        return nullptr;
    }

    PyObject *__getitem__(unsigned long i)
    {
        teca_py_gil_state gil;

        if (i >= self->size())
        {
            TECA_PY_ERROR(PyExc_IndexError,
                "index " << i << " is out of bounds in teca_variant_array "
                " with size " << self->size())
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

        TECA_PY_ERROR(PyExc_TypeError,
            "failed to get value at index " << i)
        return nullptr;
    }

    PyObject *as_array()
    {
        teca_py_gil_state gil;

        return reinterpret_cast<PyObject*>(
            teca_py_array::new_object(self));
    }

    PyObject *append(PyObject *obj)
    {
        teca_py_gil_state gil;

        Py_INCREF(Py_None);

        if (teca_py_object::append(self, obj)
            || teca_py_array::append(self, obj)
            || teca_py_sequence::append(self, obj))
            return Py_None;

        TECA_PY_ERROR(PyExc_TypeError,
            "Failed to convert value")

        Py_DECREF(Py_None);
        return nullptr;
    }

    PyObject *copy(PyObject *obj)
    {
        teca_py_gil_state gil;

        Py_INCREF(Py_None);

        if (teca_py_object::copy(self, obj)
            || teca_py_array::copy(self, obj)
            || teca_py_sequence::copy(self, obj))
            return Py_None;

        Py_DECREF(Py_None);

        TECA_PY_ERROR(PyExc_TypeError,
            "Failed to convert value")
        return nullptr;
    }

    PyObject *set(unsigned long i, PyObject *value)
    {
        return teca_variant_array___setitem__(self, i, value);
    }

    PyObject *get(unsigned long i)
    {
        return teca_variant_array___getitem__(self, i);
    }
}
TECA_PY_DYNAMIC_VARIANT_ARRAY_CAST(double, double)
TECA_PY_DYNAMIC_VARIANT_ARRAY_CAST(float, float)
TECA_PY_DYNAMIC_VARIANT_ARRAY_CAST(char, char)
TECA_PY_DYNAMIC_VARIANT_ARRAY_CAST(int, int)
TECA_PY_DYNAMIC_VARIANT_ARRAY_CAST(long, long)
TECA_PY_DYNAMIC_VARIANT_ARRAY_CAST(long long, long_long)
TECA_PY_DYNAMIC_VARIANT_ARRAY_CAST(unsigned char, unsigned_char)
TECA_PY_DYNAMIC_VARIANT_ARRAY_CAST(unsigned int, unsigned_int)
TECA_PY_DYNAMIC_VARIANT_ARRAY_CAST(unsigned long, unsigned_long)
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
%ignore teca_metadata::set; /* use __setitem__ instead */
%ignore teca_metadata::get; /* use __getitem__ instead */
%include "teca_metadata.h"
%extend teca_metadata
{
    TECA_PY_STR()

    /* md['name'] = value */
    PyObject *__setitem__(const std::string &name, PyObject *value)
    {
        teca_py_gil_state gil;

        Py_INCREF(Py_None);

        p_teca_variant_array varr;
        if ((varr = teca_py_object::new_variant_array(value))
            || (varr = teca_py_array::new_variant_array(value))
            || (varr = teca_py_sequence::new_variant_array(value))
            || (varr = teca_py_iterator::new_variant_array(value)))
        {
            self->set(name, varr);
            return Py_None;
        }

        TECA_PY_ERROR(PyExc_TypeError,
            "Failed to convert value for key \"" << name << "\"")

        Py_DECREF(Py_None);
        return nullptr;
    }

    /* return md['name'] */
    PyObject *__getitem__(const std::string &name)
    {
        teca_py_gil_state gil;

        p_teca_variant_array varr = self->get(name);
        if (!varr)
        {
            TECA_PY_ERROR(PyExc_KeyError,
                "key \"" << name << "\" not found")
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
                return teca_py_object::py_tt<NT>::new_object(varrt->get(0));
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
                        teca_py_object::py_tt<NT>::new_object(varrt->get(i)));
                }
                return list;
                )
        }

        TECA_PY_ERROR(PyExc_TypeError,
            "Failed to convert value for key \"" << name << "\"")
        return nullptr;
    }

    PyObject *append(const std::string &name, PyObject *obj)
    {
        teca_py_gil_state gil;

        teca_variant_array *varr = self->get(name).get();
        if (!varr)
        {
            TECA_PY_ERROR(PyExc_KeyError,
                "key \"" << name << "\" not found")
            return nullptr;
        }

        Py_INCREF(Py_None);

        if (teca_py_object::append(varr, obj)
            || teca_py_array::append(varr, obj)
            || teca_py_sequence::append(varr, obj)
            || teca_py_iterator::append(varr, obj))
            return Py_None;

        TECA_PY_ERROR(PyExc_TypeError,
            "Failed to convert value")

        Py_DECREF(Py_None);
        return nullptr;
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
%ignore teca_dataset::set_index_request_key(std::string const *);
%include "teca_dataset_fwd.h"
%include "teca_dataset.h"
TECA_PY_CONST_CAST(teca_dataset)
%extend teca_dataset
{
    TECA_PY_STR()
}
%template(std_vector_dataset) std::vector<std::shared_ptr<teca_dataset>>;

/***************************************************************************/
%define TECA_PY_DATASET_METADATA(_type, _name)
    PyObject *set_## _name(PyObject *obj)
    {
        teca_py_gil_state gil;

        Py_INCREF(Py_None);

        // numpy scalars
        TECA_PY_ARRAY_SCALAR_DISPATCH(obj,
            self->set_ ## _name(teca_py_array::numpy_scalar_tt
                <_type>::value(obj));
            return Py_None;
            )

        // regular Python objects
        TECA_PY_OBJECT_DISPATCH_NUM(obj,
            self->set_ ## _name(teca_py_object::cpp_tt
                <teca_py_object::py_tt<_type>::tag>::value(obj));
            return Py_None;
            )

        TECA_PY_ERROR(PyExc_TypeError,
            "Failed to set metadata \"" #_name "\"")

        Py_DECREF(Py_None);
        return nullptr;
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
            TECA_PY_ERROR(PyExc_TypeError,
                "Failed to get metadata \"" # _name "\"")
        }

        return list;
    }

    PyObject *set_## _name(PyObject *array)
    {
        teca_py_gil_state gil;

        Py_INCREF(Py_None);

        p_teca_variant_array varr;
        if ((varr = teca_py_array::new_variant_array(array))
            || (varr = teca_py_sequence::new_variant_array(array))
            || (varr = teca_py_iterator::new_variant_array(array)))
        {
            self->set_## _name(varr);
            return Py_None;
        }

        TECA_PY_ERROR(PyExc_TypeError,
            "Failed to set metadata \"" #_name "\"")

        Py_DECREF(Py_None);
        return nullptr;
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
 index_executive
 ***************************************************************************/
%ignore teca_index_executive::shared_from_this;
%shared_ptr(teca_index_executive)
%ignore teca_index_executive::operator=;
%include "teca_index_executive.h"

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
    PyObject *set_## _name(PyObject *obj)
    {
        teca_py_gil_state gil;

        Py_INCREF(Py_None);

        // numpy scalars
        TECA_PY_ARRAY_SCALAR_DISPATCH(obj,
            self->set_ ## _name(teca_py_array::numpy_scalar_tt
                <_type>::value(obj));
            return Py_None;
            )

        // regular Python objects
        TECA_PY_OBJECT_DISPATCH_NUM(obj,
            self->set_ ## _name(teca_py_object::cpp_tt<
                teca_py_object::py_tt<_type>::tag>::value(obj));
            return Py_None;
            )

        TECA_PY_ERROR_NOW(PyExc_TypeError,
            "Failed to set property \"" #_name "\"")

        Py_DECREF(Py_None);
        return nullptr;
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
            TECA_PY_ERROR(PyExc_TypeError,
                "Failed to get property \"" #_name "\"")
        }

        return list;
    }

    PyObject *set_## _name ##s(PyObject *array)
    {
        teca_py_gil_state gil;

        Py_INCREF(Py_None);

        p_teca_variant_array varr;
        if ((varr = teca_py_array::new_variant_array(array))
            || (varr = teca_py_sequence::new_variant_array(array))
            || (varr = teca_py_iterator::new_variant_array(array)))
        {
            self->set_## _name ##s(varr);
            return Py_None;
        }

        TECA_PY_ERROR_NOW(PyExc_TypeError,
            "Failed to set property \"" #_name "\"")

        Py_DECREF(Py_None);
        return nullptr;
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
 index_reduce
 ***************************************************************************/
%ignore teca_index_reduce::shared_from_this;
%shared_ptr(teca_index_reduce)
%ignore teca_index_reduce::operator=;
%include "teca_index_reduce_fwd.h"
%include "teca_index_reduce.h"

/***************************************************************************
 programmable_algorithm
 ***************************************************************************/
%ignore teca_programmable_algorithm::shared_from_this;
%shared_ptr(teca_programmable_algorithm)
%extend teca_programmable_algorithm
{
    void set_report_callback(PyObject *f)
    {
        teca_py_gil_state gil;

        self->set_report_callback(teca_py_algorithm::report_callback(f));
    }

    void set_request_callback(PyObject *f)
    {
        teca_py_gil_state gil;

        self->set_request_callback(teca_py_algorithm::request_callback(f));
    }

    void set_execute_callback(PyObject *f)
    {
        teca_py_gil_state gil;

        self->set_execute_callback(teca_py_algorithm::execute_callback(f));
    }
}
%ignore teca_programmable_algorithm::operator=;
%ignore teca_programmable_algorithm::set_report_callback;
%ignore teca_programmable_algorithm::get_report_callback;
%ignore teca_programmable_algorithm::set_request_callback;
%ignore teca_programmable_algorithm::get_request_callback;
%ignore teca_programmable_algorithm::set_execute_callback;
%ignore teca_programmable_algorithm::get_execute_callback;
%include "teca_programmable_algorithm_fwd.h"
%include "teca_programmable_algorithm.h"

/***************************************************************************
 threaded programmable_algorithm
 ***************************************************************************/
%ignore teca_threaded_programmable_algorithm::shared_from_this;
%shared_ptr(teca_threaded_programmable_algorithm)
%extend teca_threaded_programmable_algorithm
{
    void set_report_callback(PyObject *f)
    {
        teca_py_gil_state gil;

        self->set_report_callback(teca_py_algorithm::report_callback(f));
    }

    void set_request_callback(PyObject *f)
    {
        teca_py_gil_state gil;

        self->set_request_callback(teca_py_algorithm::request_callback(f));
    }

    void set_execute_callback(PyObject *f)
    {
        teca_py_gil_state gil;

        self->set_execute_callback(teca_py_algorithm::threaded_execute_callback(f));
    }
}
%ignore teca_threaded_programmable_algorithm::operator=;
%ignore teca_threaded_programmable_algorithm::set_report_callback;
%ignore teca_threaded_programmable_algorithm::get_report_callback;
%ignore teca_threaded_programmable_algorithm::set_request_callback;
%ignore teca_threaded_programmable_algorithm::get_request_callback;
%ignore teca_threaded_programmable_algorithm::set_execute_callback;
%ignore teca_threaded_programmable_algorithm::get_execute_callback;
%include "teca_programmable_algorithm_fwd.h"
%include "teca_threaded_programmable_algorithm.h"

/***************************************************************************
 programmable_reduce
 ***************************************************************************/
%ignore teca_programmable_reduce::shared_from_this;
%shared_ptr(teca_programmable_reduce)
%extend teca_programmable_reduce
{
    void set_report_callback(PyObject *f)
    {
        teca_py_gil_state gil;

        self->set_report_callback(teca_py_algorithm::report_callback(f));
    }

    void set_request_callback(PyObject *f)
    {
        teca_py_gil_state gil;

        self->set_request_callback(teca_py_algorithm::request_callback(f));
    }

    void set_reduce_callback(PyObject *f)
    {
        teca_py_gil_state gil;

        self->set_reduce_callback(teca_py_algorithm::reduce_callback(f));
    }

    void set_finalize_callback(PyObject *f)
    {
        teca_py_gil_state gil;

        self->set_finalize_callback(teca_py_algorithm::finalize_callback(f));
    }
}
%ignore teca_programmable_reduce::operator=;
%ignore teca_programmable_reduce::set_report_callback;
%ignore teca_programmable_reduce::get_report_callback;
%ignore teca_programmable_reduce::set_request_callback;
%ignore teca_programmable_reduce::get_request_callback;
%ignore teca_programmable_reduce::set_reduce_callback;
%ignore teca_programmable_reduce::get_reduce_callback;
%ignore teca_programmable_reduce::set_finalize_callback;
%ignore teca_programmable_reduce::get_finalize_callback;
%include "teca_programmable_reduce_fwd.h"
%include "teca_programmable_reduce.h"

/***************************************************************************
 python_algorithm
 ***************************************************************************/
%pythoncode "teca_python_algorithm.py"

/***************************************************************************
 threaded python_algorithm
 ***************************************************************************/
%pythoncode "teca_threaded_python_algorithm.py"

/***************************************************************************
 python_reduce
 ***************************************************************************/
%pythoncode "teca_python_reduce.py"

/***************************************************************************
 dataset_source
 ***************************************************************************/
%ignore teca_dataset_source::shared_from_this;
%shared_ptr(teca_dataset_source)
%ignore teca_dataset_source::operator=;
%include "teca_dataset_source.h"

/***************************************************************************
 dataset_capture
 ***************************************************************************/
%ignore teca_dataset_capture::shared_from_this;
%shared_ptr(teca_dataset_capture)
%ignore teca_dataset_capture::operator=;
%include "teca_dataset_capture.h"

/***************************************************************************
 calcalcs
 ***************************************************************************/
%inline
%{
struct calendar_util
{
// for the given offset in the specified units and caledar returns
// year, month, day, hours, minutes, seconds
static
PyObject *date(double offset, const char *units, const char *calendar)
{
    teca_py_gil_state gil;

    int year = -1;
    int month = -1;
    int day = -1;
    int hour = -1;
    int minute = -1;
    double second = -1.0;

    if (calcalcs::date(offset, &year, &month, &day, &hour,
        &minute, &second, units, calendar))
    {
        TECA_PY_ERROR_NOW(PyExc_RuntimeError, "Failed to convert time")
        Py_INCREF(Py_None);
        return Py_None;
    }

    PyObject *ret = Py_BuildValue("(iiiiid)",
        year, month, day, hour, minute, second);

    return ret;
}

// determine if the specified year is a leap year in the specified calendar
static
PyObject *is_leap_year(const char *calendar, const char *units,
    int year)
{
    teca_py_gil_state gil;

    int leap = 0;
    if (calcalcs::is_leap_year(calendar, units, year, leap))
    {
        TECA_PY_ERROR_NOW(PyExc_RuntimeError,
            "Failed to determine leap year status")
        Py_INCREF(Py_None);
        return Py_None;
    }
    return PyBool_FromLong(leap);
}

// get the days in the month
static
PyObject *days_in_month(const char *calendar, const char *units,
                   int year, int month)
{
    teca_py_gil_state gil;

    int dpm = 0;
    if (calcalcs::days_in_month(calendar, units, year, month, dpm))
    {
        TECA_PY_ERROR_NOW(PyExc_RuntimeError,
            "Failed to determine days in month")
        Py_INCREF(Py_None);
        return Py_None;
    }
    return CIntToPyInteger(dpm);
}
};
%}

/***************************************************************************
 thread util
 ***************************************************************************/
%inline
%{
struct thread_util
{
// determine the number of threads , taking into account, all MPI ranks
// running on the node, such that each thread has a dedicated physical
// core.  builds an affinity map that explicitly specifies the core for
// each thread.
static
PyObject *thread_parameters(MPI_Comm comm,
    int n_requested, int bind, int verbose)
{
    teca_py_gil_state gil;

    std::deque<int> affinity;
    int n_threads = n_requested;
    if (teca_thread_util::thread_parameters(comm, -1,
        n_requested, bind, verbose, n_threads, affinity))
    {
        // caller requested automatic load balancing but this,
        // failed.
        TECA_PY_ERROR(PyExc_RuntimeError,
            "Automatic load balancing failed")
        return nullptr;
    }

    // convert the affinity map to a Python list
    int len = bind ? n_threads : 0;
    PyObject *py_affinity = PyList_New(len);
    for (int i = 0; i < len; ++i)
        PyList_SET_ITEM(py_affinity, i,
            CIntToPyInteger(affinity[i]));

    // return the number of threads and affinity map
    return Py_BuildValue("(iN)", n_threads, py_affinity);
}
};
%}

/***************************************************************************
 system util
 ***************************************************************************/
%inline
%{
struct system_util
{
static
PyObject *get_environment_variable_bool(const char *str, int def)
{
    bool tmp = def;
    int ierr = teca_system_util::get_environment_variable(str, tmp);
    if (ierr < 0)
    {
        TECA_PY_ERROR(PyExc_RuntimeError, "conversion error")
        return nullptr;
    }
    return PyLong_FromLong(long(tmp));
}
};
%}
