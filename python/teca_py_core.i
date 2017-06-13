%{
#include <vector>

#include "teca_algorithm_executive.h"
#include "teca_index_executive.h"
#include "teca_metadata.h"
#include "teca_algorithm.h"
#include "teca_threaded_algorithm.h"
#include "teca_index_reduce.h"
#include "teca_variant_array.h"
#include "teca_binary_stream.h"
#include "teca_parallel_id.h"

#include "teca_py_object.h"
#include "teca_py_sequence.h"
#include "teca_py_array.h"
#include "teca_py_iterator.h"
#include "teca_py_gil_state.h"
%}

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
        if (teca_py_array::set(self, i, value) ||
            teca_py_object::set(self, i, value))
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
            self->set(name, varr);
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
 index_reduce
 ***************************************************************************/
%ignore teca_index_reduce::shared_from_this;
%shared_ptr(teca_index_reduce)
%ignore teca_index_reduce::operator=;
%include "teca_index_reduce_fwd.h"
%include "teca_index_reduce.h"
