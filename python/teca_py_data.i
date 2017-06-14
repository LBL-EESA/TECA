%{
#include <memory>
#include <sstream>
#include "teca_variant_array.h"
#include "teca_array_collection.h"
#include "teca_cartesian_mesh.h"
#include "teca_mesh.h"
#include "teca_table.h"
#include "teca_py_object.h"
#include "teca_table_collection.h"
#include "teca_database.h"
#include "teca_py_object.h"
#include "teca_py_string.h"
%}

/***************************************************************************
 array_collection
 ***************************************************************************/
%ignore teca_array_collection::shared_from_this;
%shared_ptr(teca_array_collection)
%ignore teca_array_collection::operator=;
%ignore teca_array_collection::operator[];
%include "teca_array_collection_fwd.h"
%include "teca_array_collection.h"
%extend teca_array_collection
{
    TECA_PY_STR()

    /* add or replace an array using syntax: col['name'] = array */
    PyObject *__setitem__(const std::string &name, PyObject *array)
    {
        teca_py_gil_state gil;

        Py_INCREF(Py_None);

        p_teca_variant_array varr;
        if ((varr = teca_py_array::new_variant_array(array))
            || (varr = teca_py_sequence::new_variant_array(array))
            || (varr = teca_py_iterator::new_variant_array(array)))
        {
            self->set(name, varr);
            return Py_None;
        }

        TECA_PY_ERROR(0, PyExc_TypeError,
            "Failed to convert array for key \"" <<  name << "\"")

        Py_DECREF(Py_None);
        return nullptr;
    }

    /* return an array using the syntax: col['name'] */
    PyObject *__getitem__(const std::string &name)
    {
        teca_py_gil_state gil;

        p_teca_variant_array varr = self->get(name);
        if (!varr)
        {
            TECA_PY_ERROR(0, PyExc_KeyError,
                "key \"" << name << "\" not found")
            return nullptr;
        }

        TEMPLATE_DISPATCH(teca_variant_array_impl,
            varr.get(),
            TT *varrt = static_cast<TT*>(varr.get());
            return reinterpret_cast<PyObject*>(
                teca_py_array::new_object(varrt));
            )

        TECA_PY_ERROR(0, PyExc_TypeError,
            "Failed to convert array for key \"" << name << "\"")

        return nullptr;
    }

    /* handle conversion to variant arrays */
    void append(const std::string &name, PyObject *array)
    {
       teca_array_collection___setitem__(self, name, array);
    }
}

/***************************************************************************
 mesh
 ***************************************************************************/
%ignore teca_mesh::shared_from_this;
%shared_ptr(teca_mesh)
%ignore teca_mesh::operator=;
%include "teca_mesh_fwd.h"
%include "teca_mesh.h"
TECA_PY_DYNAMIC_CAST(teca_mesh, teca_dataset)
%extend teca_mesh
{
    TECA_PY_STR()
}

/***************************************************************************
 cartesian_mesh
 ***************************************************************************/
%ignore teca_cartesian_mesh::shared_from_this;
%shared_ptr(teca_cartesian_mesh)
%ignore teca_cartesian_mesh::operator=;
%ignore teca_cartesian_mesh::get_time(double *) const;
%ignore teca_cartesian_mesh::get_time_step(unsigned long *) const;
%ignore teca_cartesian_mesh::set_calendar(std::string const *);
%ignore teca_cartesian_mesh::set_time_units(std::string const *);
%ignore teca_cartesian_mesh::get_periodic_in_x(int *) const;
%ignore teca_cartesian_mesh::get_periodic_in_y(int *) const;
%ignore teca_cartesian_mesh::get_periodic_in_z(int *) const;
%include "teca_cartesian_mesh_fwd.h"
%include "teca_cartesian_mesh.h"
TECA_PY_DYNAMIC_CAST(teca_cartesian_mesh, teca_dataset)
%extend teca_cartesian_mesh
{
    TECA_PY_STR()

    TECA_PY_DATASET_METADATA(double, time)
    TECA_PY_DATASET_METADATA(unsigned long, time_step)
    TECA_PY_DATASET_METADATA(std::string, calendar)
    TECA_PY_DATASET_METADATA(std::string, time_units)
    TECA_PY_DATASET_VECTOR_METADATA(unsigned long, extent)
    TECA_PY_DATASET_VECTOR_METADATA(unsigned long, whole_extent)
}

/***************************************************************************
 table
 ***************************************************************************/
%ignore teca_table::shared_from_this;
%shared_ptr(teca_table)
%ignore teca_table::operator=;
%ignore teca_table::set_calendar(std::string const *);
%ignore teca_table::set_time_units(std::string const *);
%include "teca_table_fwd.h"
%include "teca_table.h"
TECA_PY_DYNAMIC_CAST(teca_table, teca_dataset)
%extend teca_table
{
    TECA_PY_STR()

    TECA_PY_DATASET_METADATA(std::string, calendar)
    TECA_PY_DATASET_METADATA(std::string, time_units)

    /* update the value at r,c. r is a row index and c an
    either be a column index or name */
    PyObject *__setitem__(PyObject *idx, PyObject *obj)
    {
        teca_py_gil_state gil;

        if (!PySequence_Check(idx) || (PySequence_Size(idx) != 2))
        {
            TECA_PY_ERROR(0, PyExc_KeyError,
                "Requires a 2 element sequence specifying "
                "desired row and column indices")
            return nullptr;
        }

        unsigned long r = PyInt_AsLong(PySequence_GetItem(idx, 0));
        unsigned long c = PyInt_AsLong(PySequence_GetItem(idx, 1));

        if (c >= self->get_number_of_columns())
        {
            TECA_PY_ERROR(0, PyExc_IndexError,
                "Column " << c << " is out of bounds")
            return nullptr;
        }

        p_teca_variant_array col = self->get_column(c);

        // handle insertions
        if (r >= col->size())
            col->resize(r+1);

        Py_INCREF(Py_None);

        // numpy scalars
        TECA_PY_ARRAY_SCALAR_DISPATCH(obj,
            ST val = teca_py_array::numpy_scalar_tt<ST>::value(obj);
            TEMPLATE_DISPATCH(teca_variant_array_impl,
                col.get(),
                TT *arr = static_cast<TT*>(col.get());
                arr->set(r, val);
                return Py_None;
                )
            )
        // python objects
        TECA_PY_OBJECT_DISPATCH_NUM(obj,
            teca_py_object::cpp_tt<OT>::type val
                = teca_py_object::cpp_tt<OT>::value(obj);
            TEMPLATE_DISPATCH(
                teca_variant_array_impl,
                col.get(),
                TT *arr = static_cast<TT*>(col.get());
                arr->set(r, val);
                return Py_None;
                )
            )
        TECA_PY_OBJECT_DISPATCH_STR(obj,
            teca_py_object::cpp_tt<OT>::type val
                = teca_py_object::cpp_tt<OT>::value(obj);
            TEMPLATE_DISPATCH_CASE(
                teca_variant_array_impl,
                std::string,
                col.get(),
                TT *arr = static_cast<TT*>(col.get());
                arr->set(r, val);
                return Py_None;
                )
            )

        TECA_PY_ERROR(0, PyExc_TypeError,
            "Failed to convert value at " <<  r << ", " << c)

        Py_DECREF(Py_None);
        return nullptr;
    }

    /* look up the value at r,c. r is a row index and c an
    either be a column index or name */
    PyObject *__getitem__(PyObject *idx)
    {
        teca_py_gil_state gil;

        if (!PySequence_Check(idx) || (PySequence_Size(idx) != 2))
        {
            TECA_PY_ERROR(0, PyExc_KeyError,
                "Requires a 2 element sequence specifying "
                "desired row and column indices")
            return nullptr;
        }

        unsigned long r = PyInt_AsLong(PySequence_GetItem(idx, 0));
        unsigned long c = PyInt_AsLong(PySequence_GetItem(idx, 1));

        if (c >= self->get_number_of_columns())
        {
            TECA_PY_ERROR(0, PyExc_IndexError,
                "Column " << c << " is out of bounds")
            return nullptr;
        }

        p_teca_variant_array col = self->get_column(c);

        if (r >= col->size())
        {
            TECA_PY_ERROR(0, PyExc_IndexError,
                "Row " << r << " is out of bounds")
            return nullptr;
        }


        TEMPLATE_DISPATCH(teca_variant_array_impl,
            col.get(),
            TT *arr = static_cast<TT*>(col.get());
            return reinterpret_cast<PyObject*>(
                teca_py_object::py_tt<NT>::new_object(arr->get(r)));
            )
        TEMPLATE_DISPATCH_CASE(teca_variant_array_impl,
            std::string,
            col.get(),
            TT *arr = static_cast<TT*>(col.get());
            return reinterpret_cast<PyObject*>(
                teca_py_object::py_tt<NT>::new_object(arr->get(r)));
            )

        TECA_PY_ERROR(0, PyExc_TypeError,
            "Failed to convert value at " << r << ", " << c)

        return nullptr;
    }

    /* replace existing column in a single shot */
    PyObject *set_column(PyObject *id, PyObject *array)
    {
        teca_py_gil_state gil;

        p_teca_variant_array col;

        if (PyInt_Check(id))
        {
            long idx = PyInt_AsLong(id);
            if (idx >= self->get_number_of_columns())
            {
                TECA_PY_ERROR(0, PyExc_IndexError,
                    "Column " << idx << " is out of bounds")
                return nullptr;
            }
            col = self->get_column(idx);
        }
        else if (PyStringCheck(id))
        {
            const char *col_name = PyStringToCString(id);
            col = self->get_column(col_name);
            if (!col)
            {
                TECA_PY_ERROR(0, PyExc_IndexError,
                    "No such column \"" << col_name << "\"")
                return nullptr;
            }
        }

        if (!col)
        {
            TECA_PY_ERROR(0, PyExc_TypeError, "Invalid column id type.")
            return nullptr;
        }

        Py_INCREF(Py_None);

        p_teca_variant_array varr;
        if ((varr = teca_py_array::new_variant_array(array))
            || (varr = teca_py_sequence::new_variant_array(array))
            || (varr = teca_py_iterator::new_variant_array(array)))
        {
            col->copy(varr);
            return Py_None;
        }

        TECA_PY_ERROR(0, PyExc_TypeError, "Failed to convert array")

        Py_DECREF(Py_None);
        return nullptr;
    }

    /* declare a column */
    PyObject *declare_column(const char *name, const char *type)
    {
        teca_py_gil_state gil;

        using u_char_t = unsigned char;
        using u_int_t = unsigned int;
        using u_long_t = unsigned long;
        using long_long_t = long long;
        using u_long_long_t = unsigned long long;

        if (!strcmp(type, "c"))
            self->declare_column(name, char());
        else
        if (!strcmp(type, "uc"))
            self->declare_column(name, u_char_t());
        else
        if (!strcmp(type, "i"))
            self->declare_column(name, int());
        else
        if (!strcmp(type, "ui"))
            self->declare_column(name, u_int_t());
        else
        if (!strcmp(type, "l"))
            self->declare_column(name, long());
        else
        if (!strcmp(type, "ul"))
            self->declare_column(name, u_long_t());
        else
        if (!strcmp(type, "ll"))
            self->declare_column(name, long_long_t());
        else
        if (!strcmp(type, "ull"))
            self->declare_column(name, u_long_long_t());
        else
        if (!strcmp(type, "f"))
            self->declare_column(name, float());
        else
        if (!strcmp(type, "d"))
            self->declare_column(name, double());
        else
        if (!strcmp(type, "s"))
            self->declare_column(name, std::string());
        else
        {
            TECA_PY_ERROR(0, PyExc_RuntimeError,
                "Bad type code \"" << type << "\" for column \""
                 << name << "\". Must be one of: c,uc,i,ui,l,ul,"
                 "ll,ull,f,d,s")

            return nullptr;
        }

        Py_INCREF(Py_None);
        return Py_None;
    }

    /* declare a set of columns */
    PyObject *declare_columns(PyObject *names, PyObject *types)
    {
        teca_py_gil_state gil;

        if (!PyList_Check(names))
        {
            TECA_PY_ERROR(0, PyExc_TypeError,
                "names argument must be a list.")
            return nullptr;
        }

        if (!PyList_Check(types))
        {
            TECA_PY_ERROR(0, PyExc_TypeError,
                "types argument must be a list.")
            return nullptr;
        }

        Py_ssize_t n_names = PyList_Size(names);
        Py_ssize_t n_types = PyList_Size(types);

        if (n_names != n_types)
        {
            TECA_PY_ERROR(0, PyExc_RuntimeError,
                "names and types arguments must have same length.")
            return nullptr;
        }

        for (Py_ssize_t i = 0; i < n_names; ++i)
        {
            const char *name = PyStringToCString(PyList_GetItem(names, i));
            if (!name)
            {
                TECA_PY_ERROR(0, PyExc_TypeError,
                    "item at index " << i << " in names is not a string.")
                return nullptr;
            }

            const char *type = PyStringToCString(PyList_GetItem(types, i));
            if (!type)
            {
                TECA_PY_ERROR(0, PyExc_TypeError,
                    "item at index " << i << " in types is not a string.")
                return nullptr;

            }

            PyObject *rv = teca_table_declare_column(self, name, type);
            if (rv)
            {
                Py_DECREF(rv);
            }
            else
            {
                return nullptr;
            }
        }

        Py_INCREF(Py_None);
        return Py_None;
    }

    /* append sequence,array, or object in column order */
    PyObject *append(PyObject *obj)
    {
        teca_py_gil_state gil;

        Py_INCREF(Py_None);

        // numpy scalars
        if (PyArray_CheckScalar(obj))
        {
            TECA_PY_ARRAY_SCALAR_DISPATCH(obj,
                self->append(teca_py_array::numpy_scalar_tt<ST>::value(obj));
                return Py_None;
                )

            TECA_PY_ERROR(0, PyExc_TypeError,
                "failed to append array. Unsupported type.")

            Py_DECREF(Py_None);
            return nullptr;
        }

        // numpy ndarrays
        if (PyArray_Check(obj))
        {
            PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(obj);
            TECA_PY_ARRAY_DISPATCH(arr,
                NpyIter *it = NpyIter_New(arr, NPY_ITER_READONLY,
                        NPY_KEEPORDER, NPY_NO_CASTING, nullptr);
                NpyIter_IterNextFunc *next = NpyIter_GetIterNext(it, nullptr);
                AT **ptrptr = reinterpret_cast<AT**>(NpyIter_GetDataPtrArray(it));
                do
                {
                    self->append(**ptrptr);
                }
                while (next(it));
                NpyIter_Deallocate(it);
                return Py_None;
                )

            TECA_PY_ERROR(0, PyExc_TypeError,
                "failed to append array. Unsupported type.")

            Py_DECREF(Py_None);
            return nullptr;
        }

        // sequences
        if (PySequence_Check(obj) && !PyStringCheck(obj))
        {
            long n_items = PySequence_Size(obj);
            for (long i = 0; i < n_items; ++i)
            {
                PyObject *obj_i = PySequence_GetItem(obj, i);
                TECA_PY_OBJECT_DISPATCH_NUM(obj_i,
                    self->append(teca_py_object::cpp_tt<OT>::value(obj_i));
                    continue;
                    )

                TECA_PY_OBJECT_DISPATCH_STR(obj_i,
                    self->append(teca_py_object::cpp_tt<OT>::value(obj_i));
                    continue;
                    )

                TECA_PY_ERROR(0, PyExc_TypeError,
                    "failed to append sequence element " << i)

                Py_DECREF(Py_None);
                return nullptr;
            }
            return Py_None;
        }

        // objects
        TECA_PY_OBJECT_DISPATCH_NUM(obj,
            self->append(teca_py_object::cpp_tt<OT>::value(obj));
            return Py_None;
            )
        TECA_PY_OBJECT_DISPATCH_STR(obj,
            self->append(teca_py_object::cpp_tt<OT>::value(obj));
            return Py_None;
            )

        TECA_PY_ERROR(0, PyExc_TypeError, "failed to append object")

        Py_DECREF(Py_None);
        return nullptr;
    }

    /* stream insertion operator */
    p_teca_table __lshift__(PyObject *obj)
    {
        teca_table_append(self, obj);
        return self->shared_from_this();
    }
}

/***************************************************************************
 table_collection
 ***************************************************************************/
%ignore teca_table_collection::shared_from_this;
%shared_ptr(teca_table_collection)
%ignore teca_table_collection::operator=;
%ignore teca_table_collection::operator[];
%include "teca_table_collection_fwd.h"
%include "teca_table_collection.h"
%extend teca_table_collection
{
    TECA_PY_STR()

    /* add or replace a table using syntax: col['name'] = table */
    void __setitem__(const std::string &name, p_teca_table table)
    {
        self->set(name, table);
    }

    /* return an array using the syntax: col['name'] */
    p_teca_table __getitem__(const std::string &name)
    {
        teca_py_gil_state gil;

        p_teca_table table = self->get(name);
        if (!table)
        {
            TECA_PY_ERROR(1, PyExc_KeyError,
                "key \"" << name << "\" not found")
            return nullptr;
        }

        return table;
    }
}

/***************************************************************************
 database
 ***************************************************************************/
%ignore teca_database::shared_from_this;
%shared_ptr(teca_database)
%ignore teca_database::operator=;
%ignore teca_database::get_table_name(unsigned int);
%include "teca_database_fwd.h"
%include "teca_database.h"
TECA_PY_DYNAMIC_CAST(teca_database, teca_dataset)
%extend teca_database
{
    TECA_PY_STR()
}
