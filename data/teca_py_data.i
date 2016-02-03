%define MDOC
"TECA data module

This module provides high-level data structures that
are produced and consumed by teca_algorithms such as
Cartesian meshes, AMR datasets, and tables.
"
%enddef

%module (docstring=MDOC) teca_py_data
%feature("autodoc", "3");

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
#include "teca_workbook.h"
%}

%include "teca_config.h"
%include "teca_py_common.i"
%include "teca_py_shared_ptr.i"
%include "teca_py_core.i"
%include <std_string.i>

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
    void __setitem__(const std::string &name, PyObject *array)
    {
        p_teca_variant_array varr;
        if ((varr = teca_py_array::new_variant_array(array))
            || (varr = teca_py_sequence::new_variant_array(array))
            || (varr = teca_py_iterator::new_variant_array(array)))
        {
            self->set(name, varr);
            return;
        }
        PyErr_Format(PyExc_TypeError,
            "Failed to convert array for key \"%s\"", name.c_str());
    }

    /* return an array using the syntax: col['name'] */
    PyObject *__getitem__(const std::string &name)
    {
        p_teca_variant_array varr = self->get(name);
        if (!varr)
        {
            PyErr_Format(PyExc_KeyError,
                "key \"%s\" not found", name.c_str());
            return nullptr;
        }

        TEMPLATE_DISPATCH(teca_variant_array_impl,
            varr.get(),
            TT *varrt = static_cast<TT*>(varr.get());
            return reinterpret_cast<PyObject*>(
                teca_py_array::new_object(varrt));
            )

        return PyErr_Format(PyExc_TypeError,
            "Failed to convert array for key \"%s\"", name.c_str());
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
%ignore teca_cartesian_mesh::set_calendar(std::string const *);
%ignore teca_cartesian_mesh::set_time_units(std::string const *);
%ignore teca_cartesian_mesh::get_time_step(unsigned long *) const;
%include "teca_cartesian_mesh_fwd.h"
%include "teca_cartesian_mesh.h"
TECA_PY_DYNAMIC_CAST(teca_cartesian_mesh, teca_dataset)
%extend teca_cartesian_mesh
{
    TECA_PY_STR()
}

/***************************************************************************
 table
 ***************************************************************************/
%ignore teca_table::shared_from_this;
%shared_ptr(teca_table)
%ignore teca_table::operator=;
%include "teca_table_fwd.h"
%include "teca_table.h"
TECA_PY_DYNAMIC_CAST(teca_table, teca_dataset)
%extend teca_table
{
    TECA_PY_STR()

    /* update the value at r,c */
    void __setitem__(PyObject *idx, PyObject *obj)
    {
        if (!PySequence_Check(idx) || (PySequence_Size(idx) != 2))
        {
            PyErr_Format(PyExc_KeyError,
                "Requires a 2 element sequence specifying "
                "desired row and column indices");
            return;
        }

        unsigned long r = PyInt_AsLong(PySequence_GetItem(idx, 0));
        unsigned long c = PyInt_AsLong(PySequence_GetItem(idx, 1));

        p_teca_variant_array col = self->get_column(c);
        if (!col)
        {
            PyErr_Format(PyExc_IndexError,
                "Column %lu is out of bounds", c);
            return;
        }

        if (r >= col->size())
            col->resize(r+1);

        TECA_PY_OBJECT_DISPATCH_NUM(obj,
            teca_py_object::cpp_tt<OT>::type val
                = teca_py_object::cpp_tt<OT>::value(obj);
            TEMPLATE_DISPATCH(
                teca_variant_array_impl,
                col.get(),
                TT *arr = static_cast<TT*>(col.get());
                arr->set(r, val);
                return;
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
                return;
                )
            )

        PyErr_Format(PyExc_TypeError,
            "Failed to convert value at %ld,%ld", r, c);
    }

    /* look up the value at r,c */
    PyObject *__getitem__(PyObject *idx)
    {
        if (!PySequence_Check(idx) || (PySequence_Size(idx) != 2))
        {
            PyErr_Format(PyExc_KeyError,
                "Requires a 2 element sequence specifying "
                "desired row and column indices");
            return nullptr;
        }

        unsigned long r = PyInt_AsLong(PySequence_GetItem(idx, 0));
        unsigned long c = PyInt_AsLong(PySequence_GetItem(idx, 1));

        p_teca_variant_array col = self->get_column(c);
        if (!col)
        {
            PyErr_Format(PyExc_IndexError,
                "Column %lu is out of bounds", c);
            return nullptr;
        }

        if (r >= col->size())
        {
            PyErr_Format(PyExc_IndexError,
                "Row %lu is out of bounds", r);
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

        return PyErr_Format(PyExc_TypeError,
            "Failed to convert value at %lu, %lu", r, c);
    }

    /* replace existing column in a single shot */
    void set_column(PyObject *id, PyObject *array)
    {
        p_teca_variant_array col;

        if (PyInt_Check(id))
            col = self->get_column(PyInt_AsLong(id));
        else
        if (PyString_Check(id))
            col = self->get_column(PyString_AsString(id));

        if (!col)
        {
            PyErr_Format(PyExc_KeyError, "Invalid column id.");
            return;
        }

        p_teca_variant_array varr;
        if ((varr = teca_py_array::new_variant_array(array))
            || (varr = teca_py_sequence::new_variant_array(array))
            || (varr = teca_py_iterator::new_variant_array(array)))
        {
            col->copy(varr);
            return;
        }

        PyErr_Format(PyExc_TypeError, "Failed to convert array");
    }

    /* declare a column */
    void declare_column(const char *name, const char *type)
    {
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
            PyErr_Format(PyExc_RuntimeError,
                "Bad type code \"%s\" for column \"%s\". Must be one of: "
                "c,uc,i,ui,l,ul,ll,ull,f,d,s", type, name);
    }

    /* declare a set of columns */
    void declare_columns(PyObject *names, PyObject *types)
    {
        if (!PyList_Check(names))
        {
            PyErr_Format(PyExc_TypeError,
                "names argument must be a list.");
            return;
        }

        if (!PyList_Check(types))
        {
            PyErr_Format(PyExc_TypeError,
                "types argument must be a list.");
            return;
        }

        Py_ssize_t n_names = PyList_Size(names);
        Py_ssize_t n_types = PyList_Size(types);

        if (n_names != n_types)
        {
            PyErr_Format(PyExc_RuntimeError,
                "names and types arguments must have same length.");
            return;
        }

        for (Py_ssize_t i = 0; i < n_names; ++i)
        {
            const char *name = PyString_AsString(PyList_GetItem(names, i));
            if (!name)
            {
                PyErr_Format(PyExc_TypeError,
                    "item at index %ld in names is not a string.", i);
                return;
            }

            const char *type = PyString_AsString(PyList_GetItem(types, i));
            if (!type)
            {
                PyErr_Format(PyExc_TypeError,
                    "item at index %ld in types is not a string.", i);
                return;

            }

            teca_table_declare_column(self, name, type);
        }
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
        p_teca_table table = self->get(name);
        if (!table)
        {
            PyErr_Format(PyExc_KeyError,
                "key \"%s\" not found", name.c_str());
            return nullptr;
        }

        return table;
    }
}

/***************************************************************************
 workbook
 ***************************************************************************/
%ignore teca_workbook::shared_from_this;
%shared_ptr(teca_workbook)
%ignore teca_workbook::operator=;
%ignore teca_workbook::get_table_name(unsigned int);
%include "teca_workbook_fwd.h"
%include "teca_workbook.h"
TECA_PY_DYNAMIC_CAST(teca_workbook, teca_dataset)
%extend teca_workbook
{
    TECA_PY_STR()
}
