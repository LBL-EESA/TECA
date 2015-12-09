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
#include "teca_array_collection.h"
#include "teca_cartesian_mesh.h"
#include "teca_mesh.h"
#include "teca_table.h"
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
%include "teca_array_collection_fwd.h"
%include "teca_array_collection.h"
%extend teca_array_collection
{
    /* lists the names of the arrays in the collection */
    PyObject *__str__()
    {
        std::ostringstream oss;
        oss << "{";
        size_t n_arrays = self->size();
        if (n_arrays)
        {
            oss << self->get_name(0);
            for (size_t i = 1; i < n_arrays; ++i)
                oss << ", " <<  self->get_name(i);
        }
        oss << "}";
        return PyString_FromString(oss.str().c_str());
    }

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

/***************************************************************************
 cartesian_mesh
 ***************************************************************************/
%ignore teca_cartesian_mesh::shared_from_this;
%shared_ptr(teca_cartesian_mesh)
%ignore teca_cartesian_mesh::operator=;
%include "teca_cartesian_mesh_fwd.h"
%include "teca_cartesian_mesh.h"
TECA_PY_DYNAMIC_CAST(teca_cartesian_mesh, teca_dataset)

/***************************************************************************
 table
 ***************************************************************************/
%ignore teca_table::shared_from_this;
%shared_ptr(teca_table)
%ignore teca_table::operator=;
%include "teca_table_fwd.h"
%include "teca_table.h"
TECA_PY_DYNAMIC_CAST(teca_table, teca_dataset)
