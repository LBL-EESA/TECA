%define MDOC
"TECA data module

This module provides high-level data structures that
are produced and consumed by teca_algorithms such as
Cartesian meshes, AMR datasets, and tables.
"
%enddef

%module (docstring=MDOC) teca_py_data

%{
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
