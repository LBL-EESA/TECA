%define TECA_PY_IO_DOC
"TECA io module

The io module contains readers and writers.
"
%enddef
%module (docstring=TECA_PY_IO_DOC) teca_py_io
%feature("autodoc", "3");

%{
#include <Python.h>
#include "teca_config.h"
#include "teca_algorithm.h"
#include "teca_cf_reader.h"
#include "teca_csv_writer.h"
#include "teca_vtk_cartesian_mesh_writer.h"
%}

%include "teca_config.h"
%include "teca_py_common.i"
%include "teca_py_shared_ptr.i"
%include "teca_py_core.i"
%include <std_string.i>


/***************************************************************************
 cf_reader
 ***************************************************************************/
#ifdef TECA_HAS_NETCDF
%ignore teca_cf_reader::shared_from_this;
%shared_ptr(teca_cf_reader)
%ignore teca_cf_reader::operator=;
%include "teca_cf_reader.h"
#endif

/***************************************************************************
 table_writer
 ***************************************************************************/
%ignore teca_csv_writer::shared_from_this;
%shared_ptr(teca_csv_writer)
%ignore teca_csv_writer::operator=;
%include "teca_csv_writer.h"

/***************************************************************************
 vtk_cartesian_mesh_writer
 ***************************************************************************/
%ignore teca_vtk_cartesian_mesh_writer::shared_from_this;
%shared_ptr(teca_vtk_cartesian_mesh_writer)
%ignore teca_vtk_cartesian_mesh_writer::operator=;
%include "teca_vtk_cartesian_mesh_writer.h"
