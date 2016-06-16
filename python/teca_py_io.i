%{
#include "teca_algorithm.h"
#include "teca_cf_reader.h"
#include "teca_table_reader.h"
#include "teca_table_writer.h"
#include "teca_vtk_cartesian_mesh_writer.h"
%}

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
 table_reader
 ***************************************************************************/
%ignore teca_table_reader::shared_from_this;
%shared_ptr(teca_table_reader)
%ignore teca_table_reader::operator=;
%include "teca_table_reader.h"

/***************************************************************************
 table_writer
 ***************************************************************************/
%ignore teca_table_writer::shared_from_this;
%shared_ptr(teca_table_writer)
%ignore teca_table_writer::operator=;
%include "teca_table_writer.h"

/***************************************************************************
 vtk_cartesian_mesh_writer
 ***************************************************************************/
%ignore teca_vtk_cartesian_mesh_writer::shared_from_this;
%shared_ptr(teca_vtk_cartesian_mesh_writer)
%ignore teca_vtk_cartesian_mesh_writer::operator=;
%include "teca_vtk_cartesian_mesh_writer.h"
