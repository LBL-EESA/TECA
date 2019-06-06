%{
#include "teca_algorithm.h"
#include "teca_cf_reader.h"
#include "teca_cf_writer.h"
#include "teca_table_reader.h"
#include "teca_table_writer.h"
#include "teca_cartesian_mesh_reader.h"
#include "teca_cartesian_mesh_writer.h"
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
 cf_writer
 ***************************************************************************/
#ifdef TECA_HAS_NETCDF
%ignore teca_cf_writer::shared_from_this;
%shared_ptr(teca_cf_writer)
%ignore teca_cf_writer::operator=;
%include "teca_cf_writer.h"
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
 cartesian_mesh_reader
 ***************************************************************************/
%ignore teca_cartesian_mesh_reader::shared_from_this;
%shared_ptr(teca_cartesian_mesh_reader)
%ignore teca_cartesian_mesh_reader::operator=;
%include "teca_cartesian_mesh_reader.h"

/***************************************************************************
 cartesian_mesh_writer
 ***************************************************************************/
%ignore teca_cartesian_mesh_writer::shared_from_this;
%shared_ptr(teca_cartesian_mesh_writer)
%ignore teca_cartesian_mesh_writer::operator=;
%include "teca_cartesian_mesh_writer.h"
