%define TECA_PY_ALG_DOC
"TECA alg module

The io module contains data processing, analysis, remeshing,
and detectors.
"
%enddef
%module (docstring=TECA_PY_ALG_DOC) teca_py_alg
%feature("autodoc", "3");

%{
#include <Python.h>
#include "teca_config.h"
#include "teca_algorithm.h"
#include "teca_ar_detect.h"
#include "teca_cartesian_mesh_subset.h"
#include "teca_cartesian_mesh_regrid.h"
#include "teca_l2_norm.h"
#include "teca_programmable_algorithm.h"
#include "teca_table_reduce.h"
#include "teca_temporal_average.h"
#include "teca_vorticity.h"
%}

%include "teca_config.h"
%include "teca_py_common.i"
%include "teca_py_shared_ptr.i"
%include "teca_py_core.i"
%include <std_string.i>


/***************************************************************************
 ar_detect
 ***************************************************************************/
%ignore teca_ar_detect::shared_from_this;
%shared_ptr(teca_ar_detect)
%ignore teca_ar_detect::operator=;
%include "teca_ar_detect.h"

/***************************************************************************
 cartesian_mesh_subset
 ***************************************************************************/
%ignore teca_cartesian_mesh_subset::shared_from_this;
%shared_ptr(teca_cartesian_mesh_subset)
%ignore teca_cartesian_mesh_subset::operator=;
%include "teca_cartesian_mesh_subset.h"

/***************************************************************************
 cartesian_mesh_regrid
 ***************************************************************************/
%ignore teca_cartesian_mesh_regrid::shared_from_this;
%shared_ptr(teca_cartesian_mesh_regrid)
%ignore teca_cartesian_mesh_regrid::operator=;
%include "teca_cartesian_mesh_regrid.h"

/***************************************************************************
 l2_norm
 ***************************************************************************/
%ignore teca_l2_norm::shared_from_this;
%shared_ptr(teca_l2_norm)
%ignore teca_l2_norm::operator=;
%include "teca_l2_norm.h"

/***************************************************************************
 table_reduce
 ***************************************************************************/
%ignore teca_table_reduce::shared_from_this;
%shared_ptr(teca_table_reduce)
%ignore teca_table_reduce::operator=;
%include "teca_table_reduce.h"

/***************************************************************************
 temporal_average
 ***************************************************************************/
%ignore teca_temporal_average::shared_from_this;
%shared_ptr(teca_temporal_average)
%ignore teca_temporal_average::operator=;
%include "teca_temporal_average.h"

/***************************************************************************
 vorticity
 ***************************************************************************/
%ignore teca_vorticity::shared_from_this;
%shared_ptr(teca_vorticity)
%ignore teca_vorticity::operator=;
%include "teca_vorticity.h"

/***************************************************************************
 programmable_algorithm
 ***************************************************************************/
%ignore teca_programmable_algorithm::shared_from_this;
%shared_ptr(teca_programmable_algorithm)
%ignore teca_programmable_algorithm::operator=;
%ignore teca_programmable_algorithm::set_report_function;
%ignore teca_programmable_algorithm::get_report_function;
%ignore teca_programmable_algorithm::set_request_function;
%ignore teca_programmable_algorithm::get_request_function;
%ignore teca_programmable_algorithm::set_execute_function;
%ignore teca_programmable_algorithm::get_execute_function;
%include "teca_programmable_algorithm_fwd.h"
%include "teca_programmable_algorithm.h"
