%define TECA_PY_DOC
"TECA Python module

The core module contains the pipeline and executive
as well as metadata object, variant array and abstract
datasets.

The data module provides high-level data structures that
are produced and consumed by teca_algorithms such as
Cartesian meshes, AMR datasets, and tables.

The alg module contains data processing, analysis, remeshing,
and detectors.

The io module contains readers and writers.
"
%enddef
%module (docstring=TECA_PY_DOC) teca_py
%feature("autodoc", "3");

%{
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL  PyArray_API_teca_py
#include <numpy/arrayobject.h>
#include <Python.h>

/* disable some warnings that are present in SWIG generated code. */
#if !defined(TECA_DEBUG)
#if __GNUC__ > 8
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#if defined(__CUDACC__)
#pragma nv_diag_suppress = set_but_not_used
#endif
%}

/* SWIG doens't understand compiler attriibbutes */
#define __attribute__(x)

%include <std_pair.i>
%include <std_array.i>
%include <std_string.i>
%include <std_iostream.i>
%include "teca_py_common.i"
%include "teca_py_config.i"
%include "teca_py_array.i"
%include "teca_py_vector.i"
%include "teca_py_shared_ptr.i"
%include "teca_py_mpi.i"
%include "teca_py_core.i"
%include "teca_py_data.i"
%include "teca_py_alg.i"
%include "teca_py_io.i"
%include "teca_py_system.i"

%init %{
PyEval_InitThreads();
import_array();
%}
