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
/* core */
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL  PyArray_API_teca_py
#include <numpy/arrayobject.h>
#include <Python.h>
%}

%include <std_pair.i>
%include <std_string.i>
%include "teca_py_vector.i"
%include "teca_py_common.i"
%include "teca_py_shared_ptr.i"
%include "teca_py_core.i"
%include "teca_py_data.i"
%include "teca_py_alg.i"
%include "teca_py_io.i"
%include "teca_py_system.i"

%init %{
PyEval_InitThreads();
import_array();
%}
