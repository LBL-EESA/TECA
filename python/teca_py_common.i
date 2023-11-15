%{
#include "teca_common.h"
#include "teca_py_string.h"

#include <string>
#include <sstream>

#ifndef TECA_PY_ERROR_H
#define TECA_PY_ERROR_H
// print an error message to std::cerr and set the interpreter
// up for an exception. The exception will occur now if the retrun
// value is null or later at the next time the interpreter checks
// the error flag. See also TECA_PY_ERROR_NOW
#define TECA_PY_ERROR(_type, _xmsg)                     \
{                                                       \
    std::ostringstream oss;                             \
    oss << std::endl;                                   \
    TECA_MESSAGE_RAW(oss, "ERROR:", _xmsg)              \
    PyErr_Format(_type, "%s", oss.str().c_str());       \
                                                        \
    PyObject *sys_stderr = PySys_GetObject("stderr");   \
    PyObject_CallMethod(sys_stderr, "flush", nullptr);  \
                                                        \
    PyObject *sys_stdout = PySys_GetObject("stdout");   \
    PyObject_CallMethod(sys_stdout, "flush", nullptr);  \
}

// the same as TECA_PY_ERROR except forces the interpreter to
// check the error flag now. Thus this will ensure the exception
// occcurs now and not during subsequent code evaluation which
// is very confusing.
#define TECA_PY_ERROR_NOW(_type, _xmsg)             \
{                                                   \
    TECA_PY_ERROR(_type, _xmsg)                     \
    PyRun_SimpleString("pass");                     \
}
#endif
%}

/***************************************************************************/
%define TECA_PY_STR()
    PyObject *__str__()
    {
        teca_py_gil_state gil;

        std::ostringstream oss;
        self->to_stream(oss);
        return CStringToPyString(oss.str().c_str());
    }
%enddef

/***************************************************************************/
/* for each dataset type define functions that wrap
   std::dynamic_pointer_cast. if the cast fails the returned object
   evaluates to None in Python. No error should be thrown here. */
%define TECA_PY_DYNAMIC_CAST(to_t, from_t)
%inline %{
p_##to_t as_##to_t(p_##from_t in_inst)
{
    p_##to_t o_inst = std::dynamic_pointer_cast<to_t>(in_inst);
    return o_inst;
}

const_p_##to_t as_const_##to_t(const_p_##from_t inst)
{
    const_p_##to_t o_inst = std::dynamic_pointer_cast<const to_t>(inst);
    return o_inst;
}
%}
%enddef

/***************************************************************************/
/* for each dataset type define functions that wrap
   std::const_pointer_cast. if the cast fails the returned object
   evaluates to None in Python. No error should be thrown here. */
%define TECA_PY_CONST_CAST(to_t)
%inline %{
p_##to_t as_non_const_##to_t(const_p_##to_t in_inst)
{
    p_##to_t o_inst = std::const_pointer_cast<to_t>(in_inst);
    return o_inst;
}
%}
%enddef

/***************************************************************************/
/* for each type define functions that wrap std::dynamic_pointer_cast

   p_teca_X_array as_teca_X_array(p_teca_variant_array)
   const_p_teca_X_array as_const_teca_X_array(const_p_teca_variant_array)

   if the cast fails the returned object evaluates to None in Python.
   No error should be thrown here. */
%define TECA_PY_DYNAMIC_VARIANT_ARRAY_CAST(to_t, to_t_name)
%inline %{
std::shared_ptr<teca_variant_array_impl<to_t>>
as_teca_##to_t_name##_array(p_teca_variant_array in_inst)
{
    std::shared_ptr<teca_variant_array_impl<to_t>> o_inst
        = std::dynamic_pointer_cast<teca_variant_array_impl<to_t>>(in_inst);
    return o_inst;
}

std::shared_ptr<const teca_variant_array_impl<to_t>>
as_const_teca_##to_t_name##_array(const_p_teca_variant_array in_inst)
{
    std::shared_ptr<const teca_variant_array_impl<to_t>> o_inst
        = std::dynamic_pointer_cast<const teca_variant_array_impl<to_t>>(in_inst);
    return o_inst;
}
%}
%enddef

/* typemaps defining conversions from Python sequence objects to fixed length C - Arrays */
%typemap(in) double [ANY](double temp[$1_dim0]) {
  if (teca_py_object::sequence_to_array<float,double>($input, temp, $1_dim0)) {
    SWIG_fail;
  }
  $1 = &temp[0];
}

%typemap(in) float [ANY](float temp[$1_dim0]) {
  if (teca_py_object::sequence_to_array<float,float>($input, temp, $1_dim0)) {
    SWIG_fail;
  }
  $1 = &temp[0];
}

%typemap(in) long [ANY](long temp[$1_dim0]) {
  if (teca_py_object::sequence_to_array<long,long>($input, temp, $1_dim0)) {
    SWIG_fail;
  }
  $1 = &temp[0];
}

%typemap(in) unsigned long [ANY](unsigned long temp[$1_dim0]) {
  if (teca_py_object::sequence_to_array<long,unsigned long>($input, temp, $1_dim0)) {
    SWIG_fail;
  }
  $1 = &temp[0];
}
