%{
#include <sstream>
#include "teca_py_string.h"
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
/* for each dataset type define a functions that wrap std::dynamic_pointer_cast */
%define TECA_PY_DYNAMIC_CAST(to_t, from_t)
%inline %{
p_##to_t as_##to_t(p_##from_t in_inst)
{
    p_##to_t o_inst = std::dynamic_pointer_cast<to_t>(in_inst);
    /*if (!o_inst)
    {
        teca_py_gil_state gil;
        PyErr_Format(PyExc_TypeError,
            "Failed to convert from \"%s\" to \"%s\"", #from_t, #to_t);
    }*/
    return o_inst;
}

const_p_##to_t as_const_##to_t(const_p_##from_t inst)
{
    const_p_##to_t o_inst = std::dynamic_pointer_cast<const to_t>(inst);
    /*if (!o_inst)
    {
        teca_py_gil_state gil;
        PyErr_Format(PyExc_TypeError,
            "Failed to convert from \"%s\" to \"%s\"", #from_t, #to_t);
    }*/
    return o_inst;
}
%}
%enddef

/***************************************************************************/
/* for each type define functions that wrap std::dynamic_pointer_cast

    p_teca_X_array as_teca_X_array(p_teca_variant_array)
    const_p_teca_X_array as_const_teca_X_array(const_p_teca_variant_array) */
%define TECA_PY_DYNAMIC_VARIANT_ARRAY_CAST(to_t, to_t_name)
%inline %{
std::shared_ptr<teca_variant_array_impl<to_t>>
as_teca_##to_t_name##_array(p_teca_variant_array in_inst)
{
    std::shared_ptr<teca_variant_array_impl<to_t>> o_inst
        = std::dynamic_pointer_cast<teca_variant_array_impl<to_t>>(in_inst);

    if (!o_inst)
    {
        teca_py_gil_state gil;
        PyErr_Format(PyExc_TypeError,
            "Failed to convert teca_variant_array  to teca_" #to_t_name "_array");
    }

    return o_inst;
}

std::shared_ptr<const teca_variant_array_impl<to_t>>
as_const_teca_##to_t_name##_array(const_p_teca_variant_array in_inst)
{
    std::shared_ptr<const teca_variant_array_impl<to_t>> o_inst
        = std::dynamic_pointer_cast<const teca_variant_array_impl<to_t>>(in_inst);

    if (!o_inst)
    {
        teca_py_gil_state gil;
        PyErr_Format(PyExc_TypeError,
            "Failed to convert const_teca_variant_array  to const_teca_" #to_t_name "_array");
    }

    return o_inst;
}
%}
%enddef
