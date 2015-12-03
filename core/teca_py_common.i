/***************************************************************************/
%define PY_TECA_STR()
    PyObject *__str__()
    {
        std::ostringstream oss;
        self->to_stream(oss);
        return PyString_FromString(oss.str().c_str());
    }
%enddef
