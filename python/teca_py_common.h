#ifndef TECA_PY_ERROR_H
#define TECA_PY_ERROR_H
#include <Python.h>

#include "teca_common.h"

#include <string>
#include <sstream>
#include <iostream>

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
