#ifndef teca_py_gil_state_h
#define teca_py_gil_state_h

#include <Python.h>

// RAII helper for managing the Python GIL
class teca_py_gil_state
{
public:
    teca_py_gil_state()
    { m_state = PyGILState_Ensure(); }

    ~teca_py_gil_state()
    { PyGILState_Release(m_state); }

    teca_py_gil_state(const teca_py_gil_state&) = delete;
    void operator=(const teca_py_gil_state&) = delete;

private:
    PyGILState_STATE m_state;
};

#endif
