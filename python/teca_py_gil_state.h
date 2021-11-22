#ifndef teca_py_gil_state_h
#define teca_py_gil_state_h

#include <Python.h>

#include "teca_config.h"

/// A RAII helper for managing the Python GIL.
/** The GIL is aquired and held while the object exists. The GIL must be held
 * by C++ code invoking any Python C-API calls.
 */
class TECA_EXPORT teca_py_gil_state
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
