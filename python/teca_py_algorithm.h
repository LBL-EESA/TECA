#ifndef teca_py_algorithm_h
#define teca_py_algorithm_h

/// @file

#include "teca_config.h"
#include "teca_common.h"
#include "teca_metadata.h"
#include "teca_dataset.h"
#include "teca_py_object.h"
#include "teca_py_gil_state.h"
#include "teca_py_string.h"

#include <Python.h>
#include <vector>

/** Reports an error from a user provided callback.  We are going to be overly
 * verbose in an effort to help the user debug their code. package this up for
 * use in all the callbacks.
 */
#define TECA_PY_CALLBACK_ERROR(_phase, _cb_obj)                 \
    {                                                           \
    PyObject *cb_str = PyObject_Str(_cb_obj);                   \
    const char *cb_c_str = PyStringToCString(cb_str);           \
                                                                \
    TECA_MESSAGE_RAW(std::cerr, "ERROR:", "An exception "       \
    "ocurred  when invoking the user supplied Python callback " \
    "\"" << cb_c_str << "\" for the " #_phase " execution "     \
    "phase. The exception that occurred is:")                   \
                                                                \
    PyErr_Print();                                              \
                                                                \
    PyObject *sys_stderr = PySys_GetObject("stderr");           \
    PyObject_CallMethod(sys_stderr, "flush", nullptr);          \
                                                                \
    PyObject *sys_stdout = PySys_GetObject("stdout");           \
    PyObject_CallMethod(sys_stdout, "flush", nullptr);          \
                                                                \
    Py_XDECREF(cb_str);                                         \
    }

/// Codes for briding teca_algorithm to Python
namespace teca_py_algorithm
{
/// wrapper for report_callback phase callback
/**
 * Manages a python callback for use during the report_callback phase of
 * pipeline execution. In addition to holding the callback it handles translation
 * of the input and output arguments.
 *
 * the python function must accept the following arguments:
 *
 * > port - an integer set to the active port number
 * > input_md - a list of metadata objects one per input connection
 *
 * it must return: a teca_metadata object.
*/
class TECA_EXPORT report_callback
{
public:
    report_callback(PyObject *f)
        : m_callback(f) {}

    void set_callback(PyObject *f)
    { m_callback.set_object(f); }

    explicit operator bool() const
    { return static_cast<bool>(m_callback); }

    teca_metadata operator()(unsigned int port,
        const std::vector<teca_metadata> &input_md)
    {
        teca_py_gil_state gil;

        PyObject *f = m_callback.get_object();
        if (!f)
        {
            TECA_PY_ERROR_NOW(PyExc_TypeError,
                "report_callback callback not set")
            return teca_metadata();
        }

        // wrap input metadata in a list
        size_t n_elem = input_md.size();
        PyObject *py_md = PyList_New(n_elem);
        for (size_t i = 0; i < n_elem; ++i)
        {
            PyList_SET_ITEM(py_md, i, SWIG_NewPointerObj(
                SWIG_as_voidptr(new teca_metadata(input_md[i])),
                SWIGTYPE_p_teca_metadata, SWIG_POINTER_OWN));
        }

        // call the callback
        PyObject *args = Py_BuildValue("IN", port, py_md);

        PyObject *ret = nullptr;
        if (!(ret = PyObject_CallObject(f, args)) || PyErr_Occurred())
        {
            TECA_PY_CALLBACK_ERROR(report, f)
            return teca_metadata();
        }

        Py_DECREF(args);

        // convert the return metadata
        teca_metadata *md = nullptr;
        if (!SWIG_IsOK(SWIG_ConvertPtr(ret, reinterpret_cast<void**>(&md),
            SWIGTYPE_p_teca_metadata, 0)) || !md)
        {
            TECA_PY_ERROR_NOW(PyExc_TypeError,
                "invalid return type from report callback")
            return teca_metadata();
        }

        return *md;
    }

private:
    teca_py_object::teca_py_callable m_callback;
};

/// A wrapper for the request phase callback.
/** Manages a python callback for use during the request phase of pipeline
 * execution. In addition to holding the callback it handles translation of the
 *input and output arguments.
 */
class TECA_EXPORT request_callback
{
public:
    request_callback(PyObject *f)
        : m_callback(f) {}

    PyObject *set_callback(PyObject *f)
    { return m_callback.set_object(f); }

    explicit operator bool() const
    { return static_cast<bool>(m_callback); }

    std::vector<teca_metadata> operator()(unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request)
    {
        teca_py_gil_state gil;

        PyObject *f = m_callback.get_object();
        if (!f)
        {
            TECA_PY_ERROR_NOW(PyExc_TypeError, "request callback not set")
            return std::vector<teca_metadata>();
        }

        // wrap input metadata in a list
        size_t n_elem = input_md.size();
        PyObject *py_md = PyList_New(n_elem);
        for (size_t i = 0; i < n_elem; ++i)
        {
            PyList_SET_ITEM(py_md, i, SWIG_NewPointerObj(
                SWIG_as_voidptr(new teca_metadata(input_md[i])),
                SWIGTYPE_p_teca_metadata, SWIG_POINTER_OWN));
        }

        // wrap the request
        PyObject *py_req = SWIG_NewPointerObj(
            SWIG_as_voidptr(new teca_metadata(request)),
            SWIGTYPE_p_teca_metadata, SWIG_POINTER_OWN);

        // call the callback
        PyObject *args = Py_BuildValue("INN", port, py_md, py_req);

        PyObject *ret = nullptr;
        if (!(ret = PyObject_CallObject(f, args)) || PyErr_Occurred())
        {
            TECA_PY_CALLBACK_ERROR(request, f)
            return std::vector<teca_metadata>();
        }

        Py_DECREF(args);

        // convert the return
        if (!PySequence_Check(ret))
        {
            TECA_PY_ERROR_NOW(PyExc_TypeError,
                "Invalid return type from request callback. "
                "Expecting a list of teca_metadata objects.")
            return std::vector<teca_metadata>();
        }

        size_t n_items = PySequence_Size(ret);
        std::vector<teca_metadata> reqs(n_items);
        for (size_t i = 0; i < n_items; ++i)
        {
            PyObject *tmp = PySequence_GetItem(ret, i);
            teca_metadata *md = nullptr;
            if (!SWIG_IsOK(SWIG_ConvertPtr(tmp,
                reinterpret_cast<void**>(&md), SWIGTYPE_p_teca_metadata, 0))
                || !md)
            {
                TECA_PY_ERROR_NOW(PyExc_TypeError,
                    "invalid return type from request callback")
                return std::vector<teca_metadata>();
            }
            reqs[i] = *md;
        }

        return reqs;
    }

private:
    teca_py_object::teca_py_callable m_callback;
};

/// wrapper for execute_callback phase callback
/** Manages a python callback for use during the execute_callback phase of
 * pipeline execution. In addition to holding the callback it handles translation
 * of the input and output arguments.
 *
 * the python function must accept the following arguments:
 *
 * >  port - an integer set to the active port number
 * >  input_data - a list of datasets objects one per input connection,
 * >               per upstream request
 * >  request - a metadata object containing the request
 *
 * it must return: a teca_dataset object.
 */
class TECA_EXPORT execute_callback
{
public:
    execute_callback(PyObject *f)
        : m_callback(f) {}

    PyObject *set_callback(PyObject *f)
    { return m_callback.set_object(f); }

    explicit operator bool() const
    { return static_cast<bool>(m_callback); }

    const_p_teca_dataset operator()(unsigned int port,
        const std::vector<const_p_teca_dataset> &in_data,
        const teca_metadata &request)
    {
        teca_py_gil_state gil;

        PyObject *f = m_callback.get_object();
        if (!f)
        {
            TECA_PY_ERROR_NOW(PyExc_TypeError,
                "execute_callback callback not set")
            return nullptr;
        }

        // wrap the data in a list
        size_t n_elem = in_data.size();
        PyObject *py_data = PyList_New(n_elem);
        for (size_t i = 0; i < n_elem; ++i)
        {
            PyList_SET_ITEM(py_data, i, SWIG_NewPointerObj(
                SWIG_as_voidptr(new const_p_teca_dataset(in_data[i])),
                SWIGTYPE_p_std__shared_ptrT_teca_dataset_t,
                SWIG_POINTER_OWN));
        }

        // wrap the request
        PyObject *py_req = SWIG_NewPointerObj(
            SWIG_as_voidptr(new teca_metadata(request)),
            SWIGTYPE_p_teca_metadata, SWIG_POINTER_OWN);

        // call the callback
        PyObject *args = Py_BuildValue("INN", port, py_data, py_req);

        PyObject *ret = nullptr;
        if (!(ret = PyObject_CallObject(f, args)) || PyErr_Occurred())
        {
            TECA_PY_CALLBACK_ERROR(execute, f)
            return nullptr;
        }

        Py_DECREF(args);

        // convert the return
        int i_own = 0;
        p_teca_dataset *tmp_data = nullptr;
        if (!SWIG_IsOK(SWIG_ConvertPtrAndOwn(ret,
            reinterpret_cast<void**>(&tmp_data),
            SWIGTYPE_p_std__shared_ptrT_teca_dataset_t, 0, &i_own))
            || !tmp_data)
        {
            TECA_PY_ERROR_NOW(PyExc_TypeError,
                "invalid return type from execute_callback")
            return nullptr;
        }

        p_teca_dataset out_data(*tmp_data);

        if (i_own)
            delete tmp_data;

        return out_data;
    }

private:
    teca_py_object::teca_py_callable m_callback;
};

/// wrapper for execute_callback phase callback
/** Manages a python callback for use during the execute_callback phase of
 * pipeline execution. In addition to holding the callback it handles translation
 * of the input and output arguments.
 *
 * the python function must accept the following arguments:
 *
 * >  port - an integer set to the active port number
 * >  input_data - a list of datasets objects one per input connection,
 * >               per upstream request
 * >  request - a metadata object containing the request
 * >  streaming - flag indicating when processing is finished
 *
 * it must return: a teca_dataset object.
*/
class TECA_EXPORT threaded_execute_callback
{
public:
    threaded_execute_callback(PyObject *f)
        : m_callback(f) {}

    PyObject *set_callback(PyObject *f)
    { return m_callback.set_object(f); }

    explicit operator bool() const
    { return static_cast<bool>(m_callback); }

    const_p_teca_dataset operator()(unsigned int port,
        const std::vector<const_p_teca_dataset> &in_data,
        const teca_metadata &request, int streaming)
    {
        teca_py_gil_state gil;

        PyObject *f = m_callback.get_object();
        if (!f)
        {
            TECA_PY_ERROR_NOW(PyExc_TypeError,
                "execute_callback callback not set")
            return nullptr;
        }

        // wrap the data in a list
        size_t n_elem = in_data.size();
        PyObject *py_data = PyList_New(n_elem);
        for (size_t i = 0; i < n_elem; ++i)
        {
            PyList_SET_ITEM(py_data, i, SWIG_NewPointerObj(
                SWIG_as_voidptr(new const_p_teca_dataset(in_data[i])),
                SWIGTYPE_p_std__shared_ptrT_teca_dataset_t,
                SWIG_POINTER_OWN));
        }

        // wrap the request
        PyObject *py_req = SWIG_NewPointerObj(
            SWIG_as_voidptr(new teca_metadata(request)),
            SWIGTYPE_p_teca_metadata, SWIG_POINTER_OWN);

        // call the callback
        PyObject *args = Py_BuildValue("INNI", port, py_data, py_req, streaming);

        PyObject *ret = nullptr;
        if (!(ret = PyObject_CallObject(f, args)) || PyErr_Occurred())
        {
            TECA_PY_CALLBACK_ERROR(execute, f)
            return nullptr;
        }

        Py_DECREF(args);

        // convert the return
        int i_own = 0;
        p_teca_dataset *tmp_data = nullptr;
        if (!SWIG_IsOK(SWIG_ConvertPtrAndOwn(ret,
            reinterpret_cast<void**>(&tmp_data),
            SWIGTYPE_p_std__shared_ptrT_teca_dataset_t, 0, &i_own))
            || !tmp_data)
        {
            TECA_PY_ERROR_NOW(PyExc_TypeError,
                "invalid return type from execute_callback")
            return nullptr;
        }

        p_teca_dataset out_data(*tmp_data);

        if (i_own)
            delete tmp_data;

        return out_data;
    }

private:
    teca_py_object::teca_py_callable m_callback;
};

/// wrapper for reduce_callback of the programmable reduction
/**
 * Manages a python callback for use during a programmable reduction. In addition
 * to holding the callback it handles translation of the input and output
 * arguments.
 *
 * the python function must accept the two datasets to reduce and return the
 * reduced data
*/
class TECA_EXPORT reduce_callback
{
public:
    reduce_callback(PyObject *f)
        : m_callback(f) {}

    PyObject *set_callback(PyObject *f)
    { return m_callback.set_object(f); }

    explicit operator bool() const
    { return static_cast<bool>(m_callback); }

    p_teca_dataset operator()(int device_id,
        const const_p_teca_dataset &input_0,
        const const_p_teca_dataset &input_1)
    {
        teca_py_gil_state gil;

        PyObject *f = m_callback.get_object();
        if (!f)
        {
            TECA_PY_ERROR_NOW(PyExc_TypeError,
                "reduce_callback callback not set")
            return nullptr;
        }

        // package input datasets
        PyObject *py_input_0 = nullptr;
        if (input_0)
        {
            py_input_0 = SWIG_NewPointerObj(
                SWIG_as_voidptr(new const_p_teca_dataset(input_0)),
                SWIGTYPE_p_std__shared_ptrT_teca_dataset_t,
                SWIG_POINTER_OWN);
        }
        else
        {
            py_input_0 = Py_None;
            Py_INCREF(Py_None);
        }

        PyObject *py_input_1 = nullptr;
        if (input_1)
        {
            py_input_1 = SWIG_NewPointerObj(
                SWIG_as_voidptr(new const_p_teca_dataset(input_1)),
                SWIGTYPE_p_std__shared_ptrT_teca_dataset_t,
                SWIG_POINTER_OWN);
        }
        else
        {
            py_input_1 = Py_None;
            Py_INCREF(Py_None);
        }

        // call the callback
        PyObject *args =
            Py_BuildValue("iNN", device_id, py_input_0, py_input_1);

        PyObject *ret = nullptr;
        if (!(ret = PyObject_CallObject(f, args)) || PyErr_Occurred())
        {
            TECA_PY_CALLBACK_ERROR(reduce, f)
            return nullptr;
        }

        Py_DECREF(args);

        // convert the return
        int i_own = 0;
        p_teca_dataset *tmp_data = nullptr;
        if (!SWIG_IsOK(SWIG_ConvertPtrAndOwn(ret,
            reinterpret_cast<void**>(&tmp_data),
            SWIGTYPE_p_std__shared_ptrT_teca_dataset_t, 0, &i_own))
            || !tmp_data)
        {
            // this is not necessarily an error
            return nullptr;
        }

        p_teca_dataset out_data(*tmp_data);

        if (i_own)
            delete tmp_data;

        return out_data;
    }

private:
    teca_py_object::teca_py_callable m_callback;
};

/// wrapper for finalize_callback of the programmable reduction
/**
 * Manages a python callback for use during a programmable reduction. In addition
 * to holding the callback it handles translation of the input and output
 * arguments.
 *
 * the python function must accept the dataset to finalize and return the
 * finalized data
*/
class TECA_EXPORT finalize_callback
{
public:
    finalize_callback(PyObject *f)
        : m_callback(f) {}

    PyObject *set_callback(PyObject *f)
    { return m_callback.set_object(f); }

    explicit operator bool() const
    { return static_cast<bool>(m_callback); }

    p_teca_dataset operator()(int device_id, const const_p_teca_dataset &ds)
    {
        teca_py_gil_state gil;

        PyObject *f = m_callback.get_object();
        if (!f)
        {
            TECA_PY_ERROR_NOW(PyExc_TypeError,
                "finalize_callback callback not set")
            return nullptr;
        }

        // package input dataset
        PyObject *py_ds = nullptr;
        if (ds)
        {
            py_ds = SWIG_NewPointerObj(
                SWIG_as_voidptr(new const_p_teca_dataset(ds)),
                SWIGTYPE_p_std__shared_ptrT_teca_dataset_t,
                SWIG_POINTER_OWN);
        }
        else
        {
            py_ds = Py_None;
            Py_INCREF(Py_None);
        }

        // call the callback
        PyObject *args =
            Py_BuildValue("(iN)", device_id, py_ds);

        PyObject *ret = nullptr;
        if (!(ret = PyObject_CallObject(f, args)) || PyErr_Occurred())
        {
            TECA_PY_CALLBACK_ERROR(reduce, f)
            return nullptr;
        }

        Py_DECREF(args);

        // convert the return
        int i_own = 0;
        p_teca_dataset *tmp_data = nullptr;
        if (!SWIG_IsOK(SWIG_ConvertPtrAndOwn(ret,
            reinterpret_cast<void**>(&tmp_data),
            SWIGTYPE_p_std__shared_ptrT_teca_dataset_t, 0, &i_own))
            || !tmp_data)
        {
            // this is not necessarily an error
            return nullptr;
        }

        p_teca_dataset out_data(*tmp_data);

        if (i_own)
            delete tmp_data;

        return out_data;
    }

private:
    teca_py_object::teca_py_callable m_callback;
};
}

#endif
