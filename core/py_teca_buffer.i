/*
 * python 2 -- map vector<T> using the Python Buffer
 *      protocol. This provides support for both numpy
 *      and native python arrays.
 */
%define buffer_to_vector(c_type)
%typemap(in) const std::vector<c_type> & {
    const c_type *buf = NULL;
    Py_ssize_t buf_size = 0;
    if(PyObject_AsReadBuffer($input,
        reinterpret_cast<const void**>(&buf), &buf_size))
    {
        TECA_ERROR("Failed to convert to buffer")
        return NULL;
    }
    $1 = new std::vector<c_type>(buf, buf+(buf_size/sizeof(c_type)));
}
%typemap(freearg) const std::vector<c_type> & {
    delete $1;
}
%enddef

buffer_to_vector(int)
buffer_to_vector(long long)
buffer_to_vector(float)
buffer_to_vector(double)

/* python 3 -- the bufffer protocol is different for
python 3. It has been back ported to python 2 for strings
but not arrays.
%typemap(in) const std::vector<int> & {
    if (!PyObject_CheckBuffer($input))
    {
        Py_buffer buf;
        if (PyObject_GetBuffer($input, &buf, PyBUF_SIMPLE))
        {
            TECA_ERROR("conversion to buffer failed")
            return NULL;
        }
        int *pbuf = static_cast<int*>(buf.buf);
        $1->assign(pbuf, pbuf+buf.len);
    }
    else
    {
        TECA_ERROR("input doesn't support buffer protocol")
        return NULL;
    }
}*/
