#include "teca_binary_stream.h"

//-----------------------------------------------------------------------------

teca_binary_stream::teca_binary_stream()
     : m_size(0), m_data(0), m_data_p(0)
{}

//-----------------------------------------------------------------------------
teca_binary_stream::~teca_binary_stream()
{
    this->clear();
}

//-----------------------------------------------------------------------------
teca_binary_stream::teca_binary_stream(const teca_binary_stream &other)
{ *this = other; }

//-----------------------------------------------------------------------------
teca_binary_stream::teca_binary_stream(teca_binary_stream &&other)
     : m_size(0), m_data(0), m_data_p(0)
{ this->swap(other); }


//-----------------------------------------------------------------------------
const teca_binary_stream &teca_binary_stream::operator=(
    const teca_binary_stream &other)
{
    if (&other == this)
        return *this;

    this->clear();
    this->resize(other.m_size);
    m_data_p = other.m_data_p;

    for (size_t i = 0; i < other.m_size; ++i)
        m_data[i] = other.m_data[i];

    return *this;
}

//-----------------------------------------------------------------------------
const teca_binary_stream &teca_binary_stream::operator=(
    teca_binary_stream &&other)
{
    teca_binary_stream tmp(std::move(other));
    this->swap(tmp);
    return *this;
}

//-----------------------------------------------------------------------------
void teca_binary_stream::clear()
{
    free(m_data);
    m_data = 0;
    m_data_p = 0;
    m_size = 0;
}

//-----------------------------------------------------------------------------
void teca_binary_stream::resize(size_t n_bytes)
{
    char *orig_data = m_data;
    m_data = (char *)realloc(m_data, n_bytes);

    // update the stream pointer
    if (m_data != orig_data)
        m_data_p = m_data + (m_data_p - orig_data);

    m_size = n_bytes;
}

//-----------------------------------------------------------------------------
void teca_binary_stream::grow(size_t n_bytes)
{
    this->resize(m_size + n_bytes);
}

//-----------------------------------------------------------------------------
void teca_binary_stream::swap(teca_binary_stream &other)
{
    char *tmp_data = m_data;
    m_data = other.m_data;
    other.m_data = tmp_data;

    char *tmp_data_p = m_data_p;
    m_data_p = other.m_data_p;
    other.m_data_p = tmp_data_p;

    size_t tmp_size = m_size;
    m_size = other.m_size;
    other.m_size = tmp_size;
}
