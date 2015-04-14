#include "teca_binary_stream.h"

//-----------------------------------------------------------------------------

teca_binary_stream::teca_binary_stream()
     : m_size(0), m_data(nullptr), m_data_p(nullptr)
{}

//-----------------------------------------------------------------------------
teca_binary_stream::~teca_binary_stream() TECA_NOEXCEPT
{
    this->clear();
}

//-----------------------------------------------------------------------------
teca_binary_stream::teca_binary_stream(const teca_binary_stream &other)
{ *this = other; }

//-----------------------------------------------------------------------------
teca_binary_stream::teca_binary_stream(teca_binary_stream &&other) TECA_NOEXCEPT
     : m_size(0), m_data(nullptr), m_data_p(nullptr)
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
    teca_binary_stream &&other) TECA_NOEXCEPT
{
    teca_binary_stream tmp(std::move(other));
    this->swap(tmp);
    return *this;
}

//-----------------------------------------------------------------------------
void teca_binary_stream::clear() TECA_NOEXCEPT
{
    free(m_data);
    m_data = nullptr;
    m_data_p = nullptr;
    m_size = 0;
}

//-----------------------------------------------------------------------------
void teca_binary_stream::resize(size_t n_bytes)
{
    // no change
    if (n_bytes == m_size)
        return;

    // free
    if (n_bytes == 0)
        this->clear();

    // shrink
    if (n_bytes < m_size)
    {
        m_size = n_bytes;
        if (m_data_p >= m_data + m_size)
            m_data_p = m_data + m_size - 1;
        return;
    }

    // grow
    unsigned char *orig_data = m_data;
    m_data = (unsigned char *)realloc(m_data, n_bytes);

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
void teca_binary_stream::swap(teca_binary_stream &other) TECA_NOEXCEPT
{
    std::swap(m_data, other.m_data);
    std::swap(m_data_p, other.m_data_p);
    std::swap(m_size, other.m_size);
}
