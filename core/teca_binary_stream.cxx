#include "teca_binary_stream.h"

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

//-----------------------------------------------------------------------------

teca_binary_stream::teca_binary_stream()
     : m_size(0), m_data(nullptr), m_read_p(nullptr), m_write_p(nullptr)
{}

//-----------------------------------------------------------------------------
teca_binary_stream::~teca_binary_stream() noexcept
{
    this->clear();
}

//-----------------------------------------------------------------------------
teca_binary_stream::teca_binary_stream(const teca_binary_stream &other)
     : m_size(0), m_data(nullptr), m_read_p(nullptr), m_write_p(nullptr)
{ *this = other; }

//-----------------------------------------------------------------------------
teca_binary_stream::teca_binary_stream(teca_binary_stream &&other) noexcept
     : m_size(0), m_data(nullptr), m_read_p(nullptr), m_write_p(nullptr)
{ this->swap(other); }

//-----------------------------------------------------------------------------
const teca_binary_stream &teca_binary_stream::operator=(
    const teca_binary_stream &other)
{
    if (&other == this)
        return *this;

    this->resize(other.m_size);
    unsigned long in_use = other.m_write_p - other.m_data;
    memcpy(m_data, other.m_data, in_use);
    m_write_p = m_data + in_use;
    m_read_p = m_data + (other.m_read_p - other.m_data);

    return *this;
}

//-----------------------------------------------------------------------------
const teca_binary_stream &teca_binary_stream::operator=(
    teca_binary_stream &&other) noexcept
{
    teca_binary_stream tmp(std::move(other));
    this->swap(tmp);
    return *this;
}

//-----------------------------------------------------------------------------
void teca_binary_stream::clear() noexcept
{
    free(m_data);
    m_data = nullptr;
    m_read_p = nullptr;
    m_write_p = nullptr;
    m_size = 0;
}

//-----------------------------------------------------------------------------
void teca_binary_stream::resize(unsigned long n_bytes)
{
    // no change
    if (n_bytes == m_size)
        return;

    // free
    if (n_bytes == 0)
    {
        this->clear();
        return;
    }

    // shrink
    if (n_bytes < m_size)
    {
        unsigned char *end =  m_data + n_bytes;
        if (m_write_p >= end)
            m_write_p = end;
        return;
    }

    // grow
    unsigned char *orig_m_data = m_data;
    m_data = (unsigned char *)realloc(m_data, n_bytes);

    // update the stream pointer
    if (m_data != orig_m_data)
    {
        m_write_p = m_data + (m_write_p - orig_m_data);
        m_read_p = m_data + (m_read_p - orig_m_data);
    }

    m_size = n_bytes;
}

//-----------------------------------------------------------------------------
void teca_binary_stream::grow(unsigned long n_bytes)
{
    unsigned long n_bytes_needed = this->size() + n_bytes;
    if (n_bytes_needed > m_size)
    {
        unsigned long new_size = m_size + this->get_block_size();
        while (new_size < n_bytes_needed)
            new_size += this->get_block_size();
        this->resize(new_size);
    }
}

//-----------------------------------------------------------------------------
void teca_binary_stream::swap(teca_binary_stream &other) noexcept
{
    std::swap(m_data, other.m_data);
    std::swap(m_write_p, other.m_write_p);
    std::swap(m_read_p, other.m_read_p);
    std::swap(m_size, other.m_size);
}

//-----------------------------------------------------------------------------
int teca_binary_stream::broadcast(int root_rank)
{
#if defined(TECA_HAS_MPI)
    int init = 0;
    int rank = 0;
    MPI_Initialized(&init);
    if (init)
    {
        unsigned long nbytes = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == root_rank)
        {
            nbytes = this->size();
            MPI_Bcast(&nbytes, 1, MPI_UNSIGNED_LONG, root_rank, MPI_COMM_WORLD);
            MPI_Bcast(this->get_data(), nbytes, MPI_BYTE, root_rank, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Bcast(&nbytes, 1, MPI_UNSIGNED_LONG, root_rank, MPI_COMM_WORLD);
            this->resize(nbytes);
            MPI_Bcast(this->get_data(), nbytes, MPI_BYTE, root_rank, MPI_COMM_WORLD);
            this->set_read_pos(0);
            this->set_write_pos(nbytes);
        }
    }
#else
    (void)root_rank;
#endif
    return 0;
}
