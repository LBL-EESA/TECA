#ifndef teca_binary_stream_h
#define teca_binary_stream_h

#include "teca_common.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <map>
#include <vector>

// Serialize objects into a binary stream.
class teca_binary_stream
{
public:
    // construct
    teca_binary_stream();
    ~teca_binary_stream() noexcept;

    // copy
    teca_binary_stream(const teca_binary_stream &s);
    const teca_binary_stream &operator=(const teca_binary_stream &other);

    // move
    teca_binary_stream(teca_binary_stream &&s) noexcept;
    const teca_binary_stream &operator=(teca_binary_stream &&other) noexcept;

    // evaluate to true when the stream is not empty.
    operator bool()
    { return m_size != 0; }

    // Release all resources, set to a uninitialized
    // state.
    void clear() noexcept;

    // Alolocate n_bytes for the stream.
    void resize(unsigned long n_bytes);

    // ensures space for n_bytes more to the stream.
    void grow(unsigned long n_bytes);

    // Get a pointer to the stream internal representation.
    unsigned char *get_data() noexcept
    { return m_data; }

    const unsigned char *get_data() const noexcept
    { return m_data; }

    // Get the size of the valid data in the stream.
    // note: the internal buffer may be larger.
    unsigned long size() const noexcept
    { return m_write_p - m_data; }

    // Get the sise of the internal buffer allocated
    // for the stream.
    unsigned long capacity() const noexcept
    { return m_size; }

    // set the stream position to n bytes from the head
    // of the stream
    void set_read_pos(unsigned long n) noexcept
    { m_read_p = m_data + n; }

    void set_write_pos(unsigned long n) noexcept
    { m_write_p = m_data + n; }

    // swap the two objects
    void swap(teca_binary_stream &other) noexcept;

    // Insert/Extract to/from the stream.
    template <typename T> void pack(T *val);
    template <typename T> void pack(const T &val);
    template <typename T> void unpack(T &val);
    template <typename T> void pack(const T *val, unsigned long n);
    template <typename T> void unpack(T *val, unsigned long n);

    // specializations
    void pack(const std::string &str);
    void unpack(std::string &str);

    void pack(const std::vector<std::string> &v);
    void unpack(std::vector<std::string> &v);

    template<typename T> void pack(const std::vector<T> &v);
    template<typename T> void unpack(std::vector<T> &v);

    // broadcast the stream from the root process to all other processes
    int broadcast(int root_rank=0);

private:
    // re-allocation size
    static
    constexpr unsigned int get_block_size()
    { return 512; }

private:
    unsigned long m_size;
    unsigned char *m_data;
    unsigned char *m_read_p;
    unsigned char *m_write_p;
};

//-----------------------------------------------------------------------------
template <typename T>
void teca_binary_stream::pack(T *val)
{
    (void)val;
    TECA_ERROR("Error: Packing a pointer.");
}

//-----------------------------------------------------------------------------
template <typename T>
void teca_binary_stream::pack(const T &val)
{
    this->grow(sizeof(T));
    *((T *)m_write_p) = val;
    m_write_p += sizeof(T);
}

//-----------------------------------------------------------------------------
template <typename T>
void teca_binary_stream::unpack(T &val)
{
    val = *((T *)m_read_p);
    m_read_p += sizeof(T);
}

//-----------------------------------------------------------------------------
template <typename T>
void teca_binary_stream::pack(const T *val, unsigned long n)
{
    unsigned long n_bytes = n*sizeof(T);
    this->grow(n_bytes);

    unsigned long nn = n*sizeof(T);
    memcpy(m_write_p, val, nn);
    m_write_p += nn;
}

//-----------------------------------------------------------------------------
template <typename T>
void teca_binary_stream::unpack(T *val, unsigned long n)
{
    unsigned long nn = n*sizeof(T);
    memcpy(val, m_read_p, nn);
    m_read_p += nn;
}

//-----------------------------------------------------------------------------
inline
void teca_binary_stream::pack(const std::string &str)
{
    unsigned long slen = str.size();
    this->pack(slen);
    this->pack(str.c_str(), slen);
}

//-----------------------------------------------------------------------------
inline
void teca_binary_stream::unpack(std::string &str)
{
    unsigned long slen = 0;
    this->unpack(slen);

    str.resize(slen);
    str.assign(reinterpret_cast<char*>(m_read_p), slen);

    m_read_p += slen;
}

//-----------------------------------------------------------------------------
inline
void teca_binary_stream::pack(const std::vector<std::string> &v)
{
    unsigned long vlen = v.size();
    this->pack(vlen);
    for (unsigned long i = 0; i < vlen; ++i)
        this->pack(v[i]);
}

//-----------------------------------------------------------------------------
inline
void teca_binary_stream::unpack(std::vector<std::string> &v)
{
    unsigned long vlen;
    this->unpack(vlen);

    v.resize(vlen);
    for (unsigned long i = 0; i < vlen; ++i)
        this->unpack(v[i]);
}

//-----------------------------------------------------------------------------
template<typename T>
void teca_binary_stream::pack(const std::vector<T> &v)
{
    const unsigned long vlen = v.size();
    this->pack(vlen);
    this->pack(v.data(), vlen);
}

//-----------------------------------------------------------------------------
template<typename T>
void teca_binary_stream::unpack(std::vector<T> &v)
{
    unsigned long vlen;
    this->unpack(vlen);

    v.resize(vlen);
    this->unpack(v.data(), vlen);
}

#endif
