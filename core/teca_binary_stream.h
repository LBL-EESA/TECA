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
    void resize(size_t n_bytes);

    // ensures space for n_bytes more to the stream.
    void grow(size_t n_bytes);

    // Get a pointer to the stream internal representation.
    unsigned char *get_data() noexcept
    { return m_data; }

    // Get the size of the valid data in the stream.
    // note: the internal buffer may be larger.
    size_t size() const noexcept
    { return m_data_p - m_data; }

    // Set the stream position to the head of the stream
    void rewind() noexcept
    { m_data_p = m_data; }

    // swap the two objects
    void swap(teca_binary_stream &other) noexcept;

    // Insert/Extract to/from the stream.
    template <typename T> void pack(T *val);
    template <typename T> void pack(const T &val);
    template <typename T> void unpack(T &val);
    template <typename T> void pack(const T *val, size_t n);
    template <typename T> void unpack(T *val, size_t n);

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
    size_t m_size;
    unsigned char *m_data;
    unsigned char *m_data_p;
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
    *((T *)m_data_p) = val;
    m_data_p += sizeof(T);
}

//-----------------------------------------------------------------------------
template <typename T>
void teca_binary_stream::unpack(T &val)
{
    val = *((T *)m_data_p);
    m_data_p += sizeof(T);
}

//-----------------------------------------------------------------------------
template <typename T>
void teca_binary_stream::pack(const T *val, size_t n)
{
    size_t n_bytes = n*sizeof(T);
    this->grow(n_bytes);

    size_t nn = n*sizeof(T);
    memcpy(m_data_p, val, nn);
    m_data_p += nn;
}

//-----------------------------------------------------------------------------
template <typename T>
void teca_binary_stream::unpack(T *val, size_t n)
{
    size_t nn = n*sizeof(T);
    memcpy(val, m_data_p, nn);
    m_data_p += nn;
}

//-----------------------------------------------------------------------------
inline
void teca_binary_stream::pack(const std::string &str)
{
    size_t slen = str.size();
    this->pack(slen);
    this->pack(str.c_str(), slen);
}

//-----------------------------------------------------------------------------
inline
void teca_binary_stream::unpack(std::string &str)
{
    size_t slen = 0;
    this->unpack(slen);

    str.resize(slen);
    str.assign(reinterpret_cast<char*>(m_data_p), slen);

    m_data_p += slen;
}

//-----------------------------------------------------------------------------
inline
void teca_binary_stream::pack(const std::vector<std::string> &v)
{
    size_t vlen = v.size();
    this->pack(vlen);
    for (size_t i = 0; i < vlen; ++i)
        this->pack(v[i]);
}

//-----------------------------------------------------------------------------
inline
void teca_binary_stream::unpack(std::vector<std::string> &v)
{
    size_t vlen;
    this->unpack(vlen);

    v.resize(vlen);
    for (size_t i = 0; i < vlen; ++i)
        this->unpack(v[i]);
}

//-----------------------------------------------------------------------------
template<typename T>
void teca_binary_stream::pack(const std::vector<T> &v)
{
    const size_t vlen = v.size();
    this->pack(vlen);
    this->pack(v.data(), vlen);
}

//-----------------------------------------------------------------------------
template<typename T>
void teca_binary_stream::unpack(std::vector<T> &v)
{
    size_t vlen;
    this->unpack(vlen);

    v.resize(vlen);
    this->unpack(v.data(), vlen);
}

#endif
