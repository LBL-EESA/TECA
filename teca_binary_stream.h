#ifndef teca_binary_stream_h
#define teca_binary_stream_h

#include "teca_common.h"
#include "teca_compiler.h"

#include <cstdlib>
#include <string>
#include <map>
#include <vector>

// Serialize objects into a binary stream.
class teca_binary_stream
{
public:
    // construct
    teca_binary_stream();
    ~teca_binary_stream() TECA_NOEXCEPT;

    // copy
    teca_binary_stream(const teca_binary_stream &s);
    const teca_binary_stream &operator=(const teca_binary_stream &other);

    // move
    teca_binary_stream(teca_binary_stream &&s) TECA_NOEXCEPT;
    const teca_binary_stream &operator=(teca_binary_stream &&other) TECA_NOEXCEPT;

    // Release all resources, set to a uninitialized
    // state.
    void clear() TECA_NOEXCEPT;

    // Alolocate n_bytes for the stream.
    void resize(size_t n_bytes);

    // Add n_bytes to the stream.
    void grow(size_t n_bytes);

    // Get a pointer to the stream internal representation.
    unsigned char *get_data() TECA_NOEXCEPT
    { return m_data; }

    // Get the size of the m_data in the stream.
    size_t size() const TECA_NOEXCEPT
    { return m_data_p - m_data; }

    // Set the stream position to the head of the strem
    void rewind() TECA_NOEXCEPT
    { m_data_p = m_data; }

    // swap the two objects
    void swap(teca_binary_stream &other) TECA_NOEXCEPT;

    // Insert/Extract to/from the stream.
    template <typename T> void pack(T *val);
    template <typename T> void pack(T val);
    template <typename T> void unpack(T &val);
    template <typename T> void pack(const T *val, size_t n);
    template <typename T> void unpack(T *val, size_t n);

    // specializations
    void pack(const std::string &str);
    void unpack(std::string &str);
    template<typename T> void pack(std::vector<T> &v);
    template<typename T> void unpack(std::vector<T> &v);
    /*void pack(std::map<std::string, int> &m);
    void unpack(std::map<std::string, int> &m);*/

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
void teca_binary_stream::pack(T val)
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

    for (size_t i = 0; i < n; ++i, m_data_p += sizeof(T))
        *((T *)m_data_p) = val[i];
}

//-----------------------------------------------------------------------------
template <typename T>
void teca_binary_stream::unpack(T *val, size_t n)
{
    for (size_t i = 0; i < n; ++i, m_data_p += sizeof(T))
        val[i] = *((T *)m_data_p);
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
template<typename T>
void teca_binary_stream::pack(std::vector<T> &v)
{
    const size_t vlen = v.size();
    this->pack(vlen);
    this->pack(&v[0], vlen);
}

//-----------------------------------------------------------------------------
template<typename T>
void teca_binary_stream::unpack(std::vector<T> &v)
{
    size_t vlen;
    this->unpack(vlen);
    v.resize(vlen);
    this->unpack(&v[0], vlen);
}

/*
//-----------------------------------------------------------------------------
inline
void teca_binary_stream::pack(std::map<std::string, int> &m)
{
    size_t mlen = m.size();
    this->pack(mlen);

    std::map<std::string, int>::iterator it = m.begin();
    std::map<std::string, int>::iterator end = m.end();
    for (; it != end; ++it)
    {
        this->pack(it->first);
        this->pack(it->second);
    }
}

//-----------------------------------------------------------------------------
inline
void teca_binary_stream::unpack(std::map<std::string, int> &m)
{
    size_t mlen = 0;
    this->unpack(mlen);

    for (size_t i = 0; i < mlen; ++i)
    {
        std::string key;
        this->unpack(key);

        int val;
        this->unpack(val);

        m[key] = val;
    }
}
*/

#endif
