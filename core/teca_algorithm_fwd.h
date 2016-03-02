#ifndef teca_algorithm_fwd_h
#define teca_algorithm_fwd_h

#include "teca_shared_object.h"

TECA_SHARED_OBJECT_FORWARD_DECL(teca_algorithm)

// this is a convenience macro to be used to declare a static
// New method that will be used to construct new objects in
// shared_ptr's. This manages the details of interoperability
// with std C++11 shared pointer
#define TECA_ALGORITHM_STATIC_NEW(T)                                \
                                                                    \
static p_##T New()                                                  \
{                                                                   \
    return p_##T(new T);                                            \
}                                                                   \
                                                                    \
using teca_algorithm::shared_from_this;                             \
                                                                    \
std::shared_ptr<T> shared_from_this()                               \
{                                                                   \
    return std::static_pointer_cast<T>(shared_from_this());         \
}                                                                   \
                                                                    \
std::shared_ptr<T const> shared_from_this() const                   \
{                                                                   \
    return std::static_pointer_cast<T const>(shared_from_this());   \
}

// this convenience macro removes copy and aassignment operators
// which generally should not be defined.
#define TECA_ALGORITHM_DELETE_COPY_ASSIGN(T)    \
                                                \
    T(const T &src) = delete;                   \
    T(T &&src) = delete;                        \
                                                \
    T &operator=(const T &src) = delete;        \
    T &operator=(T &&src) = delete;

// convenience macro to declare standard set_X/get_X methods
// where X is the name of a class member. will manage the
// algorithm's modified state for the user.
#define TECA_ALGORITHM_PROPERTY(T, NAME) \
                                         \
void set_##NAME(const T &v)              \
{                                        \
    if (this->NAME != v)                 \
    {                                    \
        this->NAME = v;                  \
        this->set_modified();            \
    }                                    \
}                                        \
                                         \
const T &get_##NAME() const              \
{                                        \
    return this->NAME;                   \
}                                        \
                                         \
T &get_##NAME()                          \
{                                        \
    return this->NAME;                   \
}

// convenience macro to declare standard set_X/get_X methods
// where X is the name of a class member. will manage the
// algorithm's modified state for the user.
#define TECA_ALGORITHM_VECTOR_PROPERTY(T, NAME) \
                                            \
size_t get_number_of_##NAME##s ()           \
{                                           \
    return this->NAME##s.size();            \
}                                           \
                                            \
void append_##NAME(const T &v)              \
{                                           \
    this->NAME##s.push_back(v);             \
}                                           \
                                            \
void set_##NAME(size_t i, const T &v)       \
{                                           \
    if (this->NAME##s[i] != v)              \
    {                                       \
        this->NAME##s[i] = v;               \
        this->set_modified();               \
    }                                       \
}                                           \
                                            \
void set_##NAME##s(const std::vector<T> &v) \
{                                           \
    if (this->NAME##s != v)                 \
    {                                       \
        this->NAME##s = v;                  \
        this->set_modified();               \
    }                                       \
}                                           \
                                            \
const T &get_##NAME(size_t i) const         \
{                                           \
    return this->NAME##s[i];                \
}                                           \
                                            \
T &get_##NAME(size_t i)                     \
{                                           \
    return this->NAME##s[i];                \
}                                           \
                                            \
void clear_##NAME()                         \
{                                           \
    this->NAME##s.clear();                  \
}

#endif
