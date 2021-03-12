#ifndef teca_algorithm_fwd_h
#define teca_algorithm_fwd_h

#include "teca_shared_object.h"

#include <initializer_list>
#include <functional>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_algorithm)

// this is a convenience macro to be used to declare a static
// New method that will be used to construct new objects in
// shared_ptr's. This manages the details of interoperability
// with std C++11 shared pointer
#define TECA_ALGORITHM_STATIC_NEW(T)                \
                                                    \
static p_##T New()                                  \
{                                                   \
    return p_##T(new T);                            \
}                                                   \
                                                    \
std::shared_ptr<T> shared_from_this()               \
{                                                   \
    return std::static_pointer_cast<T>(             \
        teca_algorithm::shared_from_this());        \
}                                                   \
                                                    \
std::shared_ptr<T const> shared_from_this() const   \
{                                                   \
    return std::static_pointer_cast<T const>(       \
        teca_algorithm::shared_from_this());        \
}

#define TECA_ALGORITHM_CLASS_NAME(T)                \
/** returns the name of the class */                \
const char *get_class_name() const override         \
{                                                   \
    return #T;                                      \
}

// this convenience macro removes copy and aassignment operators
// which generally should not be defined for reference counted types
#define TECA_ALGORITHM_DELETE_COPY_ASSIGN(T)    \
                                                \
    T(const T &src) = delete;                   \
    T(T &&src) = delete;                        \
                                                \
    T &operator=(const T &src) = delete;        \
    T &operator=(T &&src) = delete;

// convenience macro to declare standard set_NAME/get_NAME methods
// where NAME is the name of a class member. will manage the
// algorithm's modified state for the user.
#define TECA_ALGORITHM_PROPERTY(T, NAME)                 \
                                                         \
/** Set the value of the @ref NAME algorithm property */ \
void set_##NAME(const T &v)                              \
{                                                        \
    if (this->NAME != v)                                 \
    {                                                    \
        this->NAME = v;                                  \
        this->set_modified();                            \
    }                                                    \
}                                                        \
                                                         \
/** Get the value of the @ref NAME algorithm property */ \
const T &get_##NAME() const                              \
{                                                        \
    return this->NAME;                                   \
}

// similar to TECA_ALGORITHM_PROPERTY but prior to setting NAME
// will call the member function int valididate_NAME(T v). If
// the value v is valid the fucntion should return 0. If the value
// is not zero the function should invoke TECA_ERROR with a
// descriptive message and return non-zero.
#define TECA_ALGORITHM_PROPERTY_V(T, NAME) \
                                           \
void set_##NAME(const T &v)                \
{                                          \
    if (this->validate_ ## NAME (v))       \
        return;                            \
                                           \
    if (this->NAME != v)                   \
    {                                      \
        this->NAME = v;                    \
        this->set_modified();              \
    }                                      \
}                                          \
                                           \
const T &get_##NAME() const                \
{                                          \
    return this->NAME;                     \
}

// convenience macro to declare standard set_NAME/get_NAME methods
// where NAME is the name of a class member. will manage the
// algorithm's modified state for the user.
#define TECA_ALGORITHM_VECTOR_PROPERTY(T, NAME)                           \
                                                                          \
/** get the size of the @ref NAME##s algorithm vector property */         \
size_t get_number_of_##NAME##s ()                                         \
{                                                                         \
    return this->NAME##s.size();                                          \
}                                                                         \
                                                                          \
/** append to the @ref NAME##s algorithm vector property */               \
void append_##NAME(const T &v)                                            \
{                                                                         \
    this->NAME##s.push_back(v);                                           \
    this->set_modified();                                                 \
}                                                                         \
                                                                          \
/** set the i-th element of the @ref NAME##s algorithm vector property */ \
void set_##NAME(size_t i, const T &v)                                     \
{                                                                         \
    if (this->NAME##s[i] != v)                                            \
    {                                                                     \
        this->NAME##s[i] = v;                                             \
        this->set_modified();                                             \
    }                                                                     \
}                                                                         \
                                                                          \
/** set the @ref NAME##s algorithm vector property */                     \
void set_##NAME##s(const std::vector<T> &v)                               \
{                                                                         \
    if (this->NAME##s != v)                                               \
    {                                                                     \
        this->NAME##s = v;                                                \
        this->set_modified();                                             \
    }                                                                     \
}                                                                         \
                                                                          \
/** set the @ref NAME##s algorithm vector property */                     \
void set_##NAME##s(const std::initializer_list<T> &&l)                    \
{                                                                         \
    std::vector<T> v(l);                                                  \
    if (this->NAME##s != v)                                               \
    {                                                                     \
        this->NAME##s = v;                                                \
        this->set_modified();                                             \
    }                                                                     \
}                                                                         \
                                                                          \
/** get the i-th element of the @ref NAME##s algorithm vector property */ \
const T &get_##NAME(size_t i) const                                       \
{                                                                         \
    return this->NAME##s[i];                                              \
}                                                                         \
                                                                          \
/** get the @ref NAME##s algorithm vector property */                     \
const std::vector<T> &get_##NAME##s() const                               \
{                                                                         \
    return this->NAME##s;                                                 \
}                                                                         \
                                                                          \
/** clear the @ref NAME##s algorithm vector property */                   \
void clear_##NAME##s()                                                    \
{                                                                         \
    this->NAME##s.clear();                                                \
}


// helper that allows us to use std::function
// as a TECA_ALGORITHM_PROPERTY
template<typename T>
bool operator!=(const std::function<T> &lhs, const std::function<T> &rhs)
{
    return &rhs != &lhs;
}

// TODO -- this is a work around for older versions
// of Apple clang
// Apple LLVM version 4.2 (clang-425.0.28) (based on LLVM 3.2svn)
// Target: x86_64-apple-darwin12.6.0
#define TECA_ALGORITHM_CALLBACK_PROPERTY(T, NAME)   \
                                                    \
void set_##NAME(const T &v)                         \
{                                                   \
    /*if (this->NAME != v)*/                        \
    /*{*/                                           \
        this->NAME = v;                             \
        this->set_modified();                       \
    /*}*/                                           \
}                                                   \
                                                    \
const T &get_##NAME() const                         \
{                                                   \
    return this->NAME;                              \
}                                                   \
                                                    \
T &get_##NAME()                                     \
{                                                   \
    return this->NAME;                              \
}

#endif
