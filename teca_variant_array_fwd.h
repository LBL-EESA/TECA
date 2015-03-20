#ifndef teca_variant_array_fwd_h
#define teca_variant_array_fwd_h

#include <memory>

class teca_variant_array;
using p_teca_variant_array = std::shared_ptr<teca_variant_array>;
using const_p_teca_variant_array = std::shared_ptr<const teca_variant_array>;

template<typename T>
class teca_variant_array_impl;

template<typename T>
using p_teca_variant_array_impl = std::shared_ptr<teca_variant_array_impl<T>>;

template<typename T>
using const_p_teca_variant_array_impl = std::shared_ptr<teca_variant_array_impl<const T>>;

// convenience defs for POD types
using teca_float_array = teca_variant_array_impl<float>;
using teca_double_array = teca_variant_array_impl<double>;
using teca_int_array = teca_variant_array_impl<int>;
using teca_unsigned_int_array = teca_variant_array_impl<unsigned int>;
using teca_char_array = teca_variant_array_impl<char>;
using teca_unsigned_char_array = teca_variant_array_impl<unsigned char>;
using teca_long_array = teca_variant_array_impl<long>;
using teca_unsigned_long_array = teca_variant_array_impl<unsigned long>;
using teca_long_long_array = teca_variant_array_impl<long long>;
using teca_unsigned_long_long_array = teca_variant_array_impl<unsigned long long>;

// this is a convenience macro to be used to declare a static
// New method that will be used to construct new objects in
// shared_ptr's. This manages the details of interoperability
// with std C++11 shared pointer
#define TECA_VARIANT_ARRAY_STATIC_NEW(T, t)                             \
                                                                        \
static p_##T<t> New()                                                   \
{                                                                       \
    return p_##T<t>(new T<t>);                                          \
}                                                                       \
                                                                        \
static p_##T<t> New(size_t n)                                           \
{                                                                       \
    return p_##T<t>(new T<t>(n));                                       \
}                                                                       \
                                                                        \
static p_##T<t> New(const t *vals, size_t n)                            \
{                                                                       \
    return p_##T<t>(new T<t>(vals, n));                                 \
}                                                                       \
                                                                        \
using enable_shared_from_this<teca_variant_array>::shared_from_this;    \
                                                                        \
std::shared_ptr<T> shared_from_this()                                   \
{                                                                       \
    return std::static_pointer_cast<T>(shared_from_this());             \
}                                                                       \
                                                                        \
std::shared_ptr<T const> shared_from_this() const                       \
{                                                                       \
    return std::static_pointer_cast<T const>(shared_from_this());       \
}

#endif
