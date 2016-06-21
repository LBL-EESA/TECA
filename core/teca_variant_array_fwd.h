#ifndef teca_variant_array_fwd_h
#define teca_variant_array_fwd_h

#include <string>
#include "teca_shared_object.h"

TECA_SHARED_OBJECT_FORWARD_DECL(teca_variant_array)
TECA_SHARED_OBJECT_TEMPLATE_FORWARD_DECL(teca_variant_array_impl)

#ifndef SWIG
// convenience defs for POD types
// these should not be used in API, use teca_variant_array instead
using teca_string_array = teca_variant_array_impl<std::string>;
using p_teca_string_array = std::shared_ptr<teca_variant_array_impl<std::string>>;
using const_p_teca_string_array = std::shared_ptr<const teca_variant_array_impl<std::string>>;

using teca_float_array = teca_variant_array_impl<float>;
using p_teca_float_array = std::shared_ptr<teca_variant_array_impl<float>>;
using const_p_teca_float_array = std::shared_ptr<const teca_variant_array_impl<float>>;

using teca_double_array = teca_variant_array_impl<double>;
using p_teca_double_array = std::shared_ptr<teca_variant_array_impl<double>>;
using const_p_teca_double_array = std::shared_ptr<const teca_variant_array_impl<double>>;

using teca_char_array = teca_variant_array_impl<char>;
using p_teca_char_array = std::shared_ptr<teca_variant_array_impl<char>>;
using const_p_teca_char_array = std::shared_ptr<const teca_variant_array_impl<char>>;

using teca_unsigned_char_array = teca_variant_array_impl<unsigned char>;
using p_teca_unsigned_char_array = std::shared_ptr<teca_variant_array_impl<unsigned char>>;
using const_p_teca_unsigned_char_array = std::shared_ptr<const teca_variant_array_impl<unsigned char>>;

using teca_short_array = teca_variant_array_impl<short>;
using p_teca_short_array = std::shared_ptr<teca_variant_array_impl<short>>;
using const_p_teca_short_array = std::shared_ptr<const teca_variant_array_impl<short>>;

using teca_unsigned_short_array = teca_variant_array_impl<unsigned short>;
using p_teca_unsigned_short_array = std::shared_ptr<teca_variant_array_impl<unsigned short>>;
using const_p_teca_unsigned_short_array = std::shared_ptr<const teca_variant_array_impl<unsigned short>>;

using teca_int_array = teca_variant_array_impl<int>;
using p_teca_int_array = std::shared_ptr<teca_variant_array_impl<int>>;
using const_p_teca_int_array = std::shared_ptr<const teca_variant_array_impl<int>>;

using teca_unsigned_int_array = teca_variant_array_impl<unsigned int>;
using p_teca_unsigned_int_array = std::shared_ptr<teca_variant_array_impl<unsigned int>>;
using const_p_teca_unsigned_int_array = std::shared_ptr<const teca_variant_array_impl<unsigned int>>;

using teca_long_array = teca_variant_array_impl<long>;
using p_teca_long_array = std::shared_ptr<teca_variant_array_impl<long>>;
using const_p_teca_long_array = std::shared_ptr<const teca_variant_array_impl<long>>;

using teca_unsigned_long_array = teca_variant_array_impl<unsigned long>;
using p_teca_unsigned_long_array = std::shared_ptr<teca_variant_array_impl<unsigned long>>;
using const_p_teca_unsigned_long_array = std::shared_ptr<const teca_variant_array_impl<unsigned long>>;

using teca_long_long_array = teca_variant_array_impl<long long>;
using p_teca_long_long_array = std::shared_ptr<teca_variant_array_impl<long long>>;
using const_p_teca_long_long_array = std::shared_ptr<const teca_variant_array_impl<long long>>;

using teca_unsigned_long_long_array = teca_variant_array_impl<unsigned long long>;
using p_teca_unsigned_long_long_array = std::shared_ptr<teca_variant_array_impl<unsigned long long>>;
using const_p_teca_unsigned_long_long_array = std::shared_ptr<const teca_variant_array_impl<unsigned long long>>;
#endif

// this is a convenience macro to be used to declare a static
// New method that will be used to construct new objects in
// shared_ptr's. This manages the details of interoperability
// with std C++11 shared pointer
#define TECA_VARIANT_ARRAY_STATIC_NEW(T, t)                             \
                                                                        \
static std::shared_ptr<T<t>> New()                                      \
{                                                                       \
    return std::shared_ptr<T<t>>(new T<t>);                             \
}                                                                       \
                                                                        \
static std::shared_ptr<T<t>> New(size_t n)                              \
{                                                                       \
    return std::shared_ptr<T<t>>(new T<t>(n));                          \
}                                                                       \
                                                                        \
static std::shared_ptr<T<t>> New(size_t n, const t &v)                  \
{                                                                       \
    return std::shared_ptr<T<t>>(new T<t>(n, v));                       \
}                                                                       \
                                                                        \
static std::shared_ptr<T<t>> New(const t *vals, size_t n)               \
{                                                                       \
    return std::shared_ptr<T<t>>(new T<t>(vals, n));                    \
}                                                                       \
                                                                        \
using teca_variant_array::shared_from_this;                             \
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
