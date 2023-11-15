#ifndef teca_variant_array_impl_h
#define teca_variant_array_impl_h

/// @file

#if defined(__CUDACC__)
#pragma nv_diag_suppress = partial_override
#endif

#include "teca_config.h"
#include "teca_variant_array.h"
#include "teca_common.h"
#include "teca_binary_stream.h"
#include "teca_bad_cast.h"
#include "teca_shared_object.h"
#include "teca_variant_array_util.h"

#include <hamr_buffer.h>

#include <vector>
#include <string>
#include <sstream>
#include <exception>
#include <typeinfo>
#include <iterator>
#include <algorithm>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <limits>

#if defined(TECA_HAS_CUDA)
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#endif

#if !defined(SWIG)
using namespace teca_variant_array_util;
#endif

class teca_metadata;

TECA_SHARED_OBJECT_TEMPLATE_FORWARD_DECL(teca_variant_array_impl)

#if !defined(SWIG)
template<typename T>
using p_teca_variant_array_impl = std::shared_ptr<teca_variant_array_impl<T>>;

template<typename T>
using const_p_teca_variant_array_impl = std::shared_ptr<const teca_variant_array_impl<T>>;

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

using teca_size_t_array = teca_variant_array_impl<size_t>;
using p_teca_size_t_array = std::shared_ptr<teca_variant_array_impl<size_t>>;
using const_p_teca_size_t_array = std::shared_ptr<const teca_variant_array_impl<size_t>>;
#endif


/** convert from a pointer to a non-const type to a pointer to a const type.
 * this is used to help with template deduction, which occurs before implicit
 * conversions are considered.
 */
template <typename T>
const_p_teca_variant_array_impl<T> const_ptr(const p_teca_variant_array_impl<T> &v)
{
    return const_p_teca_variant_array_impl<T>(v);
}


/// @cond
/// A tag for dispatching operations on POD data
template <typename T>
struct pod_dispatch :
    std::integral_constant<bool,
    std::is_arithmetic<T>::value>
{};

/// A tag for disp[atching operations on classes
template <typename T>
struct object_dispatch :
    std::integral_constant<bool,
    !std::is_arithmetic<T>::value>
{};
/// @endcond

/** Executes the code in body if p is a tt<nt>
 * @param tt     derived container
 * @param nt     contained type
 * @param p      base class pointer
 * @param body   the code to execute if the type matches
 *
 * The following aliases are provided to know the type within the code to execute.
 *
 *      using NT = nt;
 *      using CNT = const nt;
 *      using TT = tt<nt>;
 *      using CTT = const tt<nt>;
 *      using PT = std::shared_ptr<tt<nt>>;
 *      using CPT = std::shared_ptr<const tt<nt>>;
 *      using SP = std::shared_ptr<nt>;
 *      using CSP = std::shared_ptr<const nt>;
 *
 */
#define TEMPLATE_DISPATCH_CASE(tt, nt, p, ...)      \
    if (dynamic_cast<const tt<nt>*>(p))             \
    {                                               \
        using NT = nt;                              \
        using CNT = const nt;                       \
        using TT = tt<nt>;                          \
        using CTT = const tt<nt>;                   \
        using PT = std::shared_ptr<tt<nt>>;         \
        using CPT = std::shared_ptr<const tt<nt>>;  \
        using SP = std::shared_ptr<nt>;             \
        using CSP = std::shared_ptr<const nt>;      \
        __VA_ARGS__                                 \
    }

/** Executes the code in body if p is a tt<nt> an idnetifier disambiguates type
 * aliases  when nested
 *
 * @param tt     derived container
 * @param nt     contained type
 * @param p      base class pointer
 * @param i      identifier
 * @param body   the code to execute if the type matches
 *
 * The following aliases are provided to know the type within the code to execute.
 *
 *      using NT##i = nt;
 *      using CNT##i = const nt;
 *      using TT##i = tt<nt>;
 *      using CTT##i = const tt<nt>;
 *      using PT##i = std::shared_ptr<tt<nt>>;
 *      using CPT##i = std::shared_ptr<const tt<nt>>;
 *      using SP##i = std::shared_ptr<nt>;
 *      using CSP##i = std::shared_ptr<const nt>;
 *
 */
#define NESTED_TEMPLATE_DISPATCH_CASE(tt, nt, p, i, ...)    \
    if (dynamic_cast<const tt<nt>*>(p))                     \
    {                                                       \
        using NT##i = nt;                                   \
        using CNT##i = const nt;                            \
        using TT##i = tt<nt>;                               \
        using CTT##i = const tt<nt>;                        \
        using PT##i = std::shared_ptr<tt<nt>>;              \
        using CPT##i = std::shared_ptr<const tt<nt>>;       \
        using SP##i = std::shared_ptr<nt>;                  \
        using CSP##i = std::shared_ptr<const nt>;           \
        __VA_ARGS__                                         \
    }

/// Executes the code in body if p is a t<nt> where nt is a floating point type
#define TEMPLATE_DISPATCH_FP(t, p, ...)                 \
    TEMPLATE_DISPATCH_CASE(t, float, p, __VA_ARGS__)    \
    else TEMPLATE_DISPATCH_CASE(t, double, p, __VA_ARGS__)

/// Executes the code in body if p is a t<nt> where nt is a signed inetegral type
#define TEMPLATE_DISPATCH_SI(t, p, ...)                         \
    TEMPLATE_DISPATCH_CASE(t, long long, p, __VA_ARGS__)        \
    else TEMPLATE_DISPATCH_CASE(t, long, p, __VA_ARGS__)        \
    else TEMPLATE_DISPATCH_CASE(t, int, p, __VA_ARGS__)         \
    else TEMPLATE_DISPATCH_CASE(t, short int, p, __VA_ARGS__)   \
    else TEMPLATE_DISPATCH_CASE(t, char, p, __VA_ARGS__)

/// Executes the code in body if p is a t<nt> where nt is either a signed integral or floating point type
#define TEMPLATE_DISPATCH_FP_SI(t, p, ...)                  \
    TEMPLATE_DISPATCH_CASE(t, float, p, __VA_ARGS__)        \
    else TEMPLATE_DISPATCH_CASE(t, double, p, __VA_ARGS__)  \
    else TEMPLATE_DISPATCH_SI(t, p, __VA_ARGS__)

/// Executes the code in body if p is a t<nt> where nt is an integral type
#define TEMPLATE_DISPATCH_I(t, p, ...)                                 \
    TEMPLATE_DISPATCH_CASE(t, long long, p, __VA_ARGS__)               \
    else TEMPLATE_DISPATCH_CASE(t, unsigned long long, p, __VA_ARGS__) \
    else TEMPLATE_DISPATCH_CASE(t, long, p, __VA_ARGS__)               \
    else TEMPLATE_DISPATCH_CASE(t, int, p, __VA_ARGS__)                \
    else TEMPLATE_DISPATCH_CASE(t, unsigned int, p, __VA_ARGS__)       \
    else TEMPLATE_DISPATCH_CASE(t, unsigned long, p, __VA_ARGS__)      \
    else TEMPLATE_DISPATCH_CASE(t, short int, p, __VA_ARGS__)          \
    else TEMPLATE_DISPATCH_CASE(t, short unsigned int, p, __VA_ARGS__) \
    else TEMPLATE_DISPATCH_CASE(t, char, p, __VA_ARGS__)               \
    else TEMPLATE_DISPATCH_CASE(t, unsigned char, p, __VA_ARGS__)

/** A macro for accessing the typed contents of a teca_variant_array
 * @param t    container type
 * @param p    pointer to an instance to match on
 * @param body code to execute on match
 *
 * See #TEMPLATE_DISPATCH_CASE for details.
 */
#define TEMPLATE_DISPATCH(t, p, ...)            \
    TEMPLATE_DISPATCH_FP(t, p, __VA_ARGS__)     \
    else TEMPLATE_DISPATCH_I(t, p, __VA_ARGS__)

/** A macro for accessing the typed contents of a teca_variant_array
 * @param t    container type
 * @param p    pointer to an instance to match on
 * @param body code to execute on match
 *
 * See #TEMPLATE_DISPATCH_CASE for details.
 */
#define TEMPLATE_DISPATCH_OBJ(t, p, ...)                            \
    TEMPLATE_DISPATCH_CASE(t, std::string, p, __VA_ARGS__)          \
    else TEMPLATE_DISPATCH_CASE(t, teca_metadata, p, __VA_ARGS__)

/** A macro for accessing the typed contents of a teca_variant_array
 * @param t    container type
 * @param p    pointer to an instance to match on
 * @param body code to execute on match
 *
 * See #TEMPLATE_DISPATCH_CASE for details.
 */
#define TEMPLATE_DISPATCH_PTR(t, p, ...)                                    \
    TEMPLATE_DISPATCH_CASE(t, const_p_teca_variant_array, p, __VA_ARGS__)   \
    else TEMPLATE_DISPATCH_CASE(t, p_teca_variant_array, p, __VA_ARGS__)

/** A macro for accessing the floating point typed contents of a teca_variant_array
 * @param t    container type
 * @param p    pointer to an instance to match on
 * @param i    an indentifier to use with type aliases
 * @param body code to execute on match
 *
 * See #NESTED_TEMPLATE_DISPATCH_CASE for details.
 */
#define NESTED_TEMPLATE_DISPATCH_FP(t, p, i, ...)                       \
    NESTED_TEMPLATE_DISPATCH_CASE(t, float, p, i, __VA_ARGS__)          \
    else NESTED_TEMPLATE_DISPATCH_CASE(t, double, p, i, __VA_ARGS__)

/** A macro for accessing the inetgral typed contents of a teca_variant_array
 * @param t    container type
 * @param p    pointer to an instance to match on
 * @param i    an indentifier to use with type aliases
 * @param body code to execute on match
 *
 * See #NESTED_TEMPLATE_DISPATCH_CASE for details.
 */
#define NESTED_TEMPLATE_DISPATCH_I(t, p, i, ...)                                 \
    NESTED_TEMPLATE_DISPATCH_CASE(t, long long, p, i, __VA_ARGS__)               \
    else NESTED_TEMPLATE_DISPATCH_CASE(t, unsigned long long, p, i, __VA_ARGS__) \
    else NESTED_TEMPLATE_DISPATCH_CASE(t, long, p, i, __VA_ARGS__)               \
    else NESTED_TEMPLATE_DISPATCH_CASE(t, int, p, i, __VA_ARGS__)                \
    else NESTED_TEMPLATE_DISPATCH_CASE(t, unsigned int, p, i, __VA_ARGS__)       \
    else NESTED_TEMPLATE_DISPATCH_CASE(t, unsigned long, p, i, __VA_ARGS__)      \
    else NESTED_TEMPLATE_DISPATCH_CASE(t, short int, p, i, __VA_ARGS__)          \
    else NESTED_TEMPLATE_DISPATCH_CASE(t, short unsigned int, p, i, __VA_ARGS__) \
    else NESTED_TEMPLATE_DISPATCH_CASE(t, char, p, i, __VA_ARGS__)               \
    else NESTED_TEMPLATE_DISPATCH_CASE(t, unsigned char, p, i, __VA_ARGS__)

/** \def NESTED_TEMPLATE_DISPATCH(t, p, i, body)
 * A macro for accessing the typed contents of a teca_variant_array
 * @param t    container type
 * @param p    pointer to an instance to match on
 * @param i    an indentifier to use with type aliases
 * @param body code to execute on match
 *
 * See #NESTED_TEMPLATE_DISPATCH_CASE for details.
 */
#define NESTED_TEMPLATE_DISPATCH(t, p, i, ...)             \
    NESTED_TEMPLATE_DISPATCH_FP(t, p, i, __VA_ARGS__)      \
    else NESTED_TEMPLATE_DISPATCH_I(t, p, i, __VA_ARGS__)

/** shortcuts for NESTED_TEMPLATE_DISPATCH macros */
#define NESTED_VARIANT_ARRAY_DISPATCH(p, i, ...) \
    NESTED_TEMPLATE_DISPATCH(teca_variant_array_impl, p, i, __VA_ARGS__)

#define NESTED_VARIANT_ARRAY_DISPATCH_FP(p, i, ...) \
    NESTED_TEMPLATE_DISPATCH_FP(teca_variant_array_impl, p, i, __VA_ARGS__)

#define NESTED_VARIANT_ARRAY_DISPATCH_I(p, i, ...) \
    NESTED_TEMPLATE_DISPATCH_I(teca_variant_array_impl, p, i, __VA_ARGS__)

#define NESTED_VARIANT_ARRAY_DISPATCH_CASE(nt, p, i, ...) \
    TEMPLATE_DISPATCH_CASE(teca_variant_array_impl, nt, p, i, __VA_ARGS__)

/** shortcuts for TEMPLATE_DISPATCH macros */
#define VARIANT_ARRAY_DISPATCH(p, ...) \
    TEMPLATE_DISPATCH(teca_variant_array_impl, p, __VA_ARGS__)

#define VARIANT_ARRAY_DISPATCH_FP(p, ...) \
    TEMPLATE_DISPATCH_FP(teca_variant_array_impl, p, __VA_ARGS__)

#define VARIANT_ARRAY_DISPATCH_I(p, ...) \
    TEMPLATE_DISPATCH_I(teca_variant_array_impl, p, __VA_ARGS__)

#define VARIANT_ARRAY_DISPATCH_FP_SI(p, ...) \
    TEMPLATE_DISPATCH_FP_SI(teca_variant_array_impl, p, __VA_ARGS__)

#define VARIANT_ARRAY_DISPATCH_CASE(nt, p, ...) \
    TEMPLATE_DISPATCH_CASE(teca_variant_array_impl, nt, p, __VA_ARGS__)

/// @cond
// tag for contiguous arrays, and objects that have
// overrides in teca_binary_stream
template<typename T>
struct pack_array
    : std::integral_constant<bool,
    std::is_arithmetic<T>::value ||
    std::is_same<T, std::string>::value>
{};

// tag for arrays of pointers of other objects
template<typename T>
struct pack_object_ptr
    : std::integral_constant<bool,
    (std::is_pointer<T>::value ||
    std::is_same<T, p_teca_variant_array>::value) &&
    !pack_array<T>::value>
{};

// tag for arrays of other objects
template<typename T>
struct pack_object
    : std::integral_constant<bool,
    !pack_array<T>::value &&
    !std::is_pointer<T>::value &&
    !pack_object_ptr<T>::value>
{};
/// @endcond


/** @brief
 * The concrete implementation of our type agnostic container for contiguous
 * arrays.
 */
template<typename T>
class TECA_EXPORT teca_variant_array_impl : public teca_variant_array
{
public:
    using element_type = T;
    using pointer_type = std::shared_ptr<T>;

    /** @name Array constructors
     * Constructs a new instance containing the templated type.
     */
    ///@{
    /** Allocate an array. The default value of alloc sets the array up for
     * use on the CPU using C++ new to allocate memory to hold the contents.
     */
    static std::shared_ptr<teca_variant_array_impl<T>>
    New() { return New(allocator::malloc); }

    /** Allocate an array. The default value of alloc sets the array up for
     * use on the CPU using C++ new to allocate memory to hold the contents.
     */
    static std::shared_ptr<teca_variant_array_impl<T>>
    New(allocator alloc);

    /** Allocates an array with space for n elements. The default value of
     * alloc sets the array up for use on the CPU using C++ new to allocate
     * memory to hold the contents.
     */
    static std::shared_ptr<teca_variant_array_impl<T>>
    New(size_t n) { return New(n, allocator::malloc); }

    /** Allocates an array with space for n elements. The default value of
     * alloc sets the array up for use on the CPU using C++ new to allocate
     * memory to hold the contents.
     */
    static std::shared_ptr<teca_variant_array_impl<T>>
    New(size_t n, allocator alloc);

    /** Allocate an array with space for n elements initialized with v.  The default
     * value of alloc sets the array up for use on the CPU using C++ new to
     * allocate memory to hold the contents.
     */
    template <typename U>
    static std::shared_ptr<teca_variant_array_impl<T>>
    New(size_t n, const U &v) { return New(n, v, allocator::malloc); }

    /** Allocate an array with space for n elements initialized with v.  The default
     * value of alloc sets the array up for use on the CPU using C++ new to
     * allocate memory to hold the contents.
     */
    template <typename U>
    static std::shared_ptr<teca_variant_array_impl<T>>
    New(size_t n, const U &v, allocator alloc);

    /** Allocate an array with space for n elements initialized with v.  The default
     * value of alloc sets the array up for use on the CPU using C++ new to
     * allocate memory to hold the contents.
     */
    template <typename U>
    static std::shared_ptr<teca_variant_array_impl<T>>
    New(size_t n, const U *v) { return New(n, v, allocator::malloc); }

    /** Allocate an array with space for n elements initialized with v.  The default
     * value of alloc sets the array up for use on the CPU using C++ new to
     * allocate memory to hold the contents.
     */
    template <typename U>
    static std::shared_ptr<teca_variant_array_impl<T>>
    New(size_t n, const U *v, allocator alloc);

    /** construct by directly providing the buffer contents. This can be used
     * for zero-copy transfer of data.  One must also name the allocator type
     * and device owning the data.  In addition for new allocations the
     * allocator type and owner are used internally to know how to
     * automatically move data during inter technology transfers.
     *
     * @param[in] alloc an ::allocator indicating the technology backing the
     *                  pointer
     * @param[in] size  the number of elements in the array pointed to by ptr
     * @param[in] owner the device owning the memory, -1 for CPU. if the
     *                  allocator is a GPU allocator and -1 is passed the
     *                  driver API is used to determine the device that
     *                  allocated the memory.
     * @param[in] ptr   a pointer to the data
     * @param[in] df    a function `void df(void*ptr)` used to delete the data
     *                  when this instance is finished using it.
     */
    template <typename delete_func_t>
    static std::shared_ptr<teca_variant_array_impl<T>>
    New(size_t n, T *ptr, allocator alloc, int owner, delete_func_t df);

    /// Returns a new instance initialized with a deep copy of this one.
    p_teca_variant_array new_copy(allocator alloc = allocator::malloc) const override;

    /** virtual copy construct. return a newly allocated object, initialized
     * copy from a subset of this.
     */
    p_teca_variant_array new_copy(size_t src_start, size_t n_elem, allocator alloc) const override;

    /// Returns a new instance of the same type.
    p_teca_variant_array new_instance(allocator alloc) const override;

    /// Returns a new instance of the same type sized to hold n elements.
    p_teca_variant_array new_instance(size_t n, allocator alloc) const override;
    ///@}

    virtual ~teca_variant_array_impl() noexcept;

    /// @copydoc teca_variant_array::set_allocator(allocator)
    int set_allocator(allocator alloc) override;

    /// Returns the name of the class in a human readable form
    std::string get_class_name() const override;

    /// Initialize all elements with T()
    void initialize() override
    {
        TECA_ERROR("Not implemented")
    }

    // silence some warning from nvcc
    using teca_variant_array::append;
    using teca_variant_array::assign;
    using teca_variant_array::copy;
    using teca_variant_array::set;
    using teca_variant_array::get;

    /** @name get
     * Copy the content of this array. The desitination must be large enough to
     * hold the results.  These calls could throw teca_bad_cast if the passed
     * in type is not castable to the internal type.
     */
    ///@{
    /// get a single value
    template<typename U>
    void get(size_t i, U &val) const
    {
        this->get(i, &val, 0, 1);
    }

    /// get a single value
    T get(size_t i) const
    {
        T val = T();
        this->get(i, &val, 0, 1);
        return val;
    }

    /// get a vector of values
    template<typename U>
    void get(std::vector<U> &dest) const
    {
        size_t n_elem = this->size();
        dest.resize(n_elem);
        this->get(0, dest.data(), 0, n_elem);
    }

    /// get a range of values
    template<typename U>
    void get(size_t src_start, U *dest, size_t dest_start, size_t n_elem) const
    {
        assert(this->size() >= (src_start + n_elem));
        this->get_dispatch(src_start, dest, dest_start, n_elem);
    }

    /// get the contents into the other array.
    void get(const p_teca_variant_array &dest) const override
    {
        this->get(0, dest, 0, this->size());
    }

    /** get a subset of the contents into a subset of the other array
     *
     * @param[in] dest_start the first location to assign to
     * @param[in] src        the array to copy values from
     * @param[in] src_start  the first location to copy from
     * @param[in] n_elem     the number of elements to copy
     */
    void get(size_t src_start, const p_teca_variant_array &dest,
        size_t dest_start, size_t n_elem) const override
    {
        assert(this->size() >= (src_start + n_elem));
        this->get_dispatch<T>(src_start, dest, dest_start, n_elem);
    }

    /// get the contents into the other array.
    template <typename U>
    void get(const p_teca_variant_array_impl<U> &dest) const
    {
        this->get(0, dest, 0, this->size());
    }

    /** get a subset of the contents into a subset of the other array
     *
     * @param[in] dest_start the first location to assign to
     * @param[in] src        the array to copy values from
     * @param[in] src_start  the first location to copy from
     * @param[in] n_elem     the number of elements to copy
     */
    template <typename U>
    void get(size_t src_start, const p_teca_variant_array_impl<U> &dest,
        size_t dest_start, size_t n_elem) const
    {
        assert(this->size() >= (src_start + n_elem));
        this->get_dispatch<T>(src_start, dest, dest_start, n_elem);
    }
    ///@}

    /** @name set
     * Assign values to this array. This array must already be large enough to
     * hold the result. If automatic resizing is desired use copy instead.
     * These calls could throw teca_bad_cast if the passed in type is not
     * castable to the internal type.
     */
    ///@{
    /// set a single value
    template<typename U>
    void set(size_t i, const U &val)
    {
        this->set(i, &val, 0, 1);
    }

    /// set from a vector of values
    template<typename U>
    void set(const std::vector<U> &src)
    {
        this->set(0, src.data(), 0, src.size());
    }

    /** set from a subset of the other array
     *
     * @param[in] dest_start the first location to assign to
     * @param[in] src        the array to copy values from
     * @param[in] src_start  the first location to copy from
     * @param[in] n_elem     the number of elements to copy
     */
    template<typename U>
    void set(size_t dest_start, const U *src, size_t src_start, size_t n_elem)
    {
        assert(this->size() >= (dest_start + n_elem));
        this->set_dispatch(dest_start, src, src_start, n_elem);
    }

    /// Set from the other array
    void set(const const_p_teca_variant_array &src) override
    {
        this->set(0, src, 0, src->size());
    }

    /** Set a subset of this array from a subset of the other array.
     *
     * @param[in] dest_start the first location to assign to
     * @param[in] src        the array to copy values from
     * @param[in] src_start  the first location to copy from
     * @param[in] n_elem     the number of elements to copy
     */
    void set(size_t dest_start, const const_p_teca_variant_array &src,
        size_t src_start, size_t n_elem) override
    {
        assert(this->size() >= (dest_start + n_elem));
        this->set_dispatch(dest_start, src, src_start, n_elem);
    }

    /// Set from the other array
    template <typename U>
    void set(const const_p_teca_variant_array_impl<U> &src)
    {
        this->set(0, src, 0, src->size());
    }

    /** Set a subset of this array from a subset of the other array.
     *
     * @param[in] dest_start the first location to assign to
     * @param[in] src        the array to copy values from
     * @param[in] src_start  the first location to copy from
     * @param[in] n_elem     the number of elements to copy
     */
    template <typename U>
    void set(size_t dest_start, const const_p_teca_variant_array_impl<U> &src,
        size_t src_start, size_t n_elem)
    {
        assert(this->size() >= (dest_start + n_elem));
        this->set_dispatch(dest_start, src, src_start, n_elem);
    }
    ///@}

    /** @name assign
     * assign the contents of the passed array to this array. This array will be
     * resized to hold the results.  These calls could throw teca_bad_cast if
     * the passed in type is not castable to the internal type.
     */
    ///@{
    /// assign the contents from a vector of values
    template<typename U>
    void assign(const std::vector<U> &src)
    {
        this->assign(src.data(), 0, src.size());
    }

    /// assign a subset from the other array
    template<typename U>
    void assign(const U *src, size_t src_start, size_t n_elem)
    {
        this->assign_dispatch(src, src_start, n_elem);
    }

    /// assign the contents from the other array.
    void assign(const const_p_teca_variant_array &src) override
    {
        this->assign(src, 0, src->size());
    }

    /// assign a subset of the other array
    void assign(const const_p_teca_variant_array &src,
        size_t src_start, size_t n_elem) override
    {
        this->assign_dispatch(src, src_start, n_elem);
    }

    /// assign the contents from the other array.
    template <typename U>
    void assign(const const_p_teca_variant_array_impl<U> &src)
    {
        this->assign(src, 0, src->size());
    }

    /// assign the contents from the other array.
    template <typename U>
    void assign(const p_teca_variant_array_impl<U> &src)
    {
        // forward to the const implementation
        this->assign(const_p_teca_variant_array_impl<U>(src), 0, src->size());
    }

    /// assign a subset of the other array
    template <typename U>
    void assign(const const_p_teca_variant_array_impl<U> &src,
        size_t src_start, size_t n_elem)
    {
        this->assign_dispatch(src, src_start, n_elem);
    }

    /// copy the contents from the other array.
    template <typename U>
    void copy(const const_p_teca_variant_array_impl<U> &src)
    {
        this->assign(src, 0, src->size());
    }

    /// copy a subset of the other array
    template <typename U>
    void copy(const const_p_teca_variant_array_impl<U> &src,
        size_t src_start, size_t n_elem)
    {
        this->assign_dispatch(src, src_start, n_elem);
    }
    ///@}

    /** @name append
     * Append data at the back of the array.  These calls could throw
     * teca_bad_cast if the passed in type is not castable to the internal
     * type.
     */
    ///@{
    /// append a single value
    template<typename U>
    void append(const U &val)
    {
        this->append(&val, 0, 1);
    }

    /// append a vector of values
    template<typename U>
    void append(const std::vector<U> &vals)
    {
        this->append(vals.data(), 0, vals.size());
    }

    /// append a range of values
    template<typename U>
    void append(const U *src, size_t src_start, size_t n_elem)
    {
        this->append_dispatch(src, src_start, n_elem);
    }

    // Append the contents from the other array
    void append(const const_p_teca_variant_array &src) override
    {
        this->append(src, 0, src->size());
    }

    // Append a subset of the contents from the other array
    void append(const const_p_teca_variant_array &src,
        size_t src_start, size_t n_elem) override
    {
        this->append_dispatch(src, src_start, n_elem);
    }

    // Append the contents from the other array
    template <typename U>
    void append(const const_p_teca_variant_array_impl<U> &src)
    {
        this->append(src, 0, src->size());
    }

    // Append the contents from the other array
    template <typename U>
    void append(const p_teca_variant_array_impl<U> &src)
    {
        // forward to const implementation
        this->append(const_p_teca_variant_array_impl<U>(src), 0, src->size());
    }

    // Append a subset of the contents from the other array
    template <typename U>
    void append(const const_p_teca_variant_array_impl<U> &src,
        size_t src_start, size_t n_elem)
    {
        this->append_dispatch(src, src_start, n_elem);
    }
    ///@}

#if !defined(SWIG)
    /** @name get_accessible
     * get's a pointer to the raw data that is accessible on the named
     * accelerator device or technology.
     */
    ///@{
    /// Get a pointer to the data accessible on the CPU
    const std::shared_ptr<const T> get_host_accessible() const
    { return m_data.get_host_accessible(); }

    /// Get a pointer to the data accessible within CUDA
    const std::shared_ptr<const T> get_cuda_accessible() const
    { return m_data.get_cuda_accessible(); }
    ///@}
#endif

    /// Sycnhronize the stream used for data movement
    void synchronize() const override { m_data.synchronize(); }

    /** direct access to the internal memory. Use this when you are certain
     * that the data is already accessible in the location where you will
     * access it to save the cost of the std::shared_ptr copy constructor.
     */
    const T *data() const { return m_data.data(); }

    /** direct access to the internal memory. Use this when you are certain
     * that the data is already accessible in the location where you will
     * access it to save the cost of the std::shared_ptr copy constructor.
     */
    T *data() { return m_data.data(); }

    /** direct access to the internal memory. Use this when you are certain
     * that the data is already accessible in the location where you will
     * access it.
     */
    const std::shared_ptr<T> &pointer() const { return m_data.pointer(); }

    /** direct access to the internal memory. Use this when you are certain
     * that the data is already accessible in the location where you will
     * access it.
     */
    std::shared_ptr<T> &pointer() { return m_data.pointer(); }

    /// returns true if the data is accessible from CUDA codes
    int cuda_accessible() const noexcept override { return m_data.cuda_accessible(); }

    /// returns true if the data is accessible from codes running on the CPU
    int host_accessible() const noexcept override { return m_data.host_accessible(); }

    /// Get the current size of the data
    unsigned long size() const noexcept override;

    /// Resize the data
    void resize(unsigned long n) override;
    void resize(unsigned long n, const T &val);

    /// Reserve space
    void reserve(unsigned long n) override;

    /// Clear the data
    void clear() noexcept override;

    /// virtual swap
    void swap(const p_teca_variant_array &other) override;

    /// virtual equivalence test
    bool equal(const const_p_teca_variant_array &other) const override;

    /// Serialize to the stream
    int to_stream(teca_binary_stream &s) const override
    {
        this->to_binary<T>(s);
        return 0;
    }

    /// Deserialize from the stream
    int from_stream(teca_binary_stream &s) override
    {
        this->from_binary<T>(s);
        return 0;
    }

    /// Serialize to the stream
    int to_stream(std::ostream &s) const override
    {
        this->to_ascii<T>(s);
        return 0;
    }

    /// Deserialize from the stream
    int from_stream(std::ostream &s) override
    {
        this->from_ascii<T>(s);
        return 0;
    }

    /// Print the contents of the buffer for debugging
    template <typename U = T>
    void debug_print(typename std::enable_if< std::is_arithmetic<U>::value >::type* = 0) const
    { m_data.print(); }

    /// Print the contents of the buffer for debugging
    template <typename U = T>
    void debug_print(typename std::enable_if<!std::is_arithmetic<U>::value>::type* = 0) const
    {
        TECA_WARNING("Failed to print the buffer for T=" << typeid(T).name() << sizeof(T))
    }

    teca_variant_array::allocator get_allocator() const
    {
        return teca_variant_array::allocator(m_data.get_allocator());
    }

#if defined(SWIG)
protected:
#else
public:
#endif
    // NOTE: constructors are public to enable std::make_shared. DO NOT USE.

    /// default construct (the object is unusable)
    teca_variant_array_impl() {}

    /// construct with a specific allocator
    teca_variant_array_impl(allocator alloc) :
        m_data(alloc) {}

    /// construct with preallocated size
    teca_variant_array_impl(allocator alloc, size_t n_elem) :
        m_data(alloc, n_elem) {}

    /// construct with preallocated size and initialized to a specific value
    teca_variant_array_impl(allocator alloc, size_t n_elem, const T &val) :
        m_data(alloc, n_elem, val) {}

    /// construct with preallocated size and initialized to a specific value
    template <typename U>
    teca_variant_array_impl(allocator alloc, size_t n_elem, const U *vals) :
        m_data(alloc, n_elem, vals) {}

    /// copy construct from an instance of different type
    template<typename U>
    teca_variant_array_impl(allocator alloc,
        const const_p_teca_variant_array_impl<U> &other) :
            m_data(alloc, other->m_data) {}

    /// zero-copy construct by setting buffer contents directly
    template <typename delete_func_t>
    teca_variant_array_impl(allocator alloc, size_t size, int owner,
        T *ptr, delete_func_t df) : m_data(alloc, size, owner, ptr, df) {}

private:
    /// get from objects.
    template <typename U = T>
    void get_dispatch(size_t src_start,
        const p_teca_variant_array_impl<U> &dest, size_t dest_start, size_t n_elem,
        typename std::enable_if<object_dispatch<U>::value, U>::type* = 0) const;

    /// get from POD types
    template <typename U = T>
    void get_dispatch(size_t src_start,
        const p_teca_variant_array_impl<U> &dest, size_t dest_start, size_t n_elem,
        typename std::enable_if<pod_dispatch<U>::value, U>::type* = 0) const;

    /// set from objects.
    template <typename U = T>
    void set_dispatch(size_t dest_start,
        const const_p_teca_variant_array_impl<U> &src, size_t src_start, size_t n_elem,
        typename std::enable_if<object_dispatch<U>::value, U>::type* = 0);

    /// set from POD types
    template <typename U = T>
    void set_dispatch(size_t dest_start,
        const const_p_teca_variant_array_impl<U> &src, size_t src_start, size_t n_elem,
        typename std::enable_if<pod_dispatch<U>::value, U>::type* = 0);

    /// copy from objects.
    template <typename U = T>
    void assign_dispatch(const const_p_teca_variant_array_impl<U> &src,
        size_t src_start, size_t n_elem,
        typename std::enable_if<object_dispatch<U>::value, U>::type* = 0);

    /// copy from POD types
    template <typename U = T>
    void assign_dispatch(const const_p_teca_variant_array_impl<U> &src,
        size_t src_start, size_t n_elem,
        typename std::enable_if<pod_dispatch<U>::value, U>::type* = 0);

    /// append from objects.
    template <typename U = T>
    void append_dispatch(const const_p_teca_variant_array_impl<U> &src,
        size_t src_start, size_t n_elem,
        typename std::enable_if<object_dispatch<U>::value, U>::type* = 0);

    /// append from POD types
    template <typename U = T>
    void append_dispatch(const const_p_teca_variant_array_impl<U> &src,
        size_t src_start, size_t n_elem,
        typename std::enable_if<pod_dispatch<U>::value, U>::type* = 0);

    /// get from objects.
    template <typename U = T>
    void get_dispatch(size_t src_start,
        const p_teca_variant_array &dest, size_t dest_start, size_t n_elem,
        typename std::enable_if<object_dispatch<U>::value, U>::type* = 0) const;

    /// get from POD types
    template <typename U = T>
    void get_dispatch(size_t src_start,
        const p_teca_variant_array &dest, size_t dest_start, size_t n_elem,
        typename std::enable_if<pod_dispatch<U>::value, U>::type* = 0) const;

    /// set from objects.
    template <typename U = T>
    void set_dispatch(size_t dest_start,
        const const_p_teca_variant_array &src, size_t src_start, size_t n_elem,
        typename std::enable_if<object_dispatch<U>::value, U>::type* = 0);

    /// set from POD types
    template <typename U = T>
    void set_dispatch(size_t dest_start,
        const const_p_teca_variant_array &src, size_t src_start, size_t n_elem,
        typename std::enable_if<pod_dispatch<U>::value, U>::type* = 0);

    /// copy from objects.
    template <typename U = T>
    void assign_dispatch(const const_p_teca_variant_array &src,
        size_t src_start, size_t n_elem,
        typename std::enable_if<object_dispatch<U>::value, U>::type* = 0);

    /// copy from POD types
    template <typename U = T>
    void assign_dispatch(const const_p_teca_variant_array &src,
        size_t src_start, size_t n_elem,
        typename std::enable_if<pod_dispatch<U>::value, U>::type* = 0);

    /// append from objects.
    template <typename U = T>
    void append_dispatch(const const_p_teca_variant_array &src,
        size_t src_start, size_t n_elem,
        typename std::enable_if<object_dispatch<U>::value, U>::type* = 0);

    /// append from POD types
    template <typename U = T>
    void append_dispatch(const const_p_teca_variant_array &src,
        size_t src_start, size_t n_elem,
        typename std::enable_if<pod_dispatch<U>::value, U>::type* = 0);

    /// get from objects.
    template <typename U = T>
    void get_dispatch(size_t src_start,
        U *dest, size_t dest_start, size_t n_elem,
        typename std::enable_if<object_dispatch<U>::value, U>::type* = 0) const;

    /// get from POD types
    template <typename U = T>
    void get_dispatch(size_t src_start,
        U *dest, size_t dest_start, size_t n_elem,
        typename std::enable_if<pod_dispatch<U>::value, U>::type* = 0) const;

    /// set from objects.
    template <typename U = T>
    void set_dispatch(size_t dest_start,
        const U *src, size_t src_start, size_t n_elem,
        typename std::enable_if<object_dispatch<U>::value, U>::type* = 0);

    /// set from POD types
    template <typename U = T>
    void set_dispatch(size_t dest_start,
        const U *src, size_t src_start, size_t n_elem,
        typename std::enable_if<pod_dispatch<U>::value, U>::type* = 0);

    /// copy from objects.
    template <typename U = T>
    void assign_dispatch(const U *src,
        size_t src_start, size_t n_elem,
        typename std::enable_if<object_dispatch<U>::value, U>::type* = 0);

    /// copy from POD types
    template <typename U = T>
    void assign_dispatch(const U *src,
        size_t src_start, size_t n_elem,
        typename std::enable_if<pod_dispatch<U>::value, U>::type* = 0);

    /// append from objects.
    template <typename U = T>
    void append_dispatch(const U *src,
        size_t src_start, size_t n_elem,
        typename std::enable_if<object_dispatch<U>::value, U>::type* = 0);

    /// append from POD types
    template <typename U = T>
    void append_dispatch(const U *src,
        size_t src_start, size_t n_elem,
        typename std::enable_if<pod_dispatch<U>::value, U>::type* = 0);


    // tag dispatch c style array, and types that have overrides in
    // binary stream
    template <typename U = T>
    void to_binary(teca_binary_stream &s,
        typename std::enable_if<pack_array<U>::value, U>::type* = 0)
        const;

    template <typename U = T>
    void from_binary(teca_binary_stream &s,
        typename std::enable_if<pack_array<U>::value, U>::type* = 0);

    // tag dispatch array of other objects
    template <typename U = T>
    void to_binary(teca_binary_stream &s,
        typename std::enable_if<pack_object<U>::value, U>::type* = 0)
        const;

    template <typename U = T>
    void from_binary(teca_binary_stream &s,
        typename std::enable_if<pack_object<U>::value, U>::type* = 0);

    // tag dispatch array of pointer to other objects
    template <typename U = T>
    void to_binary(teca_binary_stream &s,
        typename std::enable_if<pack_object_ptr<U>::value, U>::type* = 0)
        const;

    template <typename U = T>
    void from_binary(teca_binary_stream &s,
        typename std::enable_if<pack_object_ptr<U>::value, U>::type* = 0);

    // ostream
    template <typename U = T>
    void to_ascii(std::ostream &s,
        typename std::enable_if<pack_array<U>::value, U>::type* = 0)
        const;

    template <typename U = T>
    void from_ascii(std::ostream &s,
        typename std::enable_if<pack_array<U>::value, U>::type* = 0);

    // tag dispatch array of other objects
    template <typename U = T>
    void to_ascii(std::ostream &s,
        typename std::enable_if<pack_object<U>::value, U>::type* = 0)
        const;

    template <typename U = T>
    void from_ascii(std::ostream &s,
        typename std::enable_if<pack_object<U>::value, U>::type* = 0);

    // tag dispatch array of pointer to other objects
    template <typename U = T>
    void to_ascii(std::ostream &s,
        typename std::enable_if<pack_object_ptr<U>::value, U>::type* = 0)
        const;

    template <typename U = T>
    void from_ascii(std::ostream &s,
        typename std::enable_if<pack_object_ptr<U>::value, U>::type* = 0);

    /// returns a code used to identify the contained type during serialization
    unsigned int type_code() const noexcept override;

    /// gets a shared pointer to this
    p_teca_variant_array_impl<T> shared_from_this()
    {
        return std::static_pointer_cast
            <teca_variant_array_impl<T>>
                (teca_variant_array::shared_from_this());
    }

    /// gets a const shared pointer to this
    const_p_teca_variant_array_impl<T> shared_from_this() const
    {
        return std::static_pointer_cast
            <const teca_variant_array_impl<T>>
                (teca_variant_array::shared_from_this());
    }

private:
    hamr::buffer<T> m_data;

    friend class teca_variant_array;
    template<typename U> friend class teca_variant_array_impl;
};




#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::get(unsigned long i, T &val) const
{
    this->get_dispatch(i, val);
}

// --------------------------------------------------------------------------
template<typename T>
T teca_variant_array::get(unsigned long i) const
{
    T val = T();
    this->get_dispatch(i, val);
    return val;
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::get(std::vector<T> &vals) const
{
    this->get_dispatch(vals);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::get(size_t src_start, T *dest, size_t dest_start, size_t n_elem) const
{
    this->get_dispatch(src_start, dest, dest_start, n_elem);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::set(unsigned long i, const T &val)
{
    this->set_dispatch(i, val);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::set(const std::vector<T> &src)
{
    this->set_dispatch(src);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::set(size_t dest_start, const T *src, size_t src_start, size_t n_elem)
{
    this->set_dispatch(dest_start, src, src_start, n_elem);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::assign(const std::vector<T> &src)
{
    this->assign_dispatch(src);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::assign(const T *src, size_t src_start, size_t n_elem)
{
    this->assign(src, src_start, n_elem);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::append(const T &val)
{
    this->append_dispatch(val);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::append(const std::vector<T> &src)
{
    this->append_dispatch(src);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::append(const T *src, size_t src_start, size_t n_elem)
{
    this->append_dispatch(src, src_start, n_elem);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::get_dispatch(unsigned long i, T &val,
    typename std::enable_if<object_dispatch<T>::value, T>::type*) const
{
    // only apply when types match
    const teca_variant_array_impl<T> *ptthis =
        dynamic_cast<const teca_variant_array_impl<T>*>(this);

    if (ptthis)
    {
        // safe. the types match
        ptthis->get(i, val);
        return;
    }

    // types do not match
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() << sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::get_dispatch(unsigned long i, T &val,
    typename std::enable_if<pod_dispatch<T>::value, T>::type*) const
{
    // apply on POD types
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        this,
        const TT *ptthis = dynamic_cast<const TT*>(this);
        ptthis->get(i, val);
        return;
        )

    // unssuported type
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() << sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::get_dispatch(std::vector<T> &vals,
    typename std::enable_if<object_dispatch<T>::value, T>::type*) const
{
    // only apply when types match
    const teca_variant_array_impl<T> *ptthis =
        dynamic_cast<const teca_variant_array_impl<T>*>(this);

    if (ptthis)
    {
        // safe. the types match
        ptthis->get(vals);
        return;
    }

    // types do not match
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() << sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::get_dispatch(std::vector<T> &vals,
    typename std::enable_if<pod_dispatch<T>::value, T>::type*) const
{
    // apply on POD types
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        this,
        const TT *ptthis = dynamic_cast<const TT*>(this);
        ptthis->get(vals);
        return;
        )

    // unssuported type
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() << sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::get_dispatch(size_t src_start, T *dest, size_t dest_start, size_t n_elem,
    typename std::enable_if<object_dispatch<T>::value, T>::type*) const
{
    // only apply when types match
    const teca_variant_array_impl<T> *ptthis =
        dynamic_cast<const teca_variant_array_impl<T>*>(this);

    if (ptthis)
    {
        // safe. the types match
        ptthis->get(src_start, dest, dest_start, n_elem);
        return;
    }

    // types do not match
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() << sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::get_dispatch(size_t src_start, T *dest, size_t dest_start, size_t n_elem,
    typename std::enable_if<pod_dispatch<T>::value, T>::type*) const
{
    // apply on POD types
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        this,
        const TT *ptthis = dynamic_cast<const TT*>(this);
        ptthis->get(src_start, dest, dest_start, n_elem);
        return;
        )

    // unssuported type
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() << sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::set_dispatch(unsigned long i, const T &val,
    typename std::enable_if<object_dispatch<T>::value, T>::type*)
{
    // only apply when types match
    teca_variant_array_impl<T> *ptthis =
        dynamic_cast<teca_variant_array_impl<T>*>(this);

    if (ptthis)
    {
        // safe. the types match
        ptthis->set(i, val);
        return;
    }

    // types do not match
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() << sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::set_dispatch(unsigned long i, const T &val,
    typename std::enable_if<pod_dispatch<T>::value, T>::type*)
{
    // apply on POD types
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        this,
        TT *ptthis = dynamic_cast<TT*>(this);
        ptthis->set(i, val);
        return;
        )

    // unssuported type
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() << sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::set_dispatch(const std::vector<T> &src,
    typename std::enable_if<object_dispatch<T>::value, T>::type*)
{
    // only apply when types match
    teca_variant_array_impl<T> *ptthis =
        dynamic_cast<teca_variant_array_impl<T>*>(this);

    if (ptthis)
    {
        // safe. the types match
        ptthis->set(src);
        return;
    }

    // types do not match
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() <<sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::set_dispatch(const std::vector<T> &src,
    typename std::enable_if<pod_dispatch<T>::value, T>::type*)
{
    // apply on POD types
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        this,
        TT *ptthis = dynamic_cast<TT*>(this);
        ptthis->set(src);
        return;
        )

    // unssuported type
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() << sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::set_dispatch(size_t dest_start,
    const T *src, size_t src_start, size_t n_elem,
    typename std::enable_if<object_dispatch<T>::value, T>::type*)
{
    // only apply when types match
    teca_variant_array_impl<T> *ptthis =
        dynamic_cast<teca_variant_array_impl<T>*>(this);

    if (ptthis)
    {
        // safe. the types match
        ptthis->set(dest_start, src, src_start, n_elem);
        return;
    }

    // types do not match
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() << sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::set_dispatch(size_t dest_start,
    const T *src, size_t src_start, size_t n_elem,
    typename std::enable_if<pod_dispatch<T>::value, T>::type*)
{
    // apply on POD types
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        this,
        TT *ptthis = dynamic_cast<TT*>(this);
        ptthis->set(dest_start, src, src_start, n_elem);
        return;
        )

    // unssuported type
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() << sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::assign_dispatch(const std::vector<T> &src,
    typename std::enable_if<object_dispatch<T>::value, T>::type*)
{
    // only apply when types match
    teca_variant_array_impl<T> *ptthis =
        dynamic_cast<teca_variant_array_impl<T>*>(this);

    if (ptthis)
    {
        // safe. the types match
        ptthis->assign(src);
        return;
    }

    // types do not match
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() << sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::assign_dispatch(const std::vector<T> &src,
    typename std::enable_if<pod_dispatch<T>::value, T>::type*)
{
    // apply on POD types
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        this,
        TT *ptthis = dynamic_cast<TT*>(this);
        ptthis->assign(src);
        return;
        )

    // unssuported type
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() << sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::assign_dispatch(const T *src, size_t src_start,
    size_t n_elem, typename std::enable_if<object_dispatch<T>::value, T>::type*)
{
    // only apply when types match
    teca_variant_array_impl<T> *ptthis =
        dynamic_cast<teca_variant_array_impl<T>*>(this);

    if (ptthis)
    {
        // safe. the types match
        ptthis->assign(src, src_start, n_elem);
        return;
    }

    // types do not match
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() << sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::assign_dispatch(const T *src, size_t src_start, size_t n_elem,
    typename std::enable_if<pod_dispatch<T>::value, T>::type*)
{
    // apply on POD types
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        this,
        TT *ptthis = dynamic_cast<TT*>(this);
        ptthis->assign(src, src_start, n_elem);
        return;
        )

    // unssuported type
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() << sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::append_dispatch(const T &val,
    typename std::enable_if<object_dispatch<T>::value, T>::type*)
{
    // only apply when types match
    teca_variant_array_impl<T> *ptthis =
        dynamic_cast<teca_variant_array_impl<T>*>(this);

    if (ptthis)
    {
        // safe. the types match
        ptthis->append(val);
        return;
    }

    // types do not match
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() << sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::append_dispatch(const T &val,
    typename std::enable_if<pod_dispatch<T>::value, T>::type*)
{
    // apply on POD types
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        this,
        TT *ptthis = dynamic_cast<TT*>(this);
        ptthis->append(val);
        return;
        )

    // unssuported type
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() <<sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::append_dispatch(const std::vector<T> &src,
    typename std::enable_if<object_dispatch<T>::value, T>::type*)
{
    // only apply when types match
    teca_variant_array_impl<T> *ptthis =
        dynamic_cast<teca_variant_array_impl<T>*>(this);

    if (ptthis)
    {
        // safe. the types match
        ptthis->append(src);
        return;
    }

    // types do not match
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() << sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::append_dispatch(const std::vector<T> &src,
    typename std::enable_if<pod_dispatch<T>::value, T>::type*)
{
    // apply on POD types
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        this,
        TT *ptthis = dynamic_cast<TT*>(this);
        ptthis->append(src);
        return;
        )

    // unssuported type
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() << sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::append_dispatch(const T *src, size_t src_start, size_t n_elem,
    typename std::enable_if<object_dispatch<T>::value, T>::type*)
{
    // only apply when types match
    teca_variant_array_impl<T> *ptthis =
        dynamic_cast<teca_variant_array_impl<T>*>(this);

    if (ptthis)
    {
        // safe. the types match
        ptthis->append(src, src_start, n_elem);
        return;
    }

    // types do not match
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() << sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::append_dispatch(const T *src, size_t src_start, size_t n_elem,
    typename std::enable_if<pod_dispatch<T>::value, T>::type*)
{
    // apply on POD types
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        this,
        TT *ptthis = dynamic_cast<TT*>(this);
        ptthis->append(src, src_start, n_elem);
        return;
        )

    // unssuported type
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << typeid(T).name() <<sizeof(T) << " to " << this->get_class_name()
        << " failed")
}




// --------------------------------------------------------------------------
template<typename T>
teca_variant_array_impl<T>::~teca_variant_array_impl() noexcept
{
    this->clear();
}

// --------------------------------------------------------------------------
template<typename T>
p_teca_variant_array_impl<T> teca_variant_array_impl<T>::New(allocator alloc)
{
    return std::make_shared<teca_variant_array_impl<T>>(alloc);
}

// --------------------------------------------------------------------------
template<typename T>
p_teca_variant_array_impl<T> teca_variant_array_impl<T>::New(size_t n, allocator alloc)
{
    return std::make_shared<teca_variant_array_impl<T>>(alloc, n);
}

// --------------------------------------------------------------------------
template<typename T>
template<typename U>
p_teca_variant_array_impl<T> teca_variant_array_impl<T>::New(size_t n,
    const U &v, allocator alloc)
{
    return std::make_shared<teca_variant_array_impl<T>>(alloc, n, v);
}

// --------------------------------------------------------------------------
template<typename T>
template<typename U>
p_teca_variant_array_impl<T> teca_variant_array_impl<T>::New(size_t n,
    const U *v, allocator alloc)
{
    return std::make_shared<teca_variant_array_impl<T>>(alloc, n, v);
}

// --------------------------------------------------------------------------
template<typename T>
template <typename delete_func_t>
p_teca_variant_array_impl<T> teca_variant_array_impl<T>::New(size_t n,
    T *ptr, allocator alloc, int owner, delete_func_t df)
{
    return std::make_shared<teca_variant_array_impl<T>>
        (alloc, n, owner, ptr, df);
}

// --------------------------------------------------------------------------
template<typename T>
p_teca_variant_array teca_variant_array_impl<T>::new_copy(allocator alloc) const
{
    if (alloc == allocator::same)
        alloc = static_cast<allocator>(m_data.get_allocator());

    return std::make_shared<teca_variant_array_impl<T>>
        (alloc, this->shared_from_this());
}

// --------------------------------------------------------------------------
template<typename T>
p_teca_variant_array teca_variant_array_impl<T>::new_copy(size_t src_start,
    size_t n_elem, allocator alloc) const
{
    if (alloc == allocator::same)
        alloc = static_cast<allocator>(m_data.get_allocator());

    p_teca_variant_array_impl<T> dest =
        std::make_shared<teca_variant_array_impl<T>>
            (alloc, n_elem);

    this->get(src_start, dest, 0, n_elem);

    return dest;
}

// --------------------------------------------------------------------------
template<typename T>
p_teca_variant_array teca_variant_array_impl<T>::new_instance(allocator alloc) const
{
    if (alloc == allocator::same)
        alloc = static_cast<allocator>(m_data.get_allocator());

    return std::make_shared<teca_variant_array_impl<T>>(alloc);
}

// --------------------------------------------------------------------------
template<typename T>
p_teca_variant_array teca_variant_array_impl<T>::new_instance(size_t n,
    allocator alloc) const
{
    if (alloc == allocator::same)
        alloc = static_cast<allocator>(m_data.get_allocator());

    return std::make_shared<teca_variant_array_impl<T>>(alloc, n);
}

// --------------------------------------------------------------------------
template<typename T>
int teca_variant_array_impl<T>::set_allocator(allocator alloc)
{
    // if the allocator is already in use do nothing
    if (this->m_data.get_allocator() == alloc)
        return 0;

    // move the data using the specified allocator
    hamr::buffer<T> tmp(alloc, m_data);
    m_data = std::move(tmp);

    return 0;
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array_impl<T>::swap(const p_teca_variant_array &other)
{
    p_teca_variant_array_impl<T> pt_other =
        std::dynamic_pointer_cast<teca_variant_array_impl<T>>(other);

    if (pt_other)
    {
        m_data.swap(pt_other->m_data);
    }

    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << other->get_class_name() << sizeof(T) << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template<typename T>
std::string teca_variant_array_impl<T>::get_class_name() const
{
    const char *element_name = typeid(T).name();
    size_t element_size = sizeof(T);
    std::ostringstream oss;
    oss << "teca_variant_array_impl<" << element_name
        << element_size << ">";
    return oss.str();
}

// --------------------------------------------------------------------------
template<typename T>
unsigned long teca_variant_array_impl<T>::size() const noexcept
{
    return m_data.size();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array_impl<T>::resize(unsigned long n)
{
    m_data.resize(n);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array_impl<T>::resize(unsigned long n, const T &val)
{
    m_data.resize(n, val);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array_impl<T>::reserve(unsigned long n)
{
    m_data.reserve(n);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array_impl<T>::clear() noexcept
{
    m_data.free();
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::get_dispatch(size_t src_start,
    const p_teca_variant_array_impl<U> &dest, size_t dest_start, size_t n_elem,
    typename std::enable_if<object_dispatch<U>::value, U>::type*) const
{
    // only act on arrays of the same type
    p_teca_variant_array_impl<T> tp_dest =
        std::dynamic_pointer_cast<teca_variant_array_impl<T>>(dest);

    if (tp_dest)
    {
        m_data.get(src_start, tp_dest->m_data, dest_start, n_elem);
        return;
    }

    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << dest->get_class_name() << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::get_dispatch(size_t src_start,
    const p_teca_variant_array_impl<U> &dest, size_t dest_start, size_t n_elem,
    typename std::enable_if<pod_dispatch<U>::value, U>::type*) const
{
    assert(dest->size() >= dest_start + n_elem);
    assert(this->size() >= src_start + n_elem);
    m_data.get(src_start, dest->m_data, dest_start, n_elem);
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::set_dispatch(size_t dest_start,
    const const_p_teca_variant_array_impl<U> &src, size_t src_start, size_t n_elem,
    typename std::enable_if<object_dispatch<U>::value, U>::type*)
{
    // only act on arrays of the same type
    const_p_teca_variant_array_impl<T> tp_src =
        std::dynamic_pointer_cast<const teca_variant_array_impl<T>>(src);

    if (tp_src)
    {
        m_data.set(dest_start, tp_src->m_data, src_start, n_elem);

        return;
    }

    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << src->get_class_name() << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::set_dispatch(size_t dest_start,
    const const_p_teca_variant_array_impl<U> &src, size_t src_start, size_t n_elem,
    typename std::enable_if<pod_dispatch<U>::value, U>::type*)
{
    m_data.set(dest_start, src->m_data, src_start, n_elem);
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::assign_dispatch(
    const const_p_teca_variant_array_impl<U> &src, size_t src_start, size_t n_elem,
    typename std::enable_if<object_dispatch<U>::value, U>::type*)
{
    // only act on arrays of the same type
    const_p_teca_variant_array_impl<T> tp_src =
        std::dynamic_pointer_cast<const teca_variant_array_impl<T>>(src);

    if (tp_src)
    {
        m_data.assign(tp_src->m_data, src_start, n_elem);
        return;
    }

    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << src->get_class_name() << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::assign_dispatch(
    const const_p_teca_variant_array_impl<U> &src, size_t src_start, size_t n_elem,
    typename std::enable_if<pod_dispatch<U>::value, U>::type*)
{
    m_data.assign(src->m_data, src_start, n_elem);
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::append_dispatch(
    const const_p_teca_variant_array_impl<U> &src, size_t src_start, size_t n_elem,
    typename std::enable_if<object_dispatch<U>::value, U>::type*)
{
    // only act on arrays of the same type
    const_p_teca_variant_array_impl<T> tp_src =
        std::dynamic_pointer_cast<const teca_variant_array_impl<T>>(src);

    if (tp_src)
    {
        m_data.append(tp_src->m_data, src_start, n_elem);
        return;
    }

    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << src->get_class_name() << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::append_dispatch(
    const const_p_teca_variant_array_impl<U> &src, size_t src_start, size_t n_elem,
    typename std::enable_if<pod_dispatch<U>::value, U>::type*)
{
    m_data.append(src->m_data, src_start, n_elem);
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::get_dispatch(size_t src_start,
    const p_teca_variant_array &dest, size_t dest_start, size_t n_elem,
    typename std::enable_if<object_dispatch<U>::value, U>::type*) const
{
    // only act on arrays of the same type
    p_teca_variant_array_impl<T> tp_dest =
        std::dynamic_pointer_cast<teca_variant_array_impl<T>>(dest);

    if (tp_dest)
    {
        this->get_dispatch(src_start, tp_dest, dest_start, n_elem);
        return;
    }

    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << dest->get_class_name() << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::get_dispatch(size_t src_start,
    const p_teca_variant_array &dest, size_t dest_start, size_t n_elem,
    typename std::enable_if<pod_dispatch<U>::value, U>::type*) const
{
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        dest.get(),
        std::shared_ptr<TT> tp_dest = std::static_pointer_cast<TT>(dest);
        this->get_dispatch(src_start, tp_dest, dest_start, n_elem);
        return;
        )

    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << dest->get_class_name() << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::set_dispatch(size_t dest_start,
    const const_p_teca_variant_array &src, size_t src_start, size_t n_elem,
    typename std::enable_if<object_dispatch<U>::value, U>::type*)
{
    // only act on arrays of the same type
    const_p_teca_variant_array_impl<T> tp_src =
        std::dynamic_pointer_cast<const teca_variant_array_impl<T>>(src);

    if (tp_src)
    {
        this->set_dispatch(dest_start, tp_src, src_start, n_elem);
        return;
    }

    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << src->get_class_name() << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::set_dispatch(size_t dest_start,
    const const_p_teca_variant_array &src, size_t src_start, size_t n_elem,
    typename std::enable_if<pod_dispatch<U>::value, U>::type*)
{
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        src.get(),
        const_p_teca_variant_array_impl<NT> tp_src =
            std::static_pointer_cast<const teca_variant_array_impl<NT>>(src);
        //std::shared_ptr<TT> tp_src = std::static_pointer_cast<TT>(src);
        this->set_dispatch(dest_start, tp_src, src_start, n_elem);
        return;
        )

    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << src->get_class_name() << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::assign_dispatch(
    const const_p_teca_variant_array &src, size_t src_start, size_t n_elem,
    typename std::enable_if<object_dispatch<U>::value, U>::type*)
{
    // only act on arrays of the same type
    const_p_teca_variant_array_impl<T> tp_src =
        std::dynamic_pointer_cast<const teca_variant_array_impl<T>>(src);

    if (tp_src)
    {
        this->assign_dispatch(tp_src, src_start, n_elem);
        return;
    }

    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << src->get_class_name() << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::assign_dispatch(
    const const_p_teca_variant_array &src, size_t src_start, size_t n_elem,
    typename std::enable_if<pod_dispatch<U>::value, U>::type*)
{
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        src.get(),
        std::shared_ptr<const TT> tp_src = std::static_pointer_cast<const TT>(src);
        this->assign_dispatch(tp_src, src_start, n_elem);
        return;
        )

    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << src->get_class_name() << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::append_dispatch(
    const const_p_teca_variant_array &src, size_t src_start, size_t n_elem,
    typename std::enable_if<object_dispatch<U>::value, U>::type*)
{
    // only act on arrays of the same type
    const_p_teca_variant_array_impl<T> tp_src =
        std::dynamic_pointer_cast<const teca_variant_array_impl<T>>(src);

    if (tp_src)
    {
        this->append_dispatch(tp_src, src_start, n_elem);
        return;
    }

    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << src->get_class_name() << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::append_dispatch(
    const const_p_teca_variant_array &src, size_t src_start, size_t n_elem,
    typename std::enable_if<pod_dispatch<U>::value, U>::type*)
{
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        src.get(),
        std::shared_ptr<const TT> tp_src = std::static_pointer_cast<const TT>(src);
        this->append_dispatch(tp_src, src_start, n_elem);
        return;
        )

    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << src->get_class_name() << " to " << this->get_class_name()
        << " failed")
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::get_dispatch(size_t src_start,
    U *dest, size_t dest_start, size_t n_elem,
    typename std::enable_if<object_dispatch<U>::value, U>::type*) const
{
    // only act on arrays of the same type
    const teca_variant_array_impl<U> *thisu =
        dynamic_cast<const teca_variant_array_impl<U>*>(this);

    if (thisu)
    {
        thisu->m_data.get(src_start, dest, dest_start, n_elem);
        return;
    }

    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << safe_class_name(this) << " to " << safe_pointer_name(dest)
        << " failed")
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::get_dispatch(size_t src_start,
    U *dest, size_t dest_start, size_t n_elem,
    typename std::enable_if<pod_dispatch<U>::value, U>::type*) const
{
    m_data.get(src_start, dest, dest_start, n_elem);
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::set_dispatch(size_t dest_start,
    const U *src, size_t src_start, size_t n_elem,
    typename std::enable_if<object_dispatch<U>::value, U>::type*)
{
    // only act on arrays of the same type
    teca_variant_array_impl<U> *thisu =
        dynamic_cast<teca_variant_array_impl<U>*>(this);

    if (thisu)
    {
        thisu->m_data.set(dest_start, src, src_start, n_elem);
        return;
    }

    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << safe_class_name(this) << " to " << safe_pointer_name(src)
        << " failed")
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::set_dispatch(size_t dest_start,
    const U *src, size_t src_start, size_t n_elem,
    typename std::enable_if<pod_dispatch<U>::value, U>::type*)
{
    m_data.set(dest_start, src, src_start, n_elem);
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::assign_dispatch(
    const U *src, size_t src_start, size_t n_elem,
    typename std::enable_if<object_dispatch<U>::value, U>::type*)
{
    // only act on arrays of the same type
    teca_variant_array_impl<U> *thisu =
        dynamic_cast<teca_variant_array_impl<U>*>(this);

    if (thisu)
    {
        thisu->m_data.assign(src, src_start, n_elem);
        return;
    }

    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << safe_class_name(this) << " to " << safe_pointer_name(src)
        << " failed")
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::assign_dispatch(
    const U *src, size_t src_start, size_t n_elem,
    typename std::enable_if<pod_dispatch<U>::value, U>::type*)
{
    m_data.assign(src, src_start, n_elem);
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::append_dispatch(
    const U *src, size_t src_start, size_t n_elem,
    typename std::enable_if<object_dispatch<U>::value, U>::type*)
{
    // only act on arrays of the same type
    teca_variant_array_impl<U> *thisu =
        dynamic_cast<teca_variant_array_impl<U>*>(this);

    if (thisu)
    {
        thisu->m_data.append(src, src_start, n_elem);
        return;
    }

    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << safe_class_name(this) << " to " << safe_pointer_name(src)
        << " failed")
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void teca_variant_array_impl<T>::append_dispatch(
    const U *src, size_t src_start, size_t n_elem,
    typename std::enable_if<pod_dispatch<U>::value, U>::type*)
{
    m_data.append(src, src_start, n_elem);
}




// --------------------------------------------------------------------------
template<typename T>
bool teca_variant_array_impl<T>::equal(const const_p_teca_variant_array &other) const
{
    using NT = T;
    using TT = teca_variant_array_impl<T>;
    const TT *other_t = dynamic_cast<const TT*>(other.get());
    if (other_t)
    {
        size_t n_elem = this->size();

        if (n_elem != other_t->size())
            return false;

        auto spthis = this->get_host_accessible();
        const NT *pthis = spthis.get();

        auto spother = other_t->get_host_accessible();
        const NT *pother = spother.get();

        sync_host_access_any(this, other);

        for (size_t i = 0; i < n_elem; ++i)
        {
            if (pthis[i] != pother[i])
                return false;
        }

        return true;
    }
    TECA_FATAL_ERROR("Operation on incompatible types. The cast from "
        << safe_class_name(other.get()) << " to " << safe_class_name(this)
        << " failed")
    return false;
}

// --------------------------------------------------------------------------
template<typename T>
    template <typename U>
void teca_variant_array_impl<T>::to_binary(teca_binary_stream &s,
    typename std::enable_if<pack_array<U>::value, U>::type*) const
{
    // access the data on the host
    std::shared_ptr<const T> pdata = this->get_host_accessible();
    sync_host_access_any(this);

    // pack the size
    unsigned long long n_elem = this->size();
    s.pack(n_elem);

    // pack the data
    s.pack(pdata.get(), n_elem);
}

// --------------------------------------------------------------------------
template<typename T>
    template <typename U>
void teca_variant_array_impl<T>::from_binary(teca_binary_stream &s,
    typename std::enable_if<pack_array<U>::value, U>::type*)
{
    // unpack the size
    unsigned long long n_elem = 0;
    s.unpack(n_elem);

    // allocate a buffer
    hamr::buffer<T> tmp(allocator::malloc, n_elem);
    std::shared_ptr<T> sptmp = tmp.pointer();

    // unpack the elements  into the buffer
    s.unpack(sptmp.get(), n_elem);

    // update the array
    m_data = std::move(tmp);
}

// --------------------------------------------------------------------------
template<typename T>
    template <typename U>
void teca_variant_array_impl<T>::to_binary(teca_binary_stream &s,
    typename std::enable_if<pack_object<U>::value, U>::type*) const
{
    // pack the size
    unsigned long long n_elem = this->size();
    s.pack(n_elem);

    // pack the data from the CPU
    std::shared_ptr<const T> data = this->get_host_accessible();
    const T *pdata = data.get();

    sync_host_access_any(this);

    for (unsigned long long i = 0; i < n_elem; ++i)
    {
        pdata[i].to_stream(s);
    }
}

// --------------------------------------------------------------------------
template<typename T>
    template <typename U>
void teca_variant_array_impl<T>::from_binary(
    teca_binary_stream &s,
    typename std::enable_if<pack_object<U>::value, U>::type*)
{
    // unpack the size
    unsigned long long n_elem = 0;
    s.unpack(n_elem);

    // always unpack the data to the CPU
    std::vector<T> tmp(n_elem);
    for (unsigned long long i = 0; i < n_elem; ++i)
    {
        tmp[i].from_stream(s);
    }

    // copy to the active device
    m_data.assign(tmp.data(), 0, n_elem);
}

// --------------------------------------------------------------------------
template<typename T>
    template <typename U>
void teca_variant_array_impl<T>::to_binary(
    teca_binary_stream &s,
    typename std::enable_if<pack_object_ptr<U>::value, U>::type*) const
{
    // pack the size
    unsigned long long n_elem = this->size();
    s.pack(n_elem);

    // pack the data from the CPU
    std::shared_ptr<const T> data = this->get_host_accessible();
    const T *pdata = data.get();

    sync_host_access_any(this);

    for (unsigned long long i = 0; i < n_elem; ++i)
    {
       pdata[i]->to_stream(s);
    }
}

// --------------------------------------------------------------------------
template<typename T>
    template <typename U>
void teca_variant_array_impl<T>::from_binary(
    teca_binary_stream &s,
    typename std::enable_if<pack_object_ptr<U>::value, U>::type*)
{
    // unpack the size
    unsigned long long n_elem = 0;
    s.unpack(n_elem);

    // unpack to the CPU
    std::vector<T> tmp(n_elem);
    for (unsigned long long i=0; i < n_elem; ++i)
    {
        tmp[i]->from_stream(s);
    }

    // copy to the active device
    m_data.assign(tmp.data(), 0, n_elem);
}

#define STR_DELIM(_a, _b) \
    (std::is_same<T, std::string>::value ? _a : _b)

// --------------------------------------------------------------------------
template<typename T>
    template <typename U>
void teca_variant_array_impl<T>::to_ascii(
    std::ostream &s,
    typename std::enable_if<pack_array<U>::value, U>::type*) const
{
    size_t n_elem = this->size();
    if (n_elem)
    {
        // serialize from the CPU
        std::shared_ptr<const T> data = this->get_host_accessible();
        const T *pdata = data.get();

        sync_host_access_any(this);

        s << STR_DELIM("\"", "") << pdata[0] << STR_DELIM("\"", "");

        for (size_t i = 1; i < n_elem; ++i)
        {
            s << STR_DELIM(", \"", ", ") << pdata[i] << STR_DELIM("\"", "");
        }
    }
}

// --------------------------------------------------------------------------
template<typename T>
    template <typename U>
void teca_variant_array_impl<T>::from_ascii(
    std::ostream &,
    typename std::enable_if<pack_array<U>::value, U>::type*)
{
    // TODO
}

// --------------------------------------------------------------------------
template<typename T>
    template <typename U>
void teca_variant_array_impl<T>::to_ascii(
    std::ostream &s,
    typename std::enable_if<pack_object<U>::value, U>::type*) const
{
    size_t n_elem = this->m_data.size();
    if (n_elem)
    {
        // serialize from the CPU
        std::shared_ptr<const T> data = this->get_host_accessible();
        const T *pdata = data.get();

        sync_host_access_any(this);

        s << "{";
        pdata[0].to_stream(s);
        s << "}";
        for (size_t i = 1; i < n_elem; ++i)
        {
            s << ", {";
            pdata[i].to_stream(s);
            s << "}";
        }
    }
}

// --------------------------------------------------------------------------
template<typename T>
    template <typename U>
void teca_variant_array_impl<T>::from_ascii(
    std::ostream &,
    typename std::enable_if<pack_object<U>::value, U>::type*)
{
    // TODO
}

// --------------------------------------------------------------------------
template<typename T>
    template <typename U>
void teca_variant_array_impl<T>::to_ascii(
    std::ostream &s,
    typename std::enable_if<pack_object_ptr<U>::value, U>::type*) const
{
    size_t n_elem = this->m_data.size();
    if (n_elem)
    {
        // serialize from the CPU
        std::shared_ptr<const T> data = this->get_host_accessible();
        const T *pdata = data.get();

        sync_host_access_any(this);

        s << "{";
        pdata[0]->to_stream(s);
        s << "}";
        for (size_t i = 1; i < n_elem; ++i)
        {
            s << ", {";
            pdata[i]->to_stream(s);
            s << "}";
        }
    }
}

// --------------------------------------------------------------------------
template<typename T>
    template <typename U>
void teca_variant_array_impl<T>::from_ascii(
    std::ostream &,
    typename std::enable_if<pack_object_ptr<U>::value, U>::type*)
{
    // TODO
}

/// @cond
template <typename T>
struct TECA_EXPORT teca_variant_array_code {};

template <unsigned int I>
struct TECA_EXPORT teca_variant_array_new {};

template <unsigned int I>
struct TECA_EXPORT teca_variant_array_type {};

#define TECA_VARIANT_ARRAY_TT_SPEC(T, v)                        \
template <>                                                     \
struct teca_variant_array_code<T>                               \
{                                                               \
    static constexpr unsigned int get()                         \
    { return v; }                                               \
};                                                              \
template <>                                                     \
struct teca_variant_array_new<v>                                \
{                                                               \
    using allocator = teca_variant_array::allocator;            \
                                                                \
    static p_teca_variant_array_impl<T> New(allocator alloc)    \
    { return teca_variant_array_impl<T>::New(alloc); }          \
};                                                              \
template <>                                                     \
struct teca_variant_array_type<v>                               \
{                                                               \
    using type = T;                                             \
                                                                \
    static constexpr const char *name()                         \
    { return #T; }                                              \
};

#define TECA_VARIANT_ARRAY_FACTORY_NEW(_v)                  \
        case _v:                                            \
            return teca_variant_array_new<_v>::New(alloc);

#include "teca_metadata.h"
class teca_metadata;

TECA_VARIANT_ARRAY_TT_SPEC(char, 1)
TECA_VARIANT_ARRAY_TT_SPEC(unsigned char, 2)
TECA_VARIANT_ARRAY_TT_SPEC(int, 3)
TECA_VARIANT_ARRAY_TT_SPEC(unsigned int, 4)
TECA_VARIANT_ARRAY_TT_SPEC(short int, 5)
TECA_VARIANT_ARRAY_TT_SPEC(short unsigned int, 6)
TECA_VARIANT_ARRAY_TT_SPEC(long, 7)
TECA_VARIANT_ARRAY_TT_SPEC(unsigned long, 8)
TECA_VARIANT_ARRAY_TT_SPEC(long long, 9)
TECA_VARIANT_ARRAY_TT_SPEC(unsigned long long, 10)
TECA_VARIANT_ARRAY_TT_SPEC(float, 11)
TECA_VARIANT_ARRAY_TT_SPEC(double, 12)
TECA_VARIANT_ARRAY_TT_SPEC(std::string, 13)
TECA_VARIANT_ARRAY_TT_SPEC(teca_metadata, 14)
TECA_VARIANT_ARRAY_TT_SPEC(p_teca_variant_array, 15)
/// @endcond

/** @brief Creates an instance of teca_variant_array_impl<T> where T is
 * determined from the type code.
 */
struct TECA_EXPORT teca_variant_array_factory
{
    using allocator = teca_variant_array::allocator;

    /** Construct a new variant array.
     * @param[in] type_code the type of array to construct
     * @param[in] alloc the allocator to use
     * @returns a new variant array or a nullptr if the type_code was invalid.
     */
    static p_teca_variant_array New(unsigned int type_code,
        allocator alloc = allocator::malloc)
    {
        switch (type_code)
        {
        TECA_VARIANT_ARRAY_FACTORY_NEW(1)
        TECA_VARIANT_ARRAY_FACTORY_NEW(2)
        TECA_VARIANT_ARRAY_FACTORY_NEW(3)
        TECA_VARIANT_ARRAY_FACTORY_NEW(4)
        TECA_VARIANT_ARRAY_FACTORY_NEW(5)
        TECA_VARIANT_ARRAY_FACTORY_NEW(6)
        TECA_VARIANT_ARRAY_FACTORY_NEW(7)
        TECA_VARIANT_ARRAY_FACTORY_NEW(8)
        TECA_VARIANT_ARRAY_FACTORY_NEW(9)
        TECA_VARIANT_ARRAY_FACTORY_NEW(10)
        TECA_VARIANT_ARRAY_FACTORY_NEW(11)
        TECA_VARIANT_ARRAY_FACTORY_NEW(12)
        TECA_VARIANT_ARRAY_FACTORY_NEW(13)
        TECA_VARIANT_ARRAY_FACTORY_NEW(14)
        TECA_VARIANT_ARRAY_FACTORY_NEW(15)
        default:
            TECA_ERROR("Failed to create a teca_variant_array,"
                " unknown code " << type_code)
        }
    return nullptr;
    }
};

/// @cond
#define CODE_DISPATCH_CASE(_v, _c, _code)               \
    if (_v == _c)                                       \
    {                                                   \
        using NT = teca_variant_array_type<_c>::type;   \
        using TT = teca_variant_array_impl<NT>;         \
        _code                                           \
    }

#define CODE_DISPATCH_I(_v, _code)          \
    CODE_DISPATCH_CASE(_v, 1, _code)        \
    else CODE_DISPATCH_CASE(_v, 2, _code)   \
    else CODE_DISPATCH_CASE(_v, 3, _code)   \
    else CODE_DISPATCH_CASE(_v, 4, _code)   \
    else CODE_DISPATCH_CASE(_v, 5, _code)   \
    else CODE_DISPATCH_CASE(_v, 6, _code)   \
    else CODE_DISPATCH_CASE(_v, 7, _code)   \
    else CODE_DISPATCH_CASE(_v, 8, _code)   \
    else CODE_DISPATCH_CASE(_v, 9, _code)   \
    else CODE_DISPATCH_CASE(_v, 10, _code)

#define CODE_DISPATCH_FP(_v, _code)         \
    CODE_DISPATCH_CASE(_v, 11, _code)       \
    else CODE_DISPATCH_CASE(_v, 12, _code)

#define CODE_DISPATCH_CLASS(_v, _code)      \
    CODE_DISPATCH_CASE(_v, 13, _code)       \
    else CODE_DISPATCH_CASE(_v, 14, _code)  \
    else CODE_DISPATCH_CASE(_v, 15, _code)

#define CODE_DISPATCH(_v, _code)            \
    CODE_DISPATCH_I(_v, _code)              \
    else CODE_DISPATCH_FP(_v, _code)

/// @endcond

// --------------------------------------------------------------------------
template <typename T>
unsigned int teca_variant_array_impl<T>::type_code() const noexcept
{
    return teca_variant_array_code<T>::get();
}

// **************************************************************************
template <typename T>
TECA_EXPORT
T min(const const_p_teca_variant_array_impl<T> &a)
{
    size_t n_elem = a->size();
    T mn = std::numeric_limits<T>::max();
#if defined(TECA_HAS_CUDA)
    if (a->cuda_accessible())
    {
        std::shared_ptr<const T> data = a->get_cuda_accessible();
        thrust::device_ptr<const T> pdata(data.get());
        mn = thrust::reduce(pdata, pdata + n_elem, mn, thrust::minimum<T>());
    }
    else
    {
#endif
        std::shared_ptr<const T> data = a->get_cuda_accessible();
        const T *pdata = data.get();
        for (size_t i = 0; i < n_elem; ++i)
            mn = mn > pdata[i] ? pdata[i] : mn;
#if defined(TECA_HAS_CUDA)
    }
#endif
    return mn;
}

// **************************************************************************
template <typename T>
TECA_EXPORT
T min(const p_teca_variant_array_impl<T> &a)
{
    return min(const_p_teca_variant_array_impl<T>(a));
}

// **************************************************************************
template <typename T>
TECA_EXPORT
T max(const const_p_teca_variant_array_impl<T> &a)
{
    size_t n_elem = a->size();
    T mx = std::numeric_limits<T>::lowest();
#if defined(TECA_HAS_CUDA)
    if (a->cuda_accessible())
    {
        std::shared_ptr<const T> data = a->get_cuda_accessible();
        thrust::device_ptr<const T> pdata(data.get());
        mx = thrust::reduce(pdata, pdata + n_elem, mx, thrust::maximum<T>());
    }
    else
    {
#endif
        std::shared_ptr<const T> data = a->get_cuda_accessible();
        const T *pdata = data.get();
        for (size_t i = 0; i < n_elem; ++i)
            mx = mx < pdata[i] ? pdata[i] : mx;
#if defined(TECA_HAS_CUDA)
    }
#endif
    return mx;
}

// **************************************************************************
template <typename T>
TECA_EXPORT
T max(const p_teca_variant_array_impl<T> &a)
{
    return max(const_p_teca_variant_array_impl<T>(a));
}

#if defined(__CUDACC__)
#pragma nv_diag_default = partial_override
#endif
#endif
