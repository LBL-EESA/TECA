#ifndef teca_variant_array_h
#define teca_variant_array_h

#include <vector>
#include <string>
#include <exception>
#include <typeinfo>
#include <iterator>
#include <algorithm>
#include <type_traits>
#include <utility>

#include "teca_common.h"
#include "teca_binary_stream.h"
#include "teca_variant_array_fwd.h"

// tag for ops on POD data
template <typename T>
struct pod_dispatch :
    std::integral_constant<bool,
    std::is_arithmetic<T>::value>
{};

// tag for ops on classes
template <typename T>
struct object_dispatch :
    std::integral_constant<bool,
    !std::is_arithmetic<T>::value>
{};

/// type agnostic container for array based data
/**
type agnostic container for array based data.
*/
class teca_variant_array : public std::enable_shared_from_this<teca_variant_array>
{
public:
    // construct
    teca_variant_array() noexcept = default;
    virtual ~teca_variant_array() noexcept = default;

    // copy/move construct. can't copy/move construct from a base
    // pointer. these require a downcast.
    teca_variant_array(const teca_variant_array &other) = delete;
    teca_variant_array(teca_variant_array &&other) = delete;

    // copy assign
    teca_variant_array &operator=(const teca_variant_array &other)
    { this->copy(other); return *this; }

    // move assign
    teca_variant_array &operator=(teca_variant_array &&other)
    { this->swap(other); return *this; }

    // virtual constructor. return a new'ly allocated
    // empty object of the same type.
    virtual p_teca_variant_array new_instance() const = 0;
    virtual p_teca_variant_array new_instance(size_t n) const = 0;

    // virtual copy construct. return a new'ly allocated object,
    // initialized copy from this. caller must delete.
    virtual p_teca_variant_array new_copy() const = 0;
    virtual p_teca_variant_array new_copy(size_t start, size_t end) const = 0;

    // return true if values are equal
    bool operator==(const teca_variant_array &other) const
    { return this->equal(other); }

    // get methods. could throw std::bad_cast if the
    // internal type is not castable to the return type.
    template<typename T>
    void get(unsigned long i, T &val) const
    { this->get_dispatch<T>(i, val); }

    template<typename T>
    void get(std::vector<T> &vals) const
    { this->get_dispatch<T>(vals); }

    template<typename T>
    void get(size_t start, size_t end, T *vals) const
    { this->get_dispatch<T>(start, end, vals); }

    // set methods. could throw std::bad_cast if the
    // passed in type is not castable to the internal type.
    template<typename T>
    void set(const std::vector<T> &vals)
    { this->set_dispatch<T>(vals); }

    template<typename T>
    void set(unsigned long i, const T &val)
    { this->set_dispatch<T>(i, val); }

    template<typename T>
    void set(size_t start, size_t end, const T *vals)
    { this->set_dispatch<T>(start, end, vals); }

    // append methods. could throw std::bad_cast if the
    // passed in type is not castable to the internal type.
    template<typename T>
    void append(const T &val)
    { this->append_dispatch(val); }

    template<typename T>
    void append(const std::vector<T> &vals)
    { this->append_dispatch(vals); }

    // get the number of elements in the array
    virtual unsigned long size() const noexcept = 0;

    // resize. allocates new storage and copies in existing values
    virtual void resize(unsigned long i) = 0;

    // reserve. reserves the requested ammount of space with out
    // constructing elements
    virtual void reserve(unsigned long i) = 0;

    // free all the internal data
    virtual void clear() noexcept = 0;

    // copy the contents from the other array.
    // an excpetion is thrown when no conversion
    // between the two types exists. This method
    // is not virtual so that string can be handled
    // as a special case in the base class.
    void copy(const teca_variant_array &other);
    void copy(const const_p_teca_variant_array &other)
    { this->copy(*other.get()); }

    void append(const teca_variant_array &other);
    void append(const const_p_teca_variant_array &other)
    { this->append(*other.get()); }

    // swap the contents of this and the other object.
    // an excpetion is thrown when no conversion
    // between the two types exists.
    virtual void swap(teca_variant_array &other) = 0;
    void swap(const p_teca_variant_array &other)
    { this->swap(*other.get()); }

    // compare the two objects for equality
    virtual bool equal(const teca_variant_array &other) const = 0;
    bool equal(const const_p_teca_variant_array &other) const
    { return this->equal(*other.get()); }

    // serrialize to/from stream
    virtual void to_stream(teca_binary_stream &s) const = 0;
    virtual void from_stream(teca_binary_stream &s) = 0;

    virtual void to_stream(std::ostream &s) const = 0;
    virtual void from_stream(std::ostream &s) = 0;

    // used for serialization
    virtual unsigned int type_code() const noexcept = 0;

private:
    // dispatch methods, each set/get above has a pair
    // one for POD and one for the rest. this allows us to
    // seamlessly handle casting and conversion between POD
    // types
    template<typename T>
    void append_dispatch(const std::vector<T> &vals,
        typename std::enable_if<pod_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void append_dispatch(const std::vector<T> &vals,
        typename std::enable_if<object_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void append_dispatch(const T &val,
        typename std::enable_if<pod_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void append_dispatch(const T &val,
        typename std::enable_if<object_dispatch<T>::value, T>::type* = 0);
    template<typename T>

    void set_dispatch(const std::vector<T> &vals,
        typename std::enable_if<pod_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void set_dispatch(const std::vector<T> &vals,
        typename std::enable_if<object_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void set_dispatch(unsigned long i, const T &val,
        typename std::enable_if<pod_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void set_dispatch(unsigned long i, const T &val,
        typename std::enable_if<object_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void set_dispatch(size_t start, size_t end, const T *vals,
        typename std::enable_if<pod_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void set_dispatch(size_t start, size_t end, const T *vals,
        typename std::enable_if<object_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void get_dispatch(std::vector<T> &vals,
        typename std::enable_if<pod_dispatch<T>::value, T>::type* = 0) const;

    template<typename T>
    void get_dispatch(std::vector<T> &vals,
        typename std::enable_if<object_dispatch<T>::value, T>::type* = 0) const;

    template<typename T>
    void get_dispatch(unsigned long i, T &val,
        typename std::enable_if<pod_dispatch<T>::value, T>::type* = 0) const;

    template<typename T>
    void get_dispatch(unsigned long i, T &val,
        typename std::enable_if<object_dispatch<T>::value, T>::type* = 0) const;

    template<typename T>
    void get_dispatch(size_t start, size_t end, T *vals,
        typename std::enable_if<pod_dispatch<T>::value, T>::type* = 0) const;

    template<typename T>
    void get_dispatch(size_t start, size_t end, T *vals,
        typename std::enable_if<object_dispatch<T>::value, T>::type* = 0) const;
};



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



// implementation of our type agnostic container
// for simple arrays
template<typename T>
class teca_variant_array_impl : public teca_variant_array
{
public:
    // construct
    TECA_VARIANT_ARRAY_STATIC_NEW(teca_variant_array_impl, T)

    // destruct
    virtual ~teca_variant_array_impl() noexcept;

    // virtual constructor
    virtual p_teca_variant_array new_copy() const override;
    virtual p_teca_variant_array new_copy(size_t start, size_t end) const override;
    virtual p_teca_variant_array new_instance() const override;
    virtual p_teca_variant_array new_instance(size_t n) const override;

    // copy
    const teca_variant_array_impl<T> &
    operator=(const teca_variant_array_impl<T> &other);

    template<typename U>
    const teca_variant_array_impl<T> &
    operator=(const teca_variant_array_impl<U> &other);

    // move
    teca_variant_array_impl(teca_variant_array_impl<T> &&other);

    const teca_variant_array_impl<T> &
    operator=(teca_variant_array_impl<T> &&other);

    // get the ith value
    T &get(unsigned long i)
    { return m_data[i]; }

    const T &get(unsigned long i) const
    { return m_data[i]; }

    // get the ith value
    template<typename U>
    void get(unsigned long i, U &val) const;

    // get a range of values decribed by [start end]
    // inclusive
    template<typename U>
    void get(size_t start, size_t end, U *vals) const;

    // copy the data out into the passed in vector
    template<typename U>
    void get(std::vector<U> &val) const;

    // pointer to the data
    T *get(){ return &m_data[0]; }
    const T *get() const { return &m_data[0]; }

    // set the ith value
    template<typename U>
    void set(unsigned long i, const U &val);

    // set a range opf values described by [start end]
    // inclusive
    template<typename U>
    void set(size_t start, size_t end, const U *vals);

    // copy data, replacing contents with the passed in
    // vector
    template<typename U>
    void set(const std::vector<U> &val);

    // insert from the passed in vector at the back
    template<typename U>
    void append(const std::vector<U> &val);

    // insert a single value at the back
    template<typename U>
    void append(const U &val);

    // get the current size of the data
    virtual unsigned long size() const noexcept override;

    // resize the data
    virtual void resize(unsigned long n) override;
    void resize(unsigned long n, const T &val);

    // reserve space
    virtual void reserve(unsigned long n) override;

    // clear the data
    virtual void clear() noexcept override;

    // copy. This method is not virtual so that
    // string can be handled as a special case in
    // the base class.
    void copy(const teca_variant_array &other);

    // append. This method is not virtual so that
    // string can be handled as a special case in
    // the base class.
    void append(const teca_variant_array &other);

    // virtual swap
    virtual void swap(teca_variant_array &other) override;

    // virtual equavalince test
    virtual bool equal(const teca_variant_array &other) const override;

    // serialize to/from stream
    virtual void to_stream(teca_binary_stream &s) const override
    { this->to_binary<T>(s); }

    virtual void from_stream(teca_binary_stream &s) override
    { this->from_binary<T>(s); }

    virtual void to_stream(std::ostream &s) const override
    { this->to_ascii<T>(s); }

    virtual void from_stream(std::ostream &s) override
    { this->from_ascii<T>(s); }

protected:
    // construct
    teca_variant_array_impl() noexcept {}

    // construct with preallocated size
    teca_variant_array_impl(unsigned long n)
        : m_data(n) {}

    // construct with preallocated size and initialized
    // to a specific value
    teca_variant_array_impl(unsigned long n, const T &v)
        : m_data(n, v) {}

    // construct from a c-array of length n
    teca_variant_array_impl(const T *vals, unsigned long n)
        : teca_variant_array(), m_data(vals, vals+n) {}

    // copy construct from an instance of different type
    template<typename U>
    teca_variant_array_impl(const teca_variant_array_impl<U> &other)
        : teca_variant_array(), m_data(other.m_data) {}

    // copy construct from an instance of same type
    teca_variant_array_impl(const teca_variant_array_impl<T> &other)
        : teca_variant_array(), m_data(other.m_data) {}

private:
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

    // tag dispatch array of poniter to other objects
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

    // for serializaztion
    virtual unsigned int type_code() const noexcept override;
private:
    std::vector<T> m_data;

    friend class teca_variant_array;
    template<typename U> friend class teca_variant_array_impl;
};


#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"

// tt - derived container
// nt - contained type
// tt<nt> - complete derived type
// p - base class pointer
// body - code to execute
#define TEMPLATE_DISPATCH_CASE(tt, nt, p, body) \
    if (dynamic_cast<tt<nt>*>(p))               \
    {                                           \
        using TT = tt<nt>;                      \
        using NT = nt;                          \
        body                                    \
    }

// tt - derived container
// nt - contained type
// tt<nt> - complete derived type
// p - base class pointer
// i - id suffix
// body - code to execute
#define NESTED_TEMPLATE_DISPATCH_CASE(tt, nt, p, i, body)   \
    if (dynamic_cast<tt<nt>*>(p))                           \
    {                                                       \
        using TT##i = tt<nt>;                               \
        using NT##i = nt;                                   \
        body                                                \
    }

// tt - container
// nt - class type
// tt<nt> - complete derived type
// p1 - base class pointer
// p2 - const base class pointer
// body - code to execute if p1 and p2's type match
#define TEMPLATE_DISPATCH_CLASS(tt, nt, p1, p2, body)   \
    {                                                   \
    using TT = tt<nt>;                                  \
    using NT = nt;                                      \
    TT *p1_tt = dynamic_cast<TT*>(p1);                  \
    const TT *p2_tt = dynamic_cast<const TT*>(p2);      \
    if (p1_tt && p2_tt)                                 \
    {                                                   \
        body                                            \
    }                                                   \
    }

// variant that limits dispatch to floating point types
// for use in numerical compuatation where integer types
// are not supported (ie, math operations from std library)
#define TEMPLATE_DISPATCH_FP(t, p, body)        \
    TEMPLATE_DISPATCH_CASE(t, float, p, body)   \
    else TEMPLATE_DISPATCH_CASE(t, double, p, body)

// variant that limits dispatch to integer types
// for use in numerical compuatation where floating point types
// are not supported
#define TEMPLATE_DISPATCH_I(t, p, body)                         \
    TEMPLATE_DISPATCH_CASE(t, long long, p, body)               \
    else TEMPLATE_DISPATCH_CASE(t, unsigned long long, p, body) \
    else TEMPLATE_DISPATCH_CASE(t, long, p, body)               \
    else TEMPLATE_DISPATCH_CASE(t, int, p, body)                \
    else TEMPLATE_DISPATCH_CASE(t, unsigned int, p, body)       \
    else TEMPLATE_DISPATCH_CASE(t, unsigned long, p, body)      \
    else TEMPLATE_DISPATCH_CASE(t, short int, p, body)          \
    else TEMPLATE_DISPATCH_CASE(t, short unsigned int, p, body) \
    else TEMPLATE_DISPATCH_CASE(t, char, p, body)               \
    else TEMPLATE_DISPATCH_CASE(t, unsigned char, p, body)

// macro for helping downcast to POD types
// don't add classes to this.
// t - derived container type
// p - pointer to base class
// body - code to execute on match
#define TEMPLATE_DISPATCH(t, p, body)       \
    TEMPLATE_DISPATCH_FP(t, p, body)        \
    else TEMPLATE_DISPATCH_I(t, p, body)

// variant that limits dispatch to floating point types
// for use in numerical compuatation where integer types
// are not supported (ie, math operations from std library)
#define NESTED_TEMPLATE_DISPATCH_FP(t, p, i, body)              \
    NESTED_TEMPLATE_DISPATCH_CASE(t, float, p, i, body)         \
    else NESTED_TEMPLATE_DISPATCH_CASE(t, double, p, i, body)

// variant that limits dispatch to integer types
// for use in numerical compuatation where integer types
// are not supported (ie, math operations from std library)
#define NESTED_TEMPLATE_DISPATCH_I(t, p, i, body)      \
    NESTED_TEMPLATE_DISPATCH_CASE(t, long long, p, i, body)               \
    else NESTED_TEMPLATE_DISPATCH_CASE(t, unsigned long long, p, i, body) \
    else NESTED_TEMPLATE_DISPATCH_CASE(t, long, p, i, body)               \
    else NESTED_TEMPLATE_DISPATCH_CASE(t, int, p, i, body)                \
    else NESTED_TEMPLATE_DISPATCH_CASE(t, unsigned int, p, i, body)       \
    else NESTED_TEMPLATE_DISPATCH_CASE(t, unsigned long, p, i, body)      \
    else NESTED_TEMPLATE_DISPATCH_CASE(t, short int, p, i, body)          \
    else NESTED_TEMPLATE_DISPATCH_CASE(t, short unsigned int, p, i, body) \
    else NESTED_TEMPLATE_DISPATCH_CASE(t, char, p, i, body)               \
    else NESTED_TEMPLATE_DISPATCH_CASE(t, unsigned char, p, i, body)

// macro for helping downcast to POD types
// don't add classes to this.
// t - templated derived type
// p - base class pointer
// i - id for nesting
// body - code to execute on match
#define NESTED_TEMPLATE_DISPATCH(t, p, i, body)     \
    NESTED_TEMPLATE_DISPATCH_FP(t, p, i, body)      \
    else NESTED_TEMPLATE_DISPATCH_I(t, p, i, body)


// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::get_dispatch(std::vector<T> &vals,
    typename std::enable_if<pod_dispatch<T>::value, T>::type*) const
{
    TEMPLATE_DISPATCH(const teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->get(vals);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::get_dispatch(std::vector<T> &vals,
    typename std::enable_if<object_dispatch<T>::value, T>::type*) const
{
    TEMPLATE_DISPATCH_CASE(const teca_variant_array_impl, T, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->get(vals);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::get_dispatch(unsigned long i, T &val,
    typename std::enable_if<pod_dispatch<T>::value, T>::type*) const
{
    TEMPLATE_DISPATCH(const teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->get(i, val);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::get_dispatch(unsigned long i, T &val,
    typename std::enable_if<object_dispatch<T>::value, T>::type*) const
{
    TEMPLATE_DISPATCH_CASE(const teca_variant_array_impl, T, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->get(i, val);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::get_dispatch(size_t start, size_t end, T *vals,
    typename std::enable_if<pod_dispatch<T>::value, T>::type*) const
{
    TEMPLATE_DISPATCH(const teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->get(start, end, vals);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::get_dispatch(size_t start, size_t end, T *vals,
    typename std::enable_if<object_dispatch<T>::value, T>::type*) const
{
    TEMPLATE_DISPATCH_CASE(const teca_variant_array_impl, T, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->get(start, end, vals);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::set_dispatch(const std::vector<T> &vals,
    typename std::enable_if<pod_dispatch<T>::value, T>::type*)
{
    TEMPLATE_DISPATCH(teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->set(vals);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::set_dispatch(const std::vector<T> &vals,
    typename std::enable_if<object_dispatch<T>::value, T>::type*)
{
    TEMPLATE_DISPATCH_CASE(teca_variant_array_impl, T, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->set(vals);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::set_dispatch(unsigned long i, const T &val,
    typename std::enable_if<pod_dispatch<T>::value, T>::type*)
{
    TEMPLATE_DISPATCH(teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->set(i, val);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::set_dispatch(unsigned long i, const T &val,
    typename std::enable_if<object_dispatch<T>::value, T>::type*)
{
    TEMPLATE_DISPATCH_CASE(teca_variant_array_impl, T, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->set(i, val);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::set_dispatch(size_t start, size_t end, const T *vals,
    typename std::enable_if<pod_dispatch<T>::value, T>::type*)
{
    TEMPLATE_DISPATCH(teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->set(start, end, vals);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::set_dispatch(size_t start, size_t end, const T *vals,
    typename std::enable_if<object_dispatch<T>::value, T>::type*)
{
    TEMPLATE_DISPATCH_CASE(const teca_variant_array_impl, T, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->set(start, end, vals);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::append_dispatch(const std::vector<T> &vals,
    typename std::enable_if<pod_dispatch<T>::value, T>::type*)
{
    TEMPLATE_DISPATCH(teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->append(vals);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::append_dispatch(const std::vector<T> &vals,
    typename std::enable_if<object_dispatch<T>::value, T>::type*)
{
    TEMPLATE_DISPATCH_CASE(teca_variant_array_impl, T, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->append(vals);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::append_dispatch(const T &val,
    typename std::enable_if<pod_dispatch<T>::value, T>::type*)
{
    TEMPLATE_DISPATCH(teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->append(val);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::append_dispatch(const T &val,
    typename std::enable_if<object_dispatch<T>::value, T>::type*)
{
    TEMPLATE_DISPATCH_CASE(teca_variant_array_impl, T, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->append(val);
        return;
        )
    throw std::bad_cast();
}





// --------------------------------------------------------------------------
template<typename T>
teca_variant_array_impl<T>::~teca_variant_array_impl() noexcept
{
    this->clear();
}

// --------------------------------------------------------------------------
template<typename T>
p_teca_variant_array teca_variant_array_impl<T>::new_copy() const
{
    return p_teca_variant_array(new teca_variant_array_impl<T>(*this));
}

// --------------------------------------------------------------------------
template<typename T>
p_teca_variant_array teca_variant_array_impl<T>::new_copy(
    size_t start, size_t end) const
{
    p_teca_variant_array_impl<T> c = teca_variant_array_impl<T>::New(end-start+1);
    this->get(start, end, c->get());
    return c;
}

// --------------------------------------------------------------------------
template<typename T>
p_teca_variant_array teca_variant_array_impl<T>::new_instance() const
{
    return p_teca_variant_array(new teca_variant_array_impl<T>());
}

// --------------------------------------------------------------------------
template<typename T>
p_teca_variant_array teca_variant_array_impl<T>::new_instance(size_t n) const
{
    return p_teca_variant_array(new teca_variant_array_impl<T>(n));
}

// --------------------------------------------------------------------------
template<typename T>
const teca_variant_array_impl<T> &
teca_variant_array_impl<T>::operator=(const teca_variant_array_impl<T> &other)
{
    m_data.assign(other.m_data.begin(), other.m_data.end());
    return *this;
}

// copy assignment from different type
// --------------------------------------------------------------------------
template<typename T>
template<typename U>
const teca_variant_array_impl<T> &
teca_variant_array_impl<T>::operator=(const teca_variant_array_impl<U> &other)
{
    m_data.assign(other.m_data.begin(), other.m_data.end());
    return *this;
}

// --------------------------------------------------------------------------
template<typename T>
teca_variant_array_impl<T>::teca_variant_array_impl(
    teca_variant_array_impl<T> &&other)
    : m_data(std::move(other.m_data))
{}

// --------------------------------------------------------------------------
template<typename T>
const teca_variant_array_impl<T> &teca_variant_array_impl<T>::operator=(
    teca_variant_array_impl<T> &&other)
{
    m_data = std::move(other.m_data);
    return *this;
}

// --------------------------------------------------------------------------
template<typename T>
template<typename U>
void teca_variant_array_impl<T>::get(unsigned long i, U &val) const
{
    val = m_data[i];
}

// --------------------------------------------------------------------------
template<typename T>
template<typename U>
void teca_variant_array_impl<T>::get(size_t start, size_t end, U *vals) const
{
    for (size_t i = start, ii = 0; i <= end; ++i, ++ii)
        vals[ii] = m_data[i];
}

// --------------------------------------------------------------------------
template<typename T>
template<typename U>
void teca_variant_array_impl<T>::get(std::vector<U> &val) const
{
    val.assign(m_data.begin(), m_data.end());
}

// --------------------------------------------------------------------------
template<typename T>
template<typename U>
void teca_variant_array_impl<T>::set(unsigned long i, const U &val)
{
    m_data[i] = val;
}

// --------------------------------------------------------------------------
template<typename T>
template<typename U>
void teca_variant_array_impl<T>::set(size_t start, size_t end, const U *vals)
{
    for (size_t i = start, ii = 0; i <= end; ++i, ++ii)
        m_data[i] = vals[ii];
}

// --------------------------------------------------------------------------
template<typename T>
template<typename U>
void teca_variant_array_impl<T>::set(const std::vector<U> &val)
{
    size_t n = val.size();
    m_data.resize(n);
    for (size_t i = 0; i < n; ++i)
        m_data[i] = static_cast<T>(val[i]);
}

// --------------------------------------------------------------------------
template<typename T>
template<typename U>
void teca_variant_array_impl<T>::append(const std::vector<U> &val)
{
    std::copy(val.begin(), val.end(), std::back_inserter(m_data));
}

// --------------------------------------------------------------------------
template<typename T>
template<typename U>
void teca_variant_array_impl<T>::append(const U &val)
{
    m_data.push_back(val);
}

// --------------------------------------------------------------------------
template<typename T>
unsigned long teca_variant_array_impl<T>::size() const noexcept
{ return m_data.size(); }

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
    m_data.clear();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array_impl<T>::copy(const teca_variant_array &other)
{
    TEMPLATE_DISPATCH(const teca_variant_array_impl, &other,
        TT *other_t = static_cast<TT*>(&other);
        *this = *other_t;
        return;
        )
     throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array_impl<T>::append(const teca_variant_array &other)
{
    TEMPLATE_DISPATCH(const teca_variant_array_impl, &other,
        TT *other_t = static_cast<TT*>(&other);
        std::copy(
            other_t->m_data.begin(),
            other_t->m_data.end(),
            std::back_inserter(this->m_data));
        return;
        )
     throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array_impl<T>::swap(teca_variant_array &other)
{
    using TT = teca_variant_array_impl<T>;
    TT *other_t = dynamic_cast<TT*>(&other);
    if (other_t)
    {
        this->m_data.swap(other_t->m_data);
        return;
    }
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
bool teca_variant_array_impl<T>::equal(const teca_variant_array &other) const
{
    using TT = teca_variant_array_impl<T>;
    const TT *other_t = dynamic_cast<const TT*>(&other);
    if (other_t)
    {
        return this->m_data == other_t->m_data;
    }
    throw std::bad_cast();
    return false;
}

// --------------------------------------------------------------------------
template<typename T>
    template <typename U>
void teca_variant_array_impl<T>::to_binary(
    teca_binary_stream &s,
    typename std::enable_if<pack_array<U>::value, U>::type*) const
{
    s.pack(this->m_data);
}

// --------------------------------------------------------------------------
template<typename T>
    template <typename U>
void teca_variant_array_impl<T>::from_binary(
    teca_binary_stream &s,
    typename std::enable_if<pack_array<U>::value, U>::type*)
{
    s.unpack(this->m_data);
}

// --------------------------------------------------------------------------
template<typename T>
    template <typename U>
void teca_variant_array_impl<T>::to_binary(
    teca_binary_stream &s,
    typename std::enable_if<pack_object<U>::value, U>::type*) const
{
    unsigned long long n = this->size();
    s.pack(n);
    for (unsigned long long i=0; i<n; ++i)
       this->m_data[i].to_stream(s);
}

// --------------------------------------------------------------------------
template<typename T>
    template <typename U>
void teca_variant_array_impl<T>::from_binary(
    teca_binary_stream &s,
    typename std::enable_if<pack_object<U>::value, U>::type*)
{
    unsigned long long n;
    s.unpack(n);
    this->resize(n);
    for (unsigned long long i=0; i<n; ++i)
       this->m_data[i].from_stream(s);
}

// --------------------------------------------------------------------------
template<typename T>
    template <typename U>
void teca_variant_array_impl<T>::to_binary(
    teca_binary_stream &s,
    typename std::enable_if<pack_object_ptr<U>::value, U>::type*) const
{
    unsigned long long n = this->size();
    s.pack(n);
    for (unsigned long long i=0; i<n; ++i)
       this->m_data[i]->to_stream(s);
}

// --------------------------------------------------------------------------
template<typename T>
    template <typename U>
void teca_variant_array_impl<T>::from_binary(
    teca_binary_stream &s,
    typename std::enable_if<pack_object_ptr<U>::value, U>::type*)
{
    unsigned long long n;
    s.unpack(n);
    this->resize(n);
    for (unsigned long long i=0; i<n; ++i)
       this->m_data[i]->from_stream(s);
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
    size_t n = this->m_data.size();
    if (n)
    {
        s << STR_DELIM("\"", "")
             << this->m_data[0] << STR_DELIM("\"", "");
        for (size_t i = 1; i < n; ++i)
        {
            s << STR_DELIM(", \"", ", ")
                << this->m_data[i] << STR_DELIM("\"", "");
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
    size_t n = this->m_data.size();
    if (n)
    {
        s << "{";
        this->m_data[0].to_stream(s);
        s << "}";
        for (size_t i = 1; i < n; ++i)
        {
            s << ", {";
            this->m_data[i].to_stream(s);
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
    size_t n = this->m_data.size();
    if (n)
    {
        s << "{";
        this->m_data[0]->to_stream(s);
        s << "}";
        for (size_t i = 1; i < n; ++i)
        {
            s << ", {";
            this->m_data[i]->to_stream(s);
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

template <typename T>
struct teca_variant_array_code
{}; // intentionally empty

template <unsigned int I>
struct teca_variant_array_new
{}; // intentionally empty

#define TECA_VARIANT_ARRAY_TT_SPEC(T, v)            \
template <>                                         \
struct teca_variant_array_code<T>                   \
{                                                   \
    static unsigned int get() noexcept              \
    { return v; }                                   \
};                                                  \
template <>                                         \
struct teca_variant_array_new<v>                    \
{                                                   \
    static p_teca_variant_array_impl<T> New()       \
    { return teca_variant_array_impl<T>::New(); }   \
};

#define TECA_VARIANT_ARRAY_FACTORY_NEW(_code)               \
        case _code:                                         \
            return teca_variant_array_new<_code>::New();

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

struct teca_variant_array_factory
{
    static p_teca_variant_array New(unsigned int type_code)
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
            TECA_ERROR(
                << "Failed to create from "
                << type_code)
        }
    return nullptr;
    }
};

// --------------------------------------------------------------------------
template<typename T>
unsigned int teca_variant_array_impl<T>::type_code() const noexcept
{
    return teca_variant_array_code<T>::get();
}

#endif
