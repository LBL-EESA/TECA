#ifndef teca_variant_array_h
#define teca_variant_array_h

#include <vector>
#include <string>
#include <exception>
#include <typeinfo>
#include <iterator>
#include <algorithm>

#include "teca_common.h"
#include "teca_compiler.h"

/// type agnostic container for simple arrays
/**
type agnostic container for array based data.
intended for use with small staticly sized
arrays, for example with metadata.
*/
class teca_variant_array
{
public:
    // construct
    teca_variant_array(){}
    virtual ~teca_variant_array() TECA_NOEXCEPT {}

    // copy
    teca_variant_array(const teca_variant_array &other)
    { this->copy(other); }

    teca_variant_array &operator=(const teca_variant_array &other)
    { this->copy(other); return *this; }

    // move
    teca_variant_array(teca_variant_array &&other)
    { this->swap(other); }

    teca_variant_array &operator=(teca_variant_array &&other)
    { this->swap(other); return *this; }

    // return a new'ly allocated object, initialized
    // copy from this. caller must delete.
    virtual teca_variant_array *new_copy() const = 0;

    // return true if values are equal
    bool operator==(const teca_variant_array &other)
    { return this->equal(other); }

    // get methods. could throw std::bad_cast if the
    // internal type is not castable to the return type.
    template<typename T>
    void get(T &val, unsigned int i=0) const;
    void get(std::string &val, unsigned int i=0) const;

    template<typename T>
    void get(std::vector<T> &vals) const;
    void get(std::vector<std::string> &vals) const;

    // set methods. could throw std::bad_cast if the
    // passed in type is not castable to the internal type.
    template<typename T>
    void set(const std::vector<T> &vals);
    void set(const std::vector<std::string> &vals);

    template<typename T>
    void set(const T &val, unsigned int i=0);
    void set(const std::string &val, unsigned int i=0);

    // append methods. could throw std::bad_cast if the
    // passed in type is not castable to the internal type.
    template<typename T>
    void append(const T &val);
    void append(const std::string &vals);

    template<typename T>
    void append(const std::vector<T> &vals);
    void append(const std::vector<std::string> &vals);

    // get the number of elements in the array
    virtual unsigned int size() const TECA_NOEXCEPT = 0;
    virtual void resize(unsigned int i) = 0;

    // free all the internal data
    virtual void clear() TECA_NOEXCEPT = 0;

    // copy the contents from the other array.
    // an excpetion is thrown when no conversion
    // between the two types exists. This method
    // is not virtual so that string can be handled
    // as a special case in the base class.
    void copy(const teca_variant_array &other);

    // swap the contents of this and the other object.
    // an excpetion is thrown when no conversion
    // between the two types exists.
    virtual void swap(teca_variant_array &other) = 0;

    // compare the two objects for equality
    virtual bool equal(const teca_variant_array &other) = 0;
};


// implementation of our type agnistic container
// for simple arrays
template <typename T>
class teca_variant_array_impl : public teca_variant_array
{
public:
    // construct
    teca_variant_array_impl() {}
    teca_variant_array_impl(unsigned int n) : m_data(n) {}

    teca_variant_array_impl(T *vals, unsigned int n)
        : m_data(vals, vals+n) {}

    ~teca_variant_array_impl() TECA_NOEXCEPT
    { this->clear(); }

    // copy
    template <typename U>
    teca_variant_array_impl(const teca_variant_array_impl<U> &other)
        : m_data(other.m_data) {}

    teca_variant_array_impl(const teca_variant_array_impl<T> &other)
        : m_data(other.m_data) {}

    virtual teca_variant_array *new_copy() const override
    { return new teca_variant_array_impl<T>(*this); }

    teca_variant_array_impl<T> &
    operator=(const teca_variant_array_impl<T> &other)
    {
        m_data.assign(other.m_data.begin(), other.m_data.end());
        return *this;
    }

    template <typename U>
    teca_variant_array_impl<T> &
    operator=(const teca_variant_array_impl<U> &other)
    {
        m_data.assign(other.m_data.begin(), other.m_data.end());
        return *this;
    }

    // move
    teca_variant_array_impl(teca_variant_array_impl<T> &&other)
        : m_data(std::move(other.m_data)) {}

    teca_variant_array_impl<T> &
    operator=(teca_variant_array_impl<T> &&other)
    {
        m_data = std::move(other.m_data);
        return *this;
    }

    template <typename U>
    void get(U &val, unsigned int i=0) const
    { val = m_data[i]; }

    template <typename U>
    void get(U *beg, U *end) const TECA_NOEXCEPT
    {
        typename std::vector<T>::iterator data_it = m_data.begin();
        for (T *it = beg; it != end; ++it)
            *it = *data_it;
    }

    template <typename U>
    void get(std::vector<U> &val) const
    { val.assign(m_data.begin(), m_data.end()); }

    template <typename U>
    void set(const U &val, unsigned int i=0)
    { m_data[i] = val; }

    template <typename U>
    void set(U *beg, U *end)
    { m_data.assign(beg, end); }

    template <typename U>
    void set(const std::vector<U> &val)
    { m_data = val; }

    template <typename U>
    void append(const std::vector<U> &val)
    { std::copy(val.begin(), val.end(), std::back_inserter(m_data)); }

    template <typename U>
    void append(const U &val) { m_data.push_back(val); }

    virtual unsigned int size() const TECA_NOEXCEPT override
    { return m_data.size(); }

    virtual void resize(unsigned int n) override
    { m_data.resize(n); }

    virtual void clear() TECA_NOEXCEPT override
    { m_data.clear(); }

    void copy(const teca_variant_array &other)
    {
        TEMPLATE_DISPATCH(const teca_variant_array_impl, &other,
            TT *other_t = static_cast<TT*>(&other);
            *this = *other_t;
            return;
            )
         throw std::bad_cast();
    }

    virtual void swap(teca_variant_array &other) override
    {
        typedef teca_variant_array_impl<T> TT;
        TT *other_t = dynamic_cast<TT*>(&other);
        if (other_t)
        {
            this->m_data.swap(other_t->m_data);
            return;
        }
        throw std::bad_cast();
    }

    virtual bool equal(const teca_variant_array &other) override
    {
        typedef teca_variant_array_impl<T> TT;
        const TT *other_t = dynamic_cast<const TT*>(&other);
        if (other_t)
        {
            return this->m_data == other_t->m_data;
        }
        throw std::bad_cast();
        return false;
    }

private:
    std::vector<T> m_data;

    friend class teca_variant_array;
    template <typename U> friend class teca_variant_array_impl;
};

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::get(T &val, unsigned int i) const
{
    TEMPLATE_DISPATCH(const teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->get(val, i);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::get(std::vector<T> &vals) const
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
void teca_variant_array::set(const std::vector<T> &vals)
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
void teca_variant_array::set(const T &val, unsigned int i)
{
    TEMPLATE_DISPATCH(teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->set(val, i);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::append(const T &val)
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
void teca_variant_array::append(const std::vector<T> &vals)
{
    TEMPLATE_DISPATCH(teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->append(vals);
        return;
        )
    throw std::bad_cast();
}

#endif
