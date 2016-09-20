#ifndef teca_metadata_h
#define teca_metadata_h

#include <iosfwd>
#include <map>
#include <string>
#include <initializer_list>
#include <vector>
#include <set>
#include "teca_variant_array.h"

// a generic container for meta data in the form
// of name=value pairs. value arrays are supported.
// see meta data producer-consumer documentation for
// information about what names are valid.
class teca_metadata
{
public:
    teca_metadata() noexcept;
    ~teca_metadata() noexcept;

    teca_metadata(const teca_metadata &other);
    teca_metadata &operator=(const teca_metadata &other);

    teca_metadata(teca_metadata &&other) noexcept;
    teca_metadata &operator=(teca_metadata &&other) noexcept;

    // get the length of the named property. return 0 if successful
    int size(
        const std::string &name,
        unsigned int &size) const noexcept;

    // resize the named property
    void resize(const std::string &name, unsigned int n);

    // declare a property. properties must be declared or inserted
    // before accessing them via set/get.
    template<typename T>
    void declare(const std::string &name);

    template<typename T>
    void declare(const std::string &name, unsigned int n);

    // insert a scalar value. if the property doesn't exist
    // it is created. if it does it is replaced.
    template<typename T>
    void insert(const std::string &name, const T &val);

    // insert an array of length n. if the property doesn't exist
    // it is created. if it does it is replaced.
    template<typename T>
    void insert(const std::string &name, const T *val, unsigned int n);

    // insert a set. if the property doesn't exist
    // it is created. if it does it is replaced.
    template<typename T>
    void insert(const std::string &name, const std::set<T> &val);

    // insert a vector. if the property doesn't exist
    // it is created. if it does it is replaced.
    template<typename T>
    void insert(const std::string &name, const std::vector<T> &val);

    template<typename T>
    void insert(const std::string &name, std::initializer_list<T> val);

    // insert a vector of vectors. if the property doesn't exist
    // it is created. if it does it is replaced.
    template<typename T>
    void insert(const std::string &name, const std::vector<std::vector<T>> &val);

    // insert a variant array directly. if the property doesn't exist
    // it is created. if it does it is replaced.
    void insert(const std::string &name,
        p_teca_variant_array prop_val);

    // append a value to the named property. reports
    // an error and does nothing if the property doesn't
    // exist. return 0 on success.
    template<typename T>
    int append(const std::string &name, const T &val);

    // set a scalar. replaces the current value and does
    // nothing if the property doesn't exist. return 0
    // on success.
    template<typename T>
    int set(const std::string &name, const T &val)
    { return this->set<T>(name, 0, val); }

    // set the ith value from a scalar. replaces the current value and does
    // nothing if the property doesn't exist. return 0
    // on success.
    template<typename T>
    int set(const std::string &name, unsigned int i, const T &val);

    // set an array of length n. replaces the current value and does
    // nothing if the property doesn't exist. return 0
    // on success.
    template<typename T>
    int set(const std::string &name, const T *val, unsigned int n);

    // set a vector. replaces the current value and does
    // nothing if the property doesn't exist. return 0
    // on success.
    template<typename T>
    int set(const std::string &name, const std::vector<T> &val);

    template<typename T>
    int set(const std::string &name, std::initializer_list<T> val);

    // set a set. replaces the current value and does
    // nothing if the property doesn't exist. return 0
    // on success.
    template<typename T>
    int set(const std::string &name, const std::set<T> &val);

    // set a variant array directly. replaces the current value and does
    // nothing if the property doesn't exist. return 0
    // on success.
    int set(const std::string &name,
        p_teca_variant_array prop_val);

    // get prop value. return 0 if successful
    template<typename T>
    int get(const std::string &name, T &val) const
    { return this->get<T>(name, 0, val); }

    // get ith prop value. return 0 if successful
    template<typename T>
    int get(const std::string &name, unsigned int i, T &val) const;

    // get n prop values to an array. see also get_size.
    // return 0 if successful
    template<typename T>
    int get(const std::string &name,
        T *val, unsigned int n) const;

    // copy prop values from the named prop into the passed in vector.
    // return 0 if successful
    template<typename T>
    int get(const std::string &name, std::vector<T> &val) const;

    // copy prop values from the named prop into the passed in set.
    // return 0 if successful
    template<typename T>
    int get(const std::string &name, std::set<T> &val) const;

    // copy prop values from the named prop into the passed in
    // array. return 0 if successful
    int get(const std::string &name, p_teca_variant_array val) const;

    // get the variant array, or nullptr if the
    // property doesn't exist
    p_teca_variant_array get(const std::string &name);
    const_p_teca_variant_array get(const std::string &name) const;

    // remove. return 0 if successful
    int remove(const std::string &name) noexcept;

    // remove all
    void clear();

    // returns true if there is a property with the given
    // name already in the container.
    int has(const std::string &name) const noexcept;

    // return true if empty
    int empty() const noexcept;

    // return true if not empty
    explicit operator bool() const noexcept
    { return !empty(); }

    // serialize to/from binary
    void to_stream(teca_binary_stream &s) const;
    void from_stream(teca_binary_stream &s);

    // serialize to/from ascii
    void to_stream(std::ostream &os) const;
    void from_stream(std::ostream &) {}

private:
    unsigned long long get_next_id() const noexcept;

private:
    unsigned long long id;
    using prop_map_t = std::map<std::string, p_teca_variant_array>;
    prop_map_t props;

    friend bool operator<(const teca_metadata &, const teca_metadata &) noexcept;
    friend bool operator==(const teca_metadata &, const teca_metadata &) noexcept;
    friend teca_metadata operator&(const teca_metadata &, const teca_metadata &);
};

// comparison function so that metadata can be
// used as a key in std::map.
bool operator<(const teca_metadata &lhs, const teca_metadata &rhs) noexcept;

// compare meta data objects. two objects are considered
// equal if both have the same set of keys and all of the values
// are equal
bool operator==(const teca_metadata &lhs, const teca_metadata &rhs) noexcept;

inline
bool operator!=(const teca_metadata &lhs, const teca_metadata &rhs) noexcept
{ return !(lhs == rhs); }

// intersect two metadata objects. return a new object with
// common key value pairs
teca_metadata operator&(const teca_metadata &lhs, const teca_metadata &rhs);

// --------------------------------------------------------------------------
template<typename T>
void teca_metadata::declare(const std::string &name)
{
    p_teca_variant_array prop_val
        = teca_variant_array_impl<T>::New();

    this->set(name, prop_val);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_metadata::declare(const std::string &name, unsigned int n)
{
    p_teca_variant_array prop_val
        = teca_variant_array_impl<T>::New(n);

    this->set(name, prop_val);
}

// --------------------------------------------------------------------------
template<typename T>
int teca_metadata::append(const std::string &name, const T &val)
{
    prop_map_t::iterator it = this->props.find(name);
    if (it == this->props.end())
    {
        return -1;
    }

    it->second->append(val);

    return 0;
}

// --------------------------------------------------------------------------
template<typename T>
void teca_metadata::insert(const std::string &name, const T &val)
{
    p_teca_variant_array prop_val
        = teca_variant_array_impl<T>::New(&val, 1);

    this->props[name] = prop_val;
}

// --------------------------------------------------------------------------
template<typename T>
void teca_metadata::insert(
    const std::string &name,
    const T *vals,
    unsigned int n_vals)
{
    p_teca_variant_array prop_val
        = teca_variant_array_impl<T>::New(vals, n_vals);

    this->props[name] = prop_val;
}

// --------------------------------------------------------------------------
template<typename T>
void teca_metadata::insert(
    const std::string &name, const std::set<T> &vals)
{
    size_t n = vals.size();

    std::vector<T> tmp(vals.begin(), vals.end());

    p_teca_variant_array prop_val
        = teca_variant_array_impl<T>::New(tmp.data(), n);

    this->props[name] = prop_val;
}

// --------------------------------------------------------------------------
template<typename T>
void teca_metadata::insert(
    const std::string &name,
    std::initializer_list<T> vals)
{
    this->insert(name, std::vector<T>(vals));
}

// --------------------------------------------------------------------------
template<typename T>
void teca_metadata::insert(
    const std::string &name,
    const std::vector<T> &vals)
{
    size_t n = vals.size();

    p_teca_variant_array prop_val
        = teca_variant_array_impl<T>::New(vals.data(), n);

    this->props[name] = prop_val;
}

// --------------------------------------------------------------------------
template<typename T>
void teca_metadata::insert(
    const std::string &name,
    const std::vector<std::vector<T>> &vals)
{
    size_t n = vals.size();

    p_teca_variant_array prop_vals
        = teca_variant_array_impl<p_teca_variant_array>::New();

    for (size_t i = 0; i < n; ++i)
    {
        p_teca_variant_array prop_val
            = teca_variant_array_impl<T>::New((vals.at(i).data()), vals.at(i).size());
        prop_vals->append(prop_val);
    }

    this->props[name] = prop_vals;
}


// --------------------------------------------------------------------------
template<typename T>
int teca_metadata::set(const std::string &name, unsigned int i, const T &val)
{
    prop_map_t::iterator it = this->props.find(name);
    if (it == this->props.end())
    {
        TECA_ERROR(
            << "attempt to access non-existant property \""
            << name << "\" ignored!")
        return -1;
    }

    it->second->set(i, val);

    return 0;
}

// --------------------------------------------------------------------------
template<typename T>
int teca_metadata::set(
    const std::string &name,
    const T *vals,
    unsigned int n_vals)
{
    prop_map_t::iterator it = this->props.find(name);
    if (it == this->props.end())
    {
        TECA_ERROR(
            << "attempt to access non-existant property \""
            << name << "\" ignored!")
        return -1;
    }

    it->second->set(0, n_vals-1, vals);

    return 0;
}

// --------------------------------------------------------------------------
template<typename T>
int teca_metadata::set(
    const std::string &name,
    const std::vector<T> &vals)
{
    prop_map_t::iterator it = this->props.find(name);
    if (it == this->props.end())
    {
        TECA_ERROR(
            << "attempt to access non-existant property \""
            << name << "\" ignored!")
        return -1;
    }

    it->second->set(vals);

    return 0;
}

// --------------------------------------------------------------------------
template<typename T>
int teca_metadata::set(
    const std::string &name,
    std::initializer_list<T> vals)
{
    return this->set(name, std::vector<T>(vals));
}

// --------------------------------------------------------------------------
template<typename T>
int teca_metadata::set(
    const std::string &name,
    const std::set<T> &vals)
{
    prop_map_t::iterator it = this->props.find(name);
    if (it == this->props.end())
    {
        TECA_ERROR(
            << "attempt to access non-existant property \""
            << name << "\" ignored!")
        return -1;
    }

    std::vector<T> tmp(vals.begin(), vals.end());
    it->second->set(tmp);

    return 0;
}

// --------------------------------------------------------------------------
template<typename T>
int teca_metadata::get(const std::string &name, unsigned int i, T &val) const
{
    prop_map_t::const_iterator it = this->props.find(name);

    if (it == this->props.end())
        return -1;

    it->second->get(i, val);

    return 0;
}

// --------------------------------------------------------------------------
template<typename T>
int teca_metadata::get(
    const std::string &name,
    std::vector<T> &vals) const
{
    prop_map_t::const_iterator it = this->props.find(name);

    if (it == this->props.end())
        return -1;

    it->second->get(vals);

    return 0;
}

// --------------------------------------------------------------------------
template<typename T>
int teca_metadata::get(
    const std::string &name,
    std::set<T> &vals) const
{
    std::vector<T> tmp;
    if (!this->get(name, tmp))
    {
        vals = std::set<T>(tmp.begin(), tmp.end());
        return 0;
    }
    return -1;
}

// --------------------------------------------------------------------------
template<typename T>
int teca_metadata::get(
    const std::string &name,
    T *vals, unsigned int n) const
{
    prop_map_t::const_iterator it = this->props.find(name);

    if (it == this->props.end())
        return -1;

    it->second->get(0, n-1, vals);

    return 0;
}

// convenience defs for nesting metadata
using teca_metadata_array = teca_variant_array_impl<teca_metadata>;
#endif
