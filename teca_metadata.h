#ifndef teca_metadata_h
#define teca_metadata_h

#include <map>
#include <string>
#include <vector>
#include "teca_variant_array.h"
#include "teca_compiler.h"

// a generic container for meta data in the form
// of name=value pairs. value arrays are supported.
// see meta data producer-consumer documentation for
// information about what names are valid.
class teca_metadata
{
public:
    teca_metadata() TECA_NOEXCEPT;
    ~teca_metadata() TECA_NOEXCEPT;

    teca_metadata(const teca_metadata &other);
    teca_metadata &operator=(const teca_metadata &other);

    teca_metadata(teca_metadata &&other) TECA_NOEXCEPT;
    teca_metadata &operator=(teca_metadata &&other) TECA_NOEXCEPT;

    // set from a scalar property
    template<typename T>
    void set_prop(const std::string &name, const T &val);

    // set from an array of length n
    template<typename T>
    void set_prop(const std::string &name, const T *val, unsigned int n);

    // set from a vector
    template<typename T>
    void set_prop(const std::string &name, const std::vector<T> &val);

    // get ith prop value. return 0 if successful
    template<typename T>
    int get_prop(const std::string &name, T &val) const;

    // get n prop values to an array. see also get_prop_size.
    // return 0 if successful
    template<typename T>
    int get_prop(
        const std::string &name,
        T *val, unsigned int n) const;

    // copy prop values from the named prop into the passed in vector.
    // return 0 if successful
    template<typename T>
    int get_prop(const std::string &name, std::vector<T> &val) const;

    // get the length of the property value. return 0 if successful
    int get_prop_size(
        const std::string &name,
        unsigned int &size) const TECA_NOEXCEPT;

    // remove. return 0 if successful
    int remove_prop(const std::string &name) TECA_NOEXCEPT;

    // remove all
    void clear_props();

    // returns true if there is a property with the given
    // name already in the container.
    int has(const std::string &name) const TECA_NOEXCEPT;

    // return true if empty
    int empty() const TECA_NOEXCEPT;

    // return true if not empty
    explicit operator bool() const TECA_NOEXCEPT
    { return !empty(); }

private:
    void set_prop(
        const std::string &name,
        p_teca_variant_array prop_val);

    unsigned long long get_next_id() const TECA_NOEXCEPT;

private:
    unsigned long long id;
    using prop_map_t = std::map<std::string, p_teca_variant_array>;
    prop_map_t props;

    friend bool operator<(const teca_metadata &, const teca_metadata &) TECA_NOEXCEPT;
    friend bool operator==(const teca_metadata &, const teca_metadata &) TECA_NOEXCEPT;
    friend teca_metadata operator&(const teca_metadata &, const teca_metadata &);
};

// comparison function so that metadata can be
// used as a key in std::map.
bool operator<(const teca_metadata &lhs, const teca_metadata &rhs) TECA_NOEXCEPT;

// compare meta data objects. two objects are considered
// equal if both have the same set of keys and all of the values
// are equal
bool operator==(const teca_metadata &lhs, const teca_metadata &rhs) TECA_NOEXCEPT;

// intersect two metadata objects. return a new object with
// common key value pairs
teca_metadata operator&(const teca_metadata &lhs, const teca_metadata &rhs);

// --------------------------------------------------------------------------
template<typename T>
void teca_metadata::set_prop(const std::string &name, const T &val)
{
    p_teca_variant_array prop_val
        = teca_variant_array_impl<T>::New(1);

    prop_val->set(val, 0);

    this->set_prop(name, prop_val);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_metadata::set_prop(
    const std::string &name,
    const T *val,
    unsigned int n)
{
    p_teca_variant_array prop_val
        = teca_variant_array_impl<T>::New(val, n);

    this->set_prop(name, prop_val);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_metadata::set_prop(
    const std::string &name,
    const std::vector<T> &val)
{
    unsigned int n_vals = val.size();

    p_teca_variant_array prop_val
        = teca_variant_array_impl<T>::New(n_vals);

    for (unsigned int i = 0; i < n_vals; ++i)
        prop_val->set(val[i], i);

    this->set_prop(name, prop_val);
}

// --------------------------------------------------------------------------
template<typename T>
int teca_metadata::get_prop(const std::string &name, T &val) const
{
    prop_map_t::const_iterator it = this->props.find(name);

    if (it == this->props.end())
        return -1;

    it->second->get(val);

    return 0;
}

// --------------------------------------------------------------------------
template<typename T>
int teca_metadata::get_prop(
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
int teca_metadata::get_prop(
    const std::string &name,
    T *vals, unsigned int n) const
{
    prop_map_t::const_iterator it = this->props.find(name);

    if (it == this->props.end())
        return -1;

    it->second->get(vals, vals+n);

    return 0;
}

// convenience defs for nesting metadata
using teca_metadata_array = teca_variant_array_impl<teca_metadata>;
#endif
