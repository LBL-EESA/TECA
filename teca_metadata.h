#ifndef teca_metadata_h
#define teca_metadata_h

#include <iosfwd>
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
    void set(const std::string &name, const T &val);

    // set from an array of length n
    template<typename T>
    void set(const std::string &name, const T *val, unsigned int n);

    // set from a vector
    template<typename T>
    void set(const std::string &name, const std::vector<T> &val);

    // set a variant array
    void set(
        const std::string &name,
        p_teca_variant_array prop_val);

    // get ith prop value. return 0 if successful
    template<typename T>
    int get(const std::string &name, T &val) const;

    // get n prop values to an array. see also get_size.
    // return 0 if successful
    template<typename T>
    int get(
        const std::string &name,
        T *val, unsigned int n) const;

    // copy prop values from the named prop into the passed in vector.
    // return 0 if successful
    template<typename T>
    int get(const std::string &name, std::vector<T> &val) const;

    // get the length of the property value. return 0 if successful
    int get_size(
        const std::string &name,
        unsigned int &size) const TECA_NOEXCEPT;

    // remove. return 0 if successful
    int remove(const std::string &name) TECA_NOEXCEPT;

    // remove all
    void clear();

    // returns true if there is a property with the given
    // name already in the container.
    int has(const std::string &name) const TECA_NOEXCEPT;

    // return true if empty
    int empty() const TECA_NOEXCEPT;

    // return true if not empty
    explicit operator bool() const TECA_NOEXCEPT
    { return !empty(); }

    // serialize to/from binary
    void to_stream(teca_binary_stream &s) const;
    void from_stream(teca_binary_stream &s);

    // serialize to/from ascii
    void to_stream(std::ostream &os) const;
    void from_stream(std::ostream &) {}

private:
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
void teca_metadata::set(const std::string &name, const T &val)
{
    p_teca_variant_array prop_val
        = teca_variant_array_impl<T>::New(1);

    prop_val->set(0, val);

    this->set(name, prop_val);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_metadata::set(
    const std::string &name,
    const T *vals,
    unsigned int n_vals)
{
    p_teca_variant_array prop_val
        = teca_variant_array_impl<T>::New(vals, n_vals);

    this->set(name, prop_val);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_metadata::set(
    const std::string &name,
    const std::vector<T> &vals)
{
    unsigned int n_vals = vals.size();

    p_teca_variant_array prop_val
        = teca_variant_array_impl<T>::New(&vals[0], n_vals);

    this->set(name, prop_val);
}

// --------------------------------------------------------------------------
template<typename T>
int teca_metadata::get(const std::string &name, T &val) const
{
    prop_map_t::const_iterator it = this->props.find(name);

    if (it == this->props.end())
        return -1;

    it->second->get(0, val);

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
