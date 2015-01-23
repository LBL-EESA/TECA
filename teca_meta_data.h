#ifndef teca_meta_data_h
#define teca_meta_data_h

#include <memory>
#include <map>
#include <string>
#include <vector>
#include "teca_variant_array.h"

// TODO --
// add api fpor nesting
// fix perfect forwarding and moving

// a generic container for meta data in the form
// of name=value pairs. value arrays are supported.
// see meta data producer-consumer documentation for
// information about what names are valid.
class teca_meta_data
{
public:
    teca_meta_data();
    ~teca_meta_data();
    teca_meta_data(const teca_meta_data &other);
    teca_meta_data(teca_meta_data &&other);
    teca_meta_data &operator=(const teca_meta_data &other);
    teca_meta_data &operator=(teca_meta_data &&other);

    // set from a scalar property
    template<typename T>
    void set_prop(const std::string &name, const T &val);

    // set from an array of length n
    template<typename T>
    void set_prop(const std::string &name, const T *val, unsigned int n);

    // set from a vector
    template<typename T>
    void set_prop(const std::string &name, const std::vector<T> &val);

    // get ith prop value (default to 0th)
    template<typename T>
    int get_prop(const std::string &name, T &val) const;

    // get n prop values to an array. see also get_prop_size
    template<typename T>
    int get_prop(const std::string &name, T *val, unsigned int n) const;

    // get all prop values to a vector
    template<typename T>
    int get_prop(const std::string &name, std::vector<T> &val) const;

    // get the length of the property value
    int get_prop_size(const std::string &name, unsigned int &size) const;

    // remove
    int remove_prop(const std::string &name);

    // remove all
    void clear_props();

    // returns true if their is a property with the given
    // name already in the container.
    int has(const std::string &name) const;

    // return true if empty
    int empty() const;

    explicit operator bool() const
    { return !empty(); }

private:
    void set_prop(
        const std::string &name,
        teca_variant_array *prop_val);

    unsigned long long get_next_id();

private:
    unsigned long long id;
    typedef std::map<std::string, teca_variant_array*> prop_map_t;
    prop_map_t props;

    friend bool operator<(const teca_meta_data &, const teca_meta_data &);
    friend bool operator==(const teca_meta_data &, const teca_meta_data &);
    friend teca_meta_data operator&(const teca_meta_data &, const teca_meta_data &);
};

typedef std::shared_ptr<teca_meta_data> p_teca_meta_data;

// comparison function so that metadata can be
// used as a key in std::map.
bool operator<(const teca_meta_data &lhs, const teca_meta_data &rhs);

// compare meta data objects. two objects are considered
// equal if both have the same set of keys and all of the values
// are equal
bool operator==(const teca_meta_data &lhs, const teca_meta_data &rhs);

// intersect two metadata objects. return a new object with
// common key value pairs
teca_meta_data operator&(const teca_meta_data &lhs, const teca_meta_data &rhs);

// --------------------------------------------------------------------------
template<typename T>
void teca_meta_data::set_prop(const std::string &name, const T &val)
{
    teca_variant_array *prop_val
        = new teca_variant_array_impl<T>(1);

    prop_val->set(val, 0);

    this->set_prop(name, prop_val);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_meta_data::set_prop(
    const std::string &name,
    const T *val,
    unsigned int n)
{
    teca_variant_array *prop_val
        = new teca_variant_array_impl<T>(val, n);

    this->set_prop(name, prop_val);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_meta_data::set_prop(
    const std::string &name,
    const std::vector<T> &val)
{
    unsigned int n_vals = val.size();

    teca_variant_array *prop_val
        = new teca_variant_array_impl<T>(n_vals);

    for (unsigned int i = 0; i < n_vals; ++i)
        prop_val->set(val[i], i);

    this->set_prop(name, prop_val);
}

// --------------------------------------------------------------------------
template<typename T>
int teca_meta_data::get_prop(const std::string &name, T &val) const
{
    prop_map_t::const_iterator it = this->props.find(name);

    if (it == this->props.end())
        return -1;

    it->second->get(val);

    return 0;
}

// --------------------------------------------------------------------------
template<typename T>
int teca_meta_data::get_prop(
    const std::string &name,
    std::vector<T> &vals)
     const
{
    prop_map_t::const_iterator it = this->props.find(name);

    if (it == this->props.end())
        return -1;

    it->second->get(vals);

    return 0;
}

// --------------------------------------------------------------------------
template<typename T>
int teca_meta_data::get_prop(const std::string &name, T *vals, unsigned int n) const
{
    prop_map_t::const_iterator it = this->props.find(name);

    if (it == this->props.end())
        return -1;

    it->second->get(vals, vals+n);

    return 0;
}
#endif
