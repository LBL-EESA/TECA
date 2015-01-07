#ifndef teca_meta_data_h
#define teca_meta_data_h

#include <memory>
#include <map>
#include <string>
#include <vector>
#include "teca_variant_array.h"

// a generic container for meta data in the form
// of name=value pairs. value arrays are supported.
// see meta data producer-consumer documentation for
// information about what names are valid.
class teca_meta_data
{
public:
    teca_meta_data();
    teca_meta_data(const teca_meta_data &other);
    teca_meta_data(teca_meta_data &&other);
    ~teca_meta_data();
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

    // get prop value to a scalar
    template<typename T>
    int get_prop(const std::string &name, T &val);
    // get all values to an array. see also get_prop_size
    template<typename T>
    int get_prop(const std::string &name, T *val);
    // get to a vector
    template<typename T>
    int get_prop(const std::string &name, std::vector<T> &val);

    // get the length of the property value
    int get_prop_size(const std::string &name, unsigned int &size);

    // remove
    int remove_prop(const std::string &name);

    // remove all
    void clear_props();

    // returns true if their is a property with the given
    // name already in the container.
    int has_prop(const std::string &name);

    // retunr true if empty
    int empty();

private:
    void set_prop(
        const std::string &name,
        teca_variant_array *prop_val);

    unsigned long long get_next_id();

private:
    unsigned long long id;
    typedef std::map<std::string, teca_variant_array*> prop_map_t;
    prop_map_t props;

    friend bool operator<(
        const teca_meta_data &lhs,
        const teca_meta_data &rhs);
};

typedef std::shared_ptr<teca_meta_data> p_teca_meta_data;

// comparison function so that metadata can be
// used as a key in std::map.
bool operator<(
    const teca_meta_data &lhs,
    const teca_meta_data &rhs);

#endif
