#include <teca_meta_data.h>
#include <utility>

using std::string;
using std::map;
using std::pair;
using std::vector;

// --------------------------------------------------------------------------
teca_meta_data::teca_meta_data()
{
    this->id = this->get_next_id();
}

// --------------------------------------------------------------------------
teca_meta_data::teca_meta_data(const teca_meta_data &other)
{
    *this = other;
}

// --------------------------------------------------------------------------
teca_meta_data::teca_meta_data(teca_meta_data &&other) :
    id(other.id),
    props(std::move(other.props))
{}

// --------------------------------------------------------------------------
teca_meta_data::~teca_meta_data()
{
    this->clear_props();
}

// --------------------------------------------------------------------------
teca_meta_data &teca_meta_data::operator=(const teca_meta_data &other)
{
    if (&other == this) return *this;
    this->id = other.id;

    this->clear_props();

    prop_map_t::const_iterator it = other.props.begin();
    prop_map_t::const_iterator end = other.props.end();
    for ( ; it != end; ++it)
        this->props[it->first] = it->second->new_copy();

    return *this;
}

// --------------------------------------------------------------------------
teca_meta_data &teca_meta_data::operator=(teca_meta_data &&other)
{
    if (&other == this) return *this;
    this->id = other.id;
    this->props = std::move(other.props);
    return *this;
}

// --------------------------------------------------------------------------
void teca_meta_data::clear_props()
{
    prop_map_t::iterator it = this->props.begin();
    prop_map_t::iterator end = this->props.end();
    for ( ; it != end; ++it)
    {
        delete it->second;
    }
    this->props.clear();
}

// --------------------------------------------------------------------------
void teca_meta_data::set_prop(
    const std::string &name,
    teca_variant_array *prop_val)
{
    pair<prop_map_t::iterator, bool> ret
        = this->props.insert(make_pair(name, prop_val));

    if (!ret.second)
    {
        delete ret.first->second;
        ret.first->second = prop_val;
    }
}

// --------------------------------------------------------------------------
template<typename T>
void teca_meta_data::set_prop(const std::string &name, const T &val)
{
    this->set_prop(name, val, 1);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_meta_data::set_prop(
    const std::string &name,
    const T *val,
    unsigned int n)
{
    teca_variant_array *prop_val
        = new teca_variant_array_impl<T>(&val, n);

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
        prop_val->set(i, val[i]);

    this->set_prop(name, prop_val);
}

// --------------------------------------------------------------------------
template<typename T>
int teca_meta_data::get_prop(const std::string &name, T &val)
{
    prop_map_t::iterator it = this->props.find(name);

    if (it == this->props.end())
        return -1;

    it->second->get(val);
    return 0;
}

// --------------------------------------------------------------------------
template<typename T>
int teca_meta_data::get_prop(const std::string &name, vector<T> &vals)
{
    prop_map_t::iterator it = this->props.find(name);

    if (it == this->props.end())
        return -1;

    T *data;
    unsigned int size;
    it->second->get_data(data, size);

    vals.assign(data, data+size);
    return 0;
}

// --------------------------------------------------------------------------
template<typename T>
int teca_meta_data::get_prop(const std::string &name, T *vals)
{
    prop_map_t::iterator it = this->props.find(name);

    if (it == this->props.end())
        return -1;

    it->second->get(vals);
    return 0;
}

// --------------------------------------------------------------------------
int teca_meta_data::get_prop_size(const std::string &name, unsigned int &n)
{
    prop_map_t::iterator it = this->props.find(name);

    if (it == this->props.end())
        return -1;

    n = it->second->size();

    return 0;
}

// --------------------------------------------------------------------------
int teca_meta_data::remove_prop(const std::string &name)
{
    prop_map_t::iterator it = this->props.find(name);

    if (it == this->props.end())
        return -1;

    delete it->second;
    this->props.erase(it);

    return 0;
}

// --------------------------------------------------------------------------
int teca_meta_data::has_prop(const std::string &name)
{
    return this->props.count(name);
}

// --------------------------------------------------------------------------
int teca_meta_data::empty()
{
    return this->props.size();
}

// --------------------------------------------------------------------------
unsigned long long teca_meta_data::get_next_id()
{
    static unsigned long long id = 0;
    return id++;
}

// --------------------------------------------------------------------------
bool operator<(
    const teca_meta_data &lhs,
    const teca_meta_data &rhs)
{
    return lhs.id < rhs.id;
}
