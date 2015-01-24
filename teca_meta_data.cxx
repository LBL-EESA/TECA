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
teca_meta_data::teca_meta_data(teca_meta_data &&other) noexcept
    : id(other.id), props(std::move(other.props))
{}

// --------------------------------------------------------------------------
teca_meta_data::~teca_meta_data() noexcept
{
    this->clear_props();
}

// --------------------------------------------------------------------------
teca_meta_data &teca_meta_data::operator=(const teca_meta_data &other)
{
    if (&other == this)
        return *this;

    this->id = other.id;
    this->clear_props();

    prop_map_t::const_iterator it = other.props.begin();
    prop_map_t::const_iterator end = other.props.end();
    for ( ; it != end; ++it)
        this->props[it->first] = it->second->new_copy();

    return *this;
}

// --------------------------------------------------------------------------
teca_meta_data &teca_meta_data::operator=(teca_meta_data &&other) noexcept
{
    if (&other == this)
        return *this;

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
int teca_meta_data::get_prop_size(
    const std::string &name,
    unsigned int &n) const noexcept
{
    prop_map_t::const_iterator it = this->props.find(name);

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
int teca_meta_data::has(const std::string &name) const noexcept
{
    return this->props.count(name);
}

// --------------------------------------------------------------------------
int teca_meta_data::empty() const noexcept
{
    return this->props.empty();
}

// --------------------------------------------------------------------------
unsigned long long teca_meta_data::get_next_id() const noexcept
{
    static unsigned long long id = 0;
    return id++;
}

// --------------------------------------------------------------------------
bool operator<(const teca_meta_data &lhs, const teca_meta_data &rhs) noexcept
{
    return lhs.id < rhs.id;
}

// --------------------------------------------------------------------------
bool operator==(const teca_meta_data &lhs, const teca_meta_data &rhs) noexcept
{
    teca_meta_data::prop_map_t::const_iterator rit = rhs.props.begin();
    teca_meta_data::prop_map_t::const_iterator rend = rhs.props.end();
    teca_meta_data::prop_map_t::const_iterator lend = lhs.props.end();
    for (; rit != rend; ++rit)
    {
        teca_meta_data::prop_map_t::const_iterator lit = lhs.props.find(rit->first);
        if ((lit == lend) || !(*lit->second == *rit->second))
            return false;
    }
    return true;
}

// --------------------------------------------------------------------------
teca_meta_data operator&(const teca_meta_data &lhs, const teca_meta_data &rhs)
{
    teca_meta_data isect;
    teca_meta_data::prop_map_t::const_iterator rit = rhs.props.begin();
    teca_meta_data::prop_map_t::const_iterator rend = rhs.props.end();
    teca_meta_data::prop_map_t::const_iterator lend = lhs.props.end();
    for (; rit != rend; ++rit)
    {
        teca_meta_data::prop_map_t::const_iterator lit = lhs.props.find(rit->first);
        if ((lit != lend) && (*lit->second == *rit->second))
        {
            isect.set_prop(rit->first, rit->second->new_copy());
        }
    }
    return isect;
}
