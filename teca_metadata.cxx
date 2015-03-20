#include <teca_metadata.h>
#include <utility>

using std::string;
using std::map;
using std::pair;
using std::vector;

// --------------------------------------------------------------------------
teca_metadata::teca_metadata() TECA_NOEXCEPT
{
    this->id = this->get_next_id();
}

// --------------------------------------------------------------------------
teca_metadata::teca_metadata(const teca_metadata &other)
{
    *this = other;
}

// --------------------------------------------------------------------------
teca_metadata::teca_metadata(teca_metadata &&other) TECA_NOEXCEPT
    : id(other.id), props(std::move(other.props))
{}

// --------------------------------------------------------------------------
teca_metadata::~teca_metadata() TECA_NOEXCEPT
{
    this->clear();
}

// --------------------------------------------------------------------------
teca_metadata &teca_metadata::operator=(const teca_metadata &other)
{
    if (&other == this)
        return *this;

    this->id = other.id;
    this->clear();

    prop_map_t::const_iterator it = other.props.begin();
    prop_map_t::const_iterator end = other.props.end();
    for ( ; it != end; ++it)
        this->props[it->first] = it->second->new_copy();

    return *this;
}

// --------------------------------------------------------------------------
teca_metadata &teca_metadata::operator=(teca_metadata &&other) TECA_NOEXCEPT
{
    if (&other == this)
        return *this;

    this->id = other.id;
    this->props = std::move(other.props);
    return *this;
}

// --------------------------------------------------------------------------
void teca_metadata::clear()
{
    this->props.clear();
}

// --------------------------------------------------------------------------
void teca_metadata::set(
    const std::string &name,
    p_teca_variant_array prop_val)
{
    pair<prop_map_t::iterator, bool> ret
        = this->props.insert(make_pair(name, prop_val));

    if (!ret.second)
    {
        ret.first->second = prop_val;
    }
}

// --------------------------------------------------------------------------
int teca_metadata::get_size(
    const std::string &name,
    unsigned int &n) const TECA_NOEXCEPT
{
    prop_map_t::const_iterator it = this->props.find(name);

    if (it == this->props.end())
        return -1;

    n = it->second->size();

    return 0;
}

// --------------------------------------------------------------------------
int teca_metadata::remove(const std::string &name) TECA_NOEXCEPT
{
    prop_map_t::iterator it = this->props.find(name);

    if (it == this->props.end())
        return -1;

    this->props.erase(it);

    return 0;
}

// --------------------------------------------------------------------------
int teca_metadata::has(const std::string &name) const TECA_NOEXCEPT
{
    return this->props.count(name);
}

// --------------------------------------------------------------------------
int teca_metadata::empty() const TECA_NOEXCEPT
{
    return this->props.empty();
}

// --------------------------------------------------------------------------
unsigned long long teca_metadata::get_next_id() const TECA_NOEXCEPT
{
    static unsigned long long id = 0;
    return id++;
}

// --------------------------------------------------------------------------
bool operator<(const teca_metadata &lhs, const teca_metadata &rhs) TECA_NOEXCEPT
{
    return lhs.id < rhs.id;
}

// --------------------------------------------------------------------------
bool operator==(const teca_metadata &lhs, const teca_metadata &rhs) TECA_NOEXCEPT
{
    teca_metadata::prop_map_t::const_iterator rit = rhs.props.begin();
    teca_metadata::prop_map_t::const_iterator rend = rhs.props.end();
    teca_metadata::prop_map_t::const_iterator lend = lhs.props.end();
    for (; rit != rend; ++rit)
    {
        teca_metadata::prop_map_t::const_iterator lit = lhs.props.find(rit->first);
        if ((lit == lend) || !(*lit->second == *rit->second))
            return false;
    }
    return true;
}

// --------------------------------------------------------------------------
teca_metadata operator&(const teca_metadata &lhs, const teca_metadata &rhs)
{
    teca_metadata isect;
    teca_metadata::prop_map_t::const_iterator rit = rhs.props.begin();
    teca_metadata::prop_map_t::const_iterator rend = rhs.props.end();
    teca_metadata::prop_map_t::const_iterator lend = lhs.props.end();
    for (; rit != rend; ++rit)
    {
        teca_metadata::prop_map_t::const_iterator lit = lhs.props.find(rit->first);
        if ((lit != lend) && (*lit->second == *rit->second))
        {
            isect.set(rit->first, rit->second->new_copy());
        }
    }
    return isect;
}
