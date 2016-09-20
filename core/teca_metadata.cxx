#include <teca_metadata.h>
#include <utility>
#include <ostream>

using std::string;
using std::map;
using std::pair;
using std::vector;
using std::ostream;
using std::endl;

// --------------------------------------------------------------------------
teca_metadata::teca_metadata() noexcept
{
    this->id = this->get_next_id();
}

// --------------------------------------------------------------------------
teca_metadata::teca_metadata(const teca_metadata &other)
{
    *this = other;
}

// --------------------------------------------------------------------------
teca_metadata::teca_metadata(teca_metadata &&other) noexcept
    : id(other.id), props(std::move(other.props))
{}

// --------------------------------------------------------------------------
teca_metadata::~teca_metadata() noexcept
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
    {
        this->props[it->first]
             = it->second ? it->second->new_copy() : it->second;
    }

    return *this;
}

// --------------------------------------------------------------------------
teca_metadata &teca_metadata::operator=(teca_metadata &&other) noexcept
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
void teca_metadata::insert(
    const std::string &name,
    p_teca_variant_array prop_val)
{
    this->props[name] = prop_val;
}

// --------------------------------------------------------------------------
int teca_metadata::set(
    const std::string &name,
    p_teca_variant_array prop_val)
{
    pair<prop_map_t::iterator, bool> ret
        = this->props.insert(make_pair(name, prop_val));

    if (!ret.second)
    {
        TECA_ERROR(
            << "attempt to access non-existant property \""
            << name << "\" ignored!")
        return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
p_teca_variant_array teca_metadata::get(const std::string &name)
{
    prop_map_t::const_iterator it = this->props.find(name);

    if (it == this->props.end())
        return nullptr;

    return it->second;
}

// --------------------------------------------------------------------------
const_p_teca_variant_array teca_metadata::get(const std::string &name) const
{
    prop_map_t::const_iterator it = this->props.find(name);

    if (it == this->props.end())
        return nullptr;

    return it->second;
}

// --------------------------------------------------------------------------
int teca_metadata::get(
    const std::string &name, p_teca_variant_array vals) const
{
    prop_map_t::const_iterator it = this->props.find(name);

    if (it == this->props.end())
        return -1;

    vals->copy(it->second);

    return 0;
}

// --------------------------------------------------------------------------
int teca_metadata::size(
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
void teca_metadata::resize(const std::string &name, unsigned int n)
{
    prop_map_t::iterator it = this->props.find(name);
    if (it == this->props.end())
    {
        TECA_ERROR("attempt to access a non-existant property ignored!")
        return;
    }
    it->second->resize(n);
}

// --------------------------------------------------------------------------
int teca_metadata::remove(const std::string &name) noexcept
{
    prop_map_t::iterator it = this->props.find(name);

    if (it == this->props.end())
        return -1;

    this->props.erase(it);

    return 0;
}

// --------------------------------------------------------------------------
int teca_metadata::has(const std::string &name) const noexcept
{
    return this->props.count(name);
}

// --------------------------------------------------------------------------
int teca_metadata::empty() const noexcept
{
    return this->props.empty();
}

// --------------------------------------------------------------------------
unsigned long long teca_metadata::get_next_id() const noexcept
{
    static unsigned long long id = 0;
    return id++;
}

// --------------------------------------------------------------------------
void teca_metadata::to_stream(teca_binary_stream &s) const
{
    unsigned int n_props = this->props.size();
    s.pack(n_props);

    teca_metadata::prop_map_t::const_iterator it = this->props.cbegin();
    teca_metadata::prop_map_t::const_iterator end = this->props.cend();
    for (; it != end; ++it)
    {
        s.pack(it->first);
        s.pack(it->second->type_code());
        it->second->to_stream(s);
    }
}

// --------------------------------------------------------------------------
void teca_metadata::from_stream(teca_binary_stream &s)
{
    this->clear();

    unsigned int n_props;
    s.unpack(n_props);

    for (unsigned int i = 0; i < n_props; ++i)
    {
        string key;
        s.unpack(key);

        unsigned int type_code;
        s.unpack(type_code);

        p_teca_variant_array val
            = teca_variant_array_factory::New(type_code);

        val->from_stream(s);

        this->set(key, val);
    }
}

// --------------------------------------------------------------------------
void teca_metadata::to_stream(ostream &os) const
{
    prop_map_t::const_iterator it = this->props.cbegin();
    prop_map_t::const_iterator end = this->props.cend();
    for (; it != end; ++it)
    {
        os << it->first << " = " << "{";
        TEMPLATE_DISPATCH_CASE(
            teca_variant_array_impl, std::string, it->second.get(),
            const TT *val = static_cast<const TT*>(it->second.get());
            val->to_stream(os);
            )
        TEMPLATE_DISPATCH_CASE(
            teca_variant_array_impl, teca_metadata, it->second.get(),
            const TT *val = static_cast<const TT*>(it->second.get());
            val->to_stream(os);
            )
        TEMPLATE_DISPATCH_CASE(
            teca_variant_array_impl, p_teca_variant_array, it->second.get(),
            const TT *val = static_cast<const TT*>(it->second.get());
            val->to_stream(os);
            )
        TEMPLATE_DISPATCH(teca_variant_array_impl, it->second.get(),
            const TT *val = static_cast<const TT*>(it->second.get());
            val->to_stream(os);
            )
        os << "}" << endl;
    }
}

// --------------------------------------------------------------------------
bool operator<(const teca_metadata &lhs, const teca_metadata &rhs) noexcept
{
    return lhs.id < rhs.id;
}

// --------------------------------------------------------------------------
bool operator==(const teca_metadata &lhs, const teca_metadata &rhs) noexcept
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
