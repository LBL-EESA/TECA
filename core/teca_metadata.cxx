#include "teca_metadata.h"
#include "teca_variant_array_impl.h"

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
int teca_metadata::set(const std::string &name, const p_teca_variant_array &prop_val)
{
    this->props[name] = prop_val;
    return 0;
}

// --------------------------------------------------------------------------
int teca_metadata::update(const std::string &name, p_teca_variant_array prop_val)
{
    prop_map_t::iterator it = this->props.find(name);
    if (it == this->props.end())
    {
        TECA_ERROR("attempt to access non-existent property \""
            << name << "\" ignored!")
        return -1;
    }

    it->second = prop_val;

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
int teca_metadata::get(const std::string &name, p_teca_variant_array vals) const
{
    prop_map_t::const_iterator it = this->props.find(name);

    if (it == this->props.end())
        return -1;

    vals->assign(it->second);

    return 0;
}

// --------------------------------------------------------------------------
int teca_metadata::get_name(unsigned long i, std::string &name) const
{
    if (i >= this->props.size())
        return -1;

    prop_map_t::const_iterator it = this->props.cbegin();
    for (unsigned long q = 0; q < i; ++q)
        ++it;

    name = it->first;

    return 0;
}

// --------------------------------------------------------------------------
int teca_metadata::get_names(std::vector<std::string> &names) const
{
    prop_map_t::const_iterator it = this->props.cbegin();
    prop_map_t::const_iterator end = this->props.cend();
    for (; it != end; ++it)
    {
        names.push_back(it->first);
    }

    return names.size() > 0 ? 0 : -1;
}

// --------------------------------------------------------------------------
int teca_metadata::size(const std::string &name, unsigned int &n) const noexcept
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
        TECA_ERROR("attempt to access a non-existent property ignored!")
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
int teca_metadata::to_stream(teca_binary_stream &s) const
{
    s.pack("teca_metadata", 13);

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

    return 0;
}

// --------------------------------------------------------------------------
int teca_metadata::from_stream(teca_binary_stream &s)
{
    this->clear();

    if (s.expect("teca_metadata"))
    {
        TECA_ERROR("invalid stream")
        return -1;
    }

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

    return 0;
}

// --------------------------------------------------------------------------
int teca_metadata::to_stream(ostream &os) const
{
    prop_map_t::const_iterator it = this->props.cbegin();
    prop_map_t::const_iterator end = this->props.cend();
    for (; it != end; ++it)
    {
        os << it->first << " = " << "{";
        VARIANT_ARRAY_DISPATCH_CASE(std::string, it->second.get(),
            const TT *val = static_cast<const TT*>(it->second.get());
            val->to_stream(os);
            )
        VARIANT_ARRAY_DISPATCH_CASE(teca_metadata, it->second.get(),
            const TT *val = static_cast<const TT*>(it->second.get());
            val->to_stream(os);
            )
        VARIANT_ARRAY_DISPATCH_CASE(p_teca_variant_array, it->second.get(),
            const TT *val = static_cast<const TT*>(it->second.get());
            val->to_stream(os);
            )
        VARIANT_ARRAY_DISPATCH(it->second.get(),
            const TT *val = static_cast<const TT*>(it->second.get());
            val->to_stream(os);
            )
        os << "}" << endl;
    }

    return 0;
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
        if ((lit == lend) || !(lit->second->equal(rit->second)))
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
        if ((lit != lend) && (lit->second->equal(rit->second)))
        {
            isect.set(rit->first, rit->second->new_copy());
        }
    }
    return isect;
}
