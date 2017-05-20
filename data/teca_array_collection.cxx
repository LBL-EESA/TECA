#include "teca_array_collection.h"

#include <sstream>
using std::ostringstream;

// ----------------------------------------------------------------------------
void teca_array_collection::clear()
{
    m_arrays.clear();
    m_names.clear();
    m_name_array_map.clear();
}

// ----------------------------------------------------------------------------
int teca_array_collection::append(p_teca_variant_array array)
{
    unsigned int id = m_arrays.size();

    ostringstream oss;
    oss << "array_" << id;

    return this->append(oss.str(), array);
}

// ----------------------------------------------------------------------------
int teca_array_collection::append(const std::string &name,
    p_teca_variant_array array)
{
    name_array_map_t::iterator loc = m_name_array_map.find(name);
    if (loc != m_name_array_map.end())
        return -1;

    unsigned int id = m_arrays.size();

    std::pair<name_array_map_t::iterator, bool> ret
        = m_name_array_map.insert(std::make_pair(name, id));

    if (!ret.second)
    {
        TECA_ERROR("Failed to append " << name << " exists")
        return -1;
    }

    m_arrays.push_back(array);
    m_names.push_back(name);

    return id;
}

// ----------------------------------------------------------------------------
int teca_array_collection::set(unsigned int i, p_teca_variant_array array)
{
    if (i >= m_names.size())
        return -1;

    m_arrays[i] = array;

    return 0;
}

// ----------------------------------------------------------------------------
int teca_array_collection::set(const std::string &name,
    p_teca_variant_array array)
{
    std::pair<name_array_map_t::iterator, bool> ret
        = m_name_array_map.insert(std::make_pair(name, m_arrays.size()));

    if (ret.second)
    {
        m_names.push_back(name);
        m_arrays.push_back(array);
    }
    else
    {
        m_arrays[(ret.first)->second] = array;
    }

    return 0;
}

// ----------------------------------------------------------------------------
int teca_array_collection::remove(const std::string &name)
{
    name_array_map_t::iterator loc = m_name_array_map.find(name);
    if (loc != m_name_array_map.end())
    {
        unsigned int id = loc->second;
        m_name_array_map.erase(loc);
        // update the map
        name_array_map_t::iterator it = m_name_array_map.begin();
        name_array_map_t::iterator end = m_name_array_map.end();
        for (; it != end; ++it)
        {
            if (it->second > id)
                it->second -= 1;
        }
        // remove name and array
        m_names.erase(m_names.begin()+id);
        m_arrays.erase(m_arrays.begin()+id);
        return 0;
    }
    return -1;
}

// ----------------------------------------------------------------------------
int teca_array_collection::remove(unsigned int id)
{
    if (id < m_names.size())
        return this->remove(m_names[id]);
    return -1;
}

// ----------------------------------------------------------------------------
bool teca_array_collection::has(const std::string &name) const
{
    return m_name_array_map.count(name) > 0;
}

// ----------------------------------------------------------------------------
p_teca_variant_array teca_array_collection::get(const std::string &name)
{
    name_array_map_t::const_iterator loc = m_name_array_map.find(name);
    if (loc == m_name_array_map.cend())
        return nullptr;

    unsigned int id = loc->second;
    return m_arrays[id];
}

// ----------------------------------------------------------------------------
const_p_teca_variant_array teca_array_collection::get(const std::string &name) const
{
    name_array_map_t::const_iterator loc = m_name_array_map.find(name);
    if (loc == m_name_array_map.cend())
        return nullptr;

    unsigned int id = loc->second;
    return m_arrays[id];
}

// ----------------------------------------------------------------------------
void teca_array_collection::copy(const const_p_teca_array_collection &other)
{
    m_names = other->m_names;
    m_name_array_map = other->m_name_array_map;

    unsigned int n = other->size();
    for (unsigned int i = 0; i < n; ++i)
        m_arrays.push_back(other->get(i)->new_copy());
}

// ----------------------------------------------------------------------------
void teca_array_collection::shallow_copy(const p_teca_array_collection &other)
{
    m_names = other->m_names;
    m_name_array_map = other->m_name_array_map;

    unsigned int n = other->size();
    for (unsigned int i = 0; i < n; ++i)
        m_arrays.push_back(other->get(i));
}

// --------------------------------------------------------------------------
void teca_array_collection::swap(p_teca_array_collection &other)
{
    std::swap(m_names, other->m_names);
    std::swap(m_name_array_map, other->m_name_array_map);
    std::swap(m_arrays, other->m_arrays);
}

// --------------------------------------------------------------------------
void teca_array_collection::to_stream(teca_binary_stream &s) const
{
    unsigned int na = m_arrays.size();
    s.pack(na);
    s.pack(m_names);
    for (unsigned int i = 0; i < na; ++i)
    {
        s.pack(m_arrays[i]->type_code());
        m_arrays[i]->to_stream(s);
    }
}

// --------------------------------------------------------------------------
void teca_array_collection::from_stream(teca_binary_stream &s)
{
    unsigned int na;
    s.unpack(na);
    s.unpack(m_names);

    m_arrays.resize(na);

    for (unsigned int i = 0; i < na; ++i)
    {
        unsigned int type_code;
        s.unpack(type_code);

        m_arrays[i] = teca_variant_array_factory::New(type_code);
        m_arrays[i]->from_stream(s);
        m_name_array_map.emplace(m_names[i], i);
    }
}

// --------------------------------------------------------------------------
void teca_array_collection::to_stream(std::ostream &s) const
{
    s << "{" << std::endl;
    size_t n_arrays = this->size();
    if (n_arrays)
    {
        s << this->get_name(0) << " = {";
        this->get(0)->to_stream(s);
        s << "}" << std::endl;

        for (size_t i = 1; i < n_arrays; ++i)
        {
            s << ", " <<  this->get_name(i) << " = {";
            this->get(i)->to_stream(s);
            s << "}" << std::endl;
        }
    }
    s << "}" << std::endl;
}
