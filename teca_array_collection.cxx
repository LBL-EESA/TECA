#include "teca_array_collection.h"


// ----------------------------------------------------------------------------
void teca_array_collection::clear()
{
    m_arrays.clear();
    m_names.clear();
    m_name_array_map.clear();
}

// ----------------------------------------------------------------------------
unsigned int teca_array_collection::add(p_teca_variant_array array)
{
    return this->add("", array);
}

// ----------------------------------------------------------------------------
unsigned int teca_array_collection::add(
    const std::string &name,
    p_teca_variant_array array)
{
    unsigned int id = m_arrays.size();
    m_arrays.push_back(array);
    m_names.push_back(name);
    m_name_array_map[name] = id;
    return id;
}

// ----------------------------------------------------------------------------
int teca_array_collection::remove(const std::string &name)
{
    name_array_map_t::iterator loc = m_name_array_map.find(name);
    if (loc != m_name_array_map.end())
    {
        unsigned int id = loc->second;
        m_name_array_map.erase(loc);
        m_names.erase(m_names.begin()+id);
        m_arrays.erase(m_arrays.begin()+id);
        return 0;
    }
    return -1;
}

// ----------------------------------------------------------------------------
int teca_array_collection::remove(unsigned int id)
{
    m_arrays.erase(m_arrays.begin()+id);
    return 0;
}

// ----------------------------------------------------------------------------
const_p_teca_variant_array teca_array_collection::get(const std::string name) const
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
void teca_array_collection::to_stream(teca_binary_stream &s) const
{
    unsigned int na = m_arrays.size();
    s.pack(na);
    s.pack(m_names);
    for (unsigned int i = 0; i < na; ++i)
        m_arrays[i]->to_stream(s);
}

// --------------------------------------------------------------------------
void teca_array_collection::from_stream(teca_binary_stream &s)
{
    unsigned int na;
    s.unpack(na);
    s.unpack(m_names);
    for (unsigned int i = 0; i < na; ++i)
        m_arrays[i]->from_stream(s);
}
