#include "teca_table_collection.h"

#include <sstream>
using std::ostringstream;

// ----------------------------------------------------------------------------
void teca_table_collection::clear()
{
    m_tables.clear();
    m_names.clear();
    m_name_array_map.clear();
}

// ----------------------------------------------------------------------------
int teca_table_collection::append(p_teca_table array)
{
    unsigned int id = m_tables.size();

    ostringstream oss;
    oss << "table_" << id;

    return this->append(oss.str(), array);
}

// ----------------------------------------------------------------------------
int teca_table_collection::append(
    const std::string &name,
    p_teca_table array)
{
    name_array_map_t::iterator loc = m_name_array_map.find(name);
    if (loc != m_name_array_map.end())
        return -1;

    unsigned int id = m_tables.size();

    std::pair<name_array_map_t::iterator, bool> ret
        = m_name_array_map.insert(std::make_pair(name, id));

    if (!ret.second)
    {
        TECA_ERROR("Failed to append " << name << " exists")
        return -1;
    }

    m_tables.push_back(array);
    m_names.push_back(name);

    return id;
}

// ----------------------------------------------------------------------------
int teca_table_collection::set(unsigned int i, p_teca_table array)
{
    if (i >= m_names.size())
        return -1;

    m_tables[i] = array;

    return 0;
}

// ----------------------------------------------------------------------------
int teca_table_collection::set(
    const std::string &name,
    p_teca_table array)
{
    std::pair<name_array_map_t::iterator, bool> ret
        = m_name_array_map.insert(std::make_pair(name, m_tables.size()));

    if (ret.second)
    {
        m_names.push_back(name);
        m_tables.push_back(array);
    }
    else
    {
        m_tables[(ret.first)->second] = array;
    }

    return 0;
}

// ----------------------------------------------------------------------------
int teca_table_collection::remove(const std::string &name)
{
    name_array_map_t::iterator loc = m_name_array_map.find(name);
    if (loc != m_name_array_map.end())
    {
        unsigned int id = loc->second;
        m_name_array_map.erase(loc);
        m_names.erase(m_names.begin()+id);
        m_tables.erase(m_tables.begin()+id);
        return 0;
    }
    return -1;
}

// ----------------------------------------------------------------------------
int teca_table_collection::remove(unsigned int id)
{
    m_tables.erase(m_tables.begin()+id);
    return 0;
}

// ----------------------------------------------------------------------------
bool teca_table_collection::has(const std::string &name) const
{
    name_array_map_t::const_iterator loc = m_name_array_map.find(name);
    if (loc == m_name_array_map.cend())
        return false;
    return true;
}

// ----------------------------------------------------------------------------
p_teca_table teca_table_collection::get(const std::string &name)
{
    name_array_map_t::const_iterator loc = m_name_array_map.find(name);
    if (loc == m_name_array_map.cend())
        return nullptr;

    unsigned int id = loc->second;
    return m_tables[id];
}

// ----------------------------------------------------------------------------
const_p_teca_table teca_table_collection::get(const std::string &name) const
{
    name_array_map_t::const_iterator loc = m_name_array_map.find(name);
    if (loc == m_name_array_map.cend())
        return nullptr;

    unsigned int id = loc->second;
    return m_tables[id];
}

// ----------------------------------------------------------------------------
void teca_table_collection::copy(const const_p_teca_table_collection &other)
{
    m_names = other->m_names;
    m_name_array_map = other->m_name_array_map;

    unsigned int n = other->size();
    for (unsigned int i = 0; i < n; ++i)
        m_tables.push_back(
            std::dynamic_pointer_cast<teca_table>(
                other->get(i)->new_copy()));
}

// ----------------------------------------------------------------------------
void teca_table_collection::shallow_copy(const p_teca_table_collection &other)
{
    m_names = other->m_names;
    m_name_array_map = other->m_name_array_map;

    unsigned int n = other->size();
    for (unsigned int i = 0; i < n; ++i)
        m_tables.push_back(other->get(i));
}

// --------------------------------------------------------------------------
void teca_table_collection::swap(p_teca_table_collection &other)
{
    std::swap(m_names, other->m_names);
    std::swap(m_name_array_map, other->m_name_array_map);
    std::swap(m_tables, other->m_tables);
}

// --------------------------------------------------------------------------
void teca_table_collection::to_stream(teca_binary_stream &s) const
{
    unsigned int na = m_tables.size();
    s.pack(na);
    s.pack(m_names);
    for (unsigned int i = 0; i < na; ++i)
        m_tables[i]->to_stream(s);
}

// --------------------------------------------------------------------------
void teca_table_collection::from_stream(teca_binary_stream &s)
{
    unsigned int na;
    s.unpack(na);
    s.unpack(m_names);

    m_tables.resize(na);

    for (unsigned int i = 0; i < na; ++i)
    {
        m_tables[i] = teca_table::New();
        m_tables[i]->from_stream(s);
    }
}

// --------------------------------------------------------------------------
void teca_table_collection::to_stream(std::ostream &s) const
{
    size_t n_tables = this->size();
    if (n_tables)
    {
        s << "table 0: " << this->get_name(0) << std::endl;
        this->get(0)->to_stream(s);
        s << std::endl;
        for (size_t i = 1; i < n_tables; ++i)
        {
            s << "table " << i << ": " <<  this->get_name(i) << std::endl;
            this->get(i)->to_stream(s);
            s << std::endl;
        }
    }
}
