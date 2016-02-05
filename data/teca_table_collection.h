#ifndef teca_table_collection_h
#define teca_table_collection_h

#include "teca_table_collection_fwd.h"
#include "teca_table.h"
#include <map>
#include <vector>
#include <string>

// a collection of named tables
/**
A collection of named tables
*/
class teca_table_collection
{
public:
    // construct on heap
    static
    p_teca_table_collection New()
    { return p_teca_table_collection(new teca_table_collection()); }

    // reset to empty state
    void clear();

    // declare a set of tables, from a list of names
    template<typename nT, typename... oT>
    void declare_set(nT &&a_name, oT &&...args);

    // declare a single array
    template<typename nT>
    void declare(nT &&a_name);

    // add, return the index of the new entry.
    // or -1 if the array name already exists.
    int append(p_teca_table array);
    int append(const std::string &name, p_teca_table array);

    // set, return 0 on success.
    int set(unsigned int i, p_teca_table array);
    int set(const std::string &name, p_teca_table array);

    // remove
    int remove(unsigned int i);
    int remove(const std::string &name);

    // number of
    unsigned int size() const noexcept
    { return m_tables.size(); }

    // access by id
    p_teca_table get(unsigned int i)
    { return m_tables[i]; }

    const_p_teca_table get(unsigned int i) const
    { return m_tables[i]; }

    // test for array
    bool has(const std::string &name) const;

    // access by name
    p_teca_table get(const std::string &name);
    const_p_teca_table get(const std::string &name) const;

    p_teca_table operator[](const std::string &name)
    { return this->get(name); }

    const_p_teca_table operator[](const std::string &name) const
    { return this->get(name); }

    // access names
    std::string &get_name(unsigned int i)
    { return m_names[i]; }

    const std::string &get_name(unsigned int i) const
    { return m_names[i]; }

    // copy
    void copy(const const_p_teca_table_collection &other);
    void shallow_copy(const p_teca_table_collection &other);

    // swap
    void swap(p_teca_table_collection &other);

    // serialize the data to/from the given stream
    // for I/O or communication
    void to_stream(teca_binary_stream &s) const;
    void from_stream(teca_binary_stream &s);

    // stream to/from human readable representation
    void to_stream(std::ostream &) const;
    void from_stream(std::istream &) {}

protected:
    teca_table_collection() = default;
    teca_table_collection(const teca_table_collection &) = delete;
    teca_table_collection(const teca_table_collection &&) = delete;
    void operator=(const teca_table_collection &) = delete;
    void operator=(const teca_table_collection &&) = delete;
    void declare_set(){}

private:
    using name_vector_t = std::vector<std::string>;
    using array_vector_t = std::vector<p_teca_table>;
    using name_array_map_t = std::map<std::string,unsigned int>;

    name_vector_t m_names;
    array_vector_t m_tables;
    name_array_map_t m_name_array_map;
};

// --------------------------------------------------------------------------
template<typename nT, typename... oT>
void teca_table_collection::declare_set(nT &&a_name, oT &&... args)
{
    this->declare(std::forward<nT>(a_name));
    this->declare_set(args...);
}

// --------------------------------------------------------------------------
template<typename nT>
void teca_table_collection::declare(nT &&a_name)
{
    unsigned int id = m_tables.size();
    m_names.emplace_back(std::forward<nT>(a_name));
    m_tables.emplace_back(teca_table::New());
    m_name_array_map.emplace(std::forward<nT>(a_name), id);
}

#endif
