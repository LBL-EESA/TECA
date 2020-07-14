#ifndef teca_array_collection_h
#define teca_array_collection_h

#include "teca_array_collection_fwd.h"
#include "teca_variant_array.h"
#include <map>
#include <vector>
#include <string>

// a collection of named arrays
/**
A collection of named arrays
*/
class teca_array_collection
{
public:
    // construct on heap
    static
    p_teca_array_collection New()
    { return p_teca_array_collection(new teca_array_collection()); }

    // reset to empty state
    void clear();

    // declare a set of arrays. requires name,type pairs
    // for ex. define("c1",int(),"c2",float()) creates
    // 2 arrays in the collection the first storing int's
    // the second storing float's.
    template<typename nT, typename aT, typename... oT>
    void declare_set(nT &&a_name, aT a_type, oT &&...args);

    // declare a single array
    template<typename nT, typename aT>
    void declare(nT &&a_name, aT a_type);

    // add, return the index of the new entry.
    // or -1 if the array name already exists.
    int append(p_teca_variant_array array);
    int append(const std::string &name, p_teca_variant_array array);

    // set, return 0 on success.
    int set(unsigned int i, p_teca_variant_array array);
    int set(const std::string &name, p_teca_variant_array array);

    // remove
    int remove(unsigned int i);
    int remove(const std::string &name);

    // number of
    unsigned int size() const noexcept
    { return m_arrays.size(); }

    // access by id
    p_teca_variant_array get(unsigned int i)
    { return m_arrays[i]; }

    const_p_teca_variant_array get(unsigned int i) const
    { return m_arrays[i]; }

    // test for array
    bool has(const std::string &name) const;

    // access by name
    p_teca_variant_array get(const std::string &name);
    const_p_teca_variant_array get(const std::string &name) const;

    p_teca_variant_array operator[](const std::string &name)
    { return this->get(name); }

    const_p_teca_variant_array operator[](const std::string &name) const
    { return this->get(name); }

    // access names
    std::string &get_name(unsigned int i)
    { return m_names[i]; }

    const std::string &get_name(unsigned int i) const
    { return m_names[i]; }

    // copy
    void copy(const const_p_teca_array_collection &other);
    void shallow_copy(const p_teca_array_collection &other);

    // append
    int append(const const_p_teca_array_collection &other);
    int shallow_append(const p_teca_array_collection &other);

    // swap
    void swap(p_teca_array_collection &other);

    // serialize the data to/from the given stream
    // for I/O or communication
    void to_stream(teca_binary_stream &s) const;
    void from_stream(teca_binary_stream &s);

    // stream to/from human readable representation
    void to_stream(std::ostream &) const;

protected:
    teca_array_collection() = default;
    teca_array_collection(const teca_array_collection &) = delete;
    teca_array_collection(const teca_array_collection &&) = delete;
    void operator=(const teca_array_collection &) = delete;
    void operator=(const teca_array_collection &&) = delete;
    void declare_set(){}

private:
    using name_vector_t = std::vector<std::string>;
    using array_vector_t = std::vector<p_teca_variant_array>;
    using name_array_map_t = std::map<std::string,unsigned int>;

    name_vector_t m_names;
    array_vector_t m_arrays;
    name_array_map_t m_name_array_map;
};

// --------------------------------------------------------------------------
template<typename nT, typename aT, typename... oT>
void teca_array_collection::declare_set(nT &&a_name, aT a_type, oT &&... args)
{
    this->declare(std::forward<nT>(a_name), a_type);
    this->declare_set(args...);
}

// --------------------------------------------------------------------------
template<typename nT, typename aT>
void teca_array_collection::declare(nT &&a_name, aT)
{
    unsigned int id = m_arrays.size();
    m_names.emplace_back(std::forward<nT>(a_name));
    m_arrays.emplace_back(teca_variant_array_impl<aT>::New());
    m_name_array_map.emplace(std::forward<nT>(a_name), id);
}

#endif
