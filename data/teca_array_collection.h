#ifndef teca_array_collection_h
#define teca_array_collection_h

#include "teca_shared_object.h"
#include "teca_variant_array.h"
#include <map>
#include <vector>
#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_array_collection)

/// A collection of named arrays.
class teca_array_collection
{
public:

    /// construct on heap
    static
    p_teca_array_collection New()
    { return p_teca_array_collection(new teca_array_collection()); }

    /// reset to empty state
    void clear();

    /** declare a set of arrays. requires name,type pairs for ex.
     * define("c1",int(),"c2",float()) creates 2 arrays in the collection the
     * first storing int the second storing float.
     */
    template<typename nT, typename aT, typename... oT>
    void declare_set(nT &&a_name, aT a_type, oT &&...args);

    /// declare a single array
    template<typename nT, typename aT>
    void declare(nT &&a_name, aT a_type);

    /** add, return the index of the new entry,  or -1 if the array name already
     * exists.
     */
    int append(p_teca_variant_array array);

    /** add, return the index of the new entry,  or -1 if the array name already
     * exists.
     */
    int append(const std::string &name, p_teca_variant_array array);

    /** replace the ith array, return 0 on success.  the name of the array is
     * not changed.
     */
    int set(unsigned int i, p_teca_variant_array array);

    /// add or replace the named array, returns 0 on success.
    int set(const std::string &name, p_teca_variant_array array);

    /// remove the ith array
    int remove(unsigned int i);

    /// remove the named array
    int remove(const std::string &name);

    /// Return the number of arrays
    unsigned int size() const noexcept
    { return m_arrays.size(); }

    /// access an array by its by id
    p_teca_variant_array get(unsigned int i)
    { return m_arrays[i]; }

    /// access an array by its by id
    const_p_teca_variant_array get(unsigned int i) const
    { return m_arrays[i]; }

    /// test for array
    bool has(const std::string &name) const;

    /// access an array by name
    p_teca_variant_array get(const std::string &name);

    /// access an array by name
    const_p_teca_variant_array get(const std::string &name) const;

    /// access an array by name
    p_teca_variant_array operator[](const std::string &name)
    { return this->get(name); }

    /// access an array by name
    const_p_teca_variant_array operator[](const std::string &name) const
    { return this->get(name); }

    // Get the name of the ith array
    std::string &get_name(unsigned int i)
    { return m_names[i]; }

    // Get the name of the ith array
    const std::string &get_name(unsigned int i) const
    { return m_names[i]; }

    // Get the list of names
    std::vector<std::string> &get_names()
    { return m_names; }

    // Get the list of names
    const std::vector<std::string> &get_names() const
    { return m_names; }

    /// Return the name of the class
    std::string get_class_name() const
    { return "teca_array_collection"; }

    /// return an integer identifier uniquely naming the dataset type
    int get_type_code() const
    { return -1; }

    /// copy
    void copy(const const_p_teca_array_collection &other);

    /// shallow copy
    void shallow_copy(const p_teca_array_collection &other);

    /// append
    int append(const const_p_teca_array_collection &other);

    /// shallow append
    int shallow_append(const p_teca_array_collection &other);

    /// swap
    void swap(p_teca_array_collection &other);

    /// serialize the data to the given stream for I/O or communication
    int to_stream(teca_binary_stream &s) const;

    /// serialize the data from the given stream for I/O or communication
    int from_stream(teca_binary_stream &s);

    /// stream to a human readable representation
    int to_stream(std::ostream &) const;

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
