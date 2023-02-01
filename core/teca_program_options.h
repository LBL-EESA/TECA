#ifndef teca_program_options_h
#define teca_program_options_h

/// @file

#if defined(TECA_HAS_BOOST) && !defined(SWIG)
#include "teca_config.h"
#include "teca_common.h"
#include "teca_mpi_util.h"

#include <iostream>
#include <boost/array.hpp>

namespace boost
{
    namespace program_options
    {
        class options_description;
        class variables_map;
    }
};

using options_description = boost::program_options::options_description;
using variables_map = boost::program_options::variables_map;

// the following allows std::array to work with boost::program_options
namespace std
{
template <typename T, size_t N>
std::istream& operator>>(std::istream& str, std::array<T,N>& arr)
{
    for (size_t i = 0; i < N; ++i)
        str >> std::skipws >> arr[i];
    return str;
}
}
namespace boost
{
template <typename T, size_t N>
std::istream& operator>>(std::istream& str, boost::array<T,N>& arr)
{
    for (size_t i = 0; i < N; ++i)
        str >> std::skipws >> arr[i];
    return str;
}
template <typename T, size_t N>
std::ostream& operator<<(std::ostream& str, const boost::array<T,N>& arr)
{
    str << arr[0];
    for (size_t i = 1; i < N; ++i)
        str << " " << arr[i];
    return str;
}
}

/// initialize the given options description with algorithm's properties
#define TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()                         \
                                                                            \
    /** Adds the class algorithm properties to the description object */    \
    void get_properties_description(const std::string &prefix,              \
        boost::program_options::options_description &opts) override;        \

/// initialize the algorithm from the given options variable map.
#define TECA_SET_ALGORITHM_PROPERTIES()                                     \
    /** Sets the class algorithm properties from the map object */          \
    void set_properties(const std::string &prefix,                          \
        boost::program_options::variables_map &opts) override;              \

// helpers for implementation dealing with Boost
// program options. NOTE: because the above declarations
// are intented to be included in class header files
// we are intentionally not including <string> and
// <boost/program_options.hpp>. These need to be
// included in your cxx files.
//
#define TECA_POPTS_GET(_type, _prefix, _name, _desc)           \
     (((_prefix.empty()?"":_prefix+"::") + #_name).c_str(),    \
         boost::program_options::value<_type>()->default_value \
            (this->get_ ## _name()), "\n" _desc "\n")

#define TECA_POPTS_MULTI_GET(_type, _prefix, _name, _desc)  \
     (((_prefix.empty()?"":_prefix+"::") + #_name).c_str(), \
         boost::program_options::value<_type>()->multitoken \
            ()->default_value(this->get_ ## _name()),       \
         "\n" _desc "\n")

#define TECA_POPTS_SET(_opts, _type, _prefix, _name)             \
    {std::string opt_name =                                      \
        (_prefix.empty()?"":_prefix+"::") + #_name;              \
    bool defd = _opts[opt_name].defaulted();                     \
    if (!defd)                                                   \
    {                                                            \
        _type val = _opts[opt_name].as<_type>();                 \
        if (this->verbose &&                                     \
            teca_mpi_util::mpi_rank_0(this->get_communicator())) \
        {                                                        \
            TECA_STATUS("Setting " << opt_name << " = " << val)  \
        }                                                        \
        this->set_##_name(val);                                  \
    }}

#else
#define TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
#define TECA_SET_ALGORITHM_PROPERTIES()
#endif
#endif
