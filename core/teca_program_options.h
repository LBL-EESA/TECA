#ifndef teca_program_options_h
#define teca_program_options_h

#include "teca_config.h"

#if defined(TECA_HAS_BOOST) && !defined(SWIG)
namespace boost
{
    namespace program_options
    {
        class options_description;
        class variables_map;
    }
};

using options_description
    = boost::program_options::options_description;

using variables_map
    = boost::program_options::variables_map;

// initialize the given options description
// with algorithm's properties
#define TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION() \
    void get_properties_description(                \
        const std::string &prefix,                  \
        options_description &opts) override;        \

// initialize the algorithm from the given options
// variable map.
#define TECA_SET_ALGORITHM_PROPERTIES()             \
    void set_properties(                            \
        const std::string &prefix,                  \
        variables_map &opts) override;              \

// helpers for implementation dealing with Boost
// program options. NOTE: because the above declarations
// are intented to be included in class header files
// we are intentionally not including <string> and
// <boost/program_options.hpp>. These need to be
// included in your cxx files.
//
#define TECA_POPTS_GET(_type, _prefix, _name, _desc)        \
     (((_prefix.empty()?"":_prefix+"::") + #_name).c_str(), \
         boost::program_options::value<_type>(), _desc)

#define TECA_POPTS_MULTI_GET(_type, _prefix, _name, _desc)     \
     (((_prefix.empty()?"":_prefix+"::") + #_name).c_str(),    \
         boost::program_options::value<_type>()->multitoken(), \
         _desc)

#define TECA_POPTS_SET(_opts, _type, _prefix, _name)    \
    {std::string opt_name =                             \
        (_prefix.empty()?"":_prefix+"::") + #_name;     \
    if (_opts.count(opt_name))                          \
    {                                                   \
        this->set_##_name(_opts[opt_name].as<_type>()); \
    }}

#else
#define TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
#define TECA_SET_ALGORITHM_PROPERTIES()
#endif
#endif
