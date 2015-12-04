#ifndef teca_program_algorithm_fwd_h
#define teca_program_algorithm_fwd_h

#include "teca_shared_object.h"

TECA_SHARED_OBJECT_FORWARD_DECL(teca_programmable_algorithm)

#ifdef SWIG
typedef void* report_funciton_t;
typedef void* request_function_t;
typedef void* execute_function_t;
#else
using report_function_t
    = std::function<teca_metadata(
        unsigned int, const std::vector<teca_metadata>&)>;

using request_function_t
    = std::function<std::vector<teca_metadata>(
        unsigned int,
        const std::vector<teca_metadata> &,
        const teca_metadata &)>;

using execute_function_t
    = std::function<const_p_teca_dataset(
        unsigned int, const std::vector<const_p_teca_dataset> &,
        const teca_metadata &)>;
#endif

// helper that allows us to use std::function
// as a TECA_ALGORITHM_PROPERTY
template<typename T>
bool operator!=(const std::function<T> &lhs, const std::function<T> &rhs)
{
    return &rhs != &lhs;
}

// TODO -- this is a work around for older versions
// of Apple clang
// Apple LLVM version 4.2 (clang-425.0.28) (based on LLVM 3.2svn)
// Target: x86_64-apple-darwin12.6.0
#define TECA_PROGRAMMABLE_ALGORITHM_PROPERTY(T, NAME) \
                                         \
void set_##NAME(const T &v)              \
{                                        \
    /*if (this->NAME != v)*/                 \
    /*{*/                                    \
        this->NAME = v;                  \
        this->set_modified();            \
    /*}*/                                    \
}                                        \
                                         \
const T &get_##NAME() const              \
{                                        \
    return this->NAME;                   \
}                                        \
                                         \
T &get_##NAME()                          \
{                                        \
    return this->NAME;                   \
}

#endif
