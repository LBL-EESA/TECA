#ifndef teca_common_h
#define teca_common_h

#include "teca_parallel_id.h"
#include <iostream>

#define TECA_ERROR(msg)                                          \
    std::cerr << teca_parallel_id()                              \
        << " ERROR " << __FILE__ << ":" << __LINE__ << std::endl \
        << "" msg << std::endl;

// TODO -- move these so we don't pull in the above headers
// inadvertantly

// t - derived type
// p - base class pointer
// body - code to execute
#define TEMPLATE_DISPATCH_CASE(tt, nt, p, body) \
    if (dynamic_cast<tt<nt>*>(p))               \
    {                                           \
        typedef tt<nt> TT;                      \
        typedef nt NT;                          \
        body                                    \
    }

// macro for helping downcast to POD types
// don't add classes to this.
// t - templated derived type
// p - pointer
// body - code to execute on match
#define TEMPLATE_DISPATCH(t, p, body)                   \
    TEMPLATE_DISPATCH_CASE(t, char, p, body)            \
    TEMPLATE_DISPATCH_CASE(t, unsigned char, p, body)   \
    TEMPLATE_DISPATCH_CASE(t, int, p, body)             \
    TEMPLATE_DISPATCH_CASE(t, unsigned int, p, body)    \
    TEMPLATE_DISPATCH_CASE(t, long, p, body)            \
    TEMPLATE_DISPATCH_CASE(t, unsigned long, p, body)   \
    TEMPLATE_DISPATCH_CASE(t, float, p, body)           \
    TEMPLATE_DISPATCH_CASE(t, double, p, body)

#endif
