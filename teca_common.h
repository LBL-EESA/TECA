#ifndef teca_common_h
#define teca_common_h

#include "teca_parallel_id.h"
#include <iostream>

#define TECA_ERROR(msg)                                          \
    std::cerr << teca_parallel_id()                              \
        << " ERROR " << __FILE__ << ":" << __LINE__ << std::endl \
        << "" msg << std::endl;

#endif
