#ifndef teca_common_h
#define teca_common_h

#include "teca_config.h"
#include "teca_parallel_id.h"
#include <iostream>

#define TECA_ERROR(msg)                                 \
    std::cerr                                           \
        << "\033[1;31mERROR:\033[0m "                   \
        << teca_parallel_id()                           \
        << " [" << __FILE__ << ":" << __LINE__          \
        << " " << TECA_VERSION_DESCR                    \
        << "]" << std::endl                             \
        << "\033[1;31mERROR:\033[0m "                   \
        << "\033[1;37m" msg << "\033[0m" << std::endl;

#endif
