#ifndef teca_common_h
#define teca_common_h

#include "teca_config.h"
#include "teca_parallel_id.h"
#include <iostream>
#include <unistd.h>
#include <cstdio>

// detect if we are writing to a tty, if not then
// we should not use ansi color codes
inline int have_tty()
{
    static int have = -1;
    if (have < 0)
        have = isatty(fileno(stderr));
    return have;
}

#define ANSI_RED "\033[1;31m"
#define ANSI_GREEN "\033[1;32m"
#define ANSI_YELLOW "\033[1;33m"
#define ANSI_WHITE "\033[1;37m"
#define ANSI_OFF "\033[0m"

#define BEGIN_HL(_color) (have_tty()?_color:"")
#define END_HL (have_tty()?ANSI_OFF:"")

#define TECA_MESSAGE(_head, _head_color, _msg)                          \
std::cerr                                                               \
    << BEGIN_HL(_head_color) << _head << END_HL                         \
    << " " << teca_parallel_id() << " [" << __FILE__ << ":" << __LINE__ \
    << " " << TECA_VERSION_DESCR << "]" << std::endl                    \
    << BEGIN_HL(_head_color) << _head << END_HL << " "                  \
    << BEGIN_HL(ANSI_WHITE) << "" _msg << END_HL << std::endl;

#define TECA_ERROR(_msg) TECA_MESSAGE("ERROR:", ANSI_RED, _msg)
#define TECA_WARNING(_msg) TECA_MESSAGE("WARNING:", ANSI_YELLOW, _msg)
#define TECA_STATUS(_msg) TECA_MESSAGE("STATUS:", ANSI_GREEN, _msg)

#endif
