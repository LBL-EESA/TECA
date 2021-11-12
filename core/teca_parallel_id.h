#ifndef teca_parallel_id_h
#define teca_parallel_id_h

/// @file

#include "teca_config.h"
#include <iosfwd>

/// A helper class for debug and error messages.
class TECA_EXPORT teca_parallel_id
{};

// Prints the callers rank and thread id to the given stream. This is a
// debug/diagnostic message and hence rank will always be reported relative to
// the WORLD communicator.
TECA_EXPORT
std::ostream &operator<<(std::ostream &os, const teca_parallel_id &id);

#endif
