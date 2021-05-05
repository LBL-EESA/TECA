#ifndef teca_parallel_id_h
#define teca_parallel_id_h

/// @file

#include <iosfwd>

/// A helper class for debug and error messages.
class teca_parallel_id
{};

// Prints the callers rank and thread id to the given stream. This is a
// debug/diagnostic message and hence rank will always be reported relative to
// the WORLD communicator.
std::ostream &operator<<(
    std::ostream &os,
    const teca_parallel_id &id);

#endif
