#ifndef teca_parallel_id_h
#define teca_parallel_id_h

#include <iosfwd>

// a helper class for debug and error messages
class teca_parallel_id
{};

// print the callers rank and thread id to the given stream.  this is a
// debug/diagnostic message and hence rank will always be reported relative to
// the WORLD communicator.
std::ostream &operator<<(
    std::ostream &os,
    const teca_parallel_id &id);

#endif
