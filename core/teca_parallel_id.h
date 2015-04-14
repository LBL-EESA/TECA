#ifndef teca_parallel_id_h
#define teca_parallel_id_h

#include <iosfwd>

// a helper class for debug and error
// messages
class teca_parallel_id
{};

// print the callers rank and thread id
// to the given stream
std::ostream &operator<<(
    std::ostream &os,
    const teca_parallel_id &id);

#endif
