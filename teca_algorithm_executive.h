#ifndef teca_algorithm_executive_h
#define teca_algorithm_executive_h

#include "teca_meta_data.h"

#include <memory>

class teca_algorithm_executive;
typedef std::shared_ptr<teca_algorithm_executive> p_teca_algorithm_executive;

// base class for executives. algorithm executives control pipeline
// execution by providing a series of requests. this allows for
// the executive to act as a load balancer. the executive can for
// example partition requests across spatial data, time steps,
// or file names. in an MPI parallel setting the executive should
// coordinate this partitioning amongst the ranks.
class teca_algorithm_executive
    : public std::enable_shared_from_this<teca_algorithm_executive>
{
public:
    static p_teca_algorithm_executive New()
    { return p_teca_algorithm_executive(new teca_algorithm_executive); }

    virtual ~teca_algorithm_executive() {}

    // initialize requests from the given metadata object.
    // this is a place where work partitioning across MPI
    // ranks can occur
    virtual int initialize(const teca_meta_data &md)
    { return 0; }

    // get the next request until all requests have been
    // processed. an empty request is returned.
    virtual teca_meta_data get_next_request()
    { return teca_meta_data(); }

protected:
    teca_algorithm_executive() {}
    teca_algorithm_executive(const teca_algorithm_executive &);
    void operator=(const teca_algorithm_executive &);
};

// this is a convenience macro to be used to declare a static
// New method that will be used to construct new objects in
// shared_ptr's. This manages the details of interoperability
// with std C++11 shared pointer
#define TECA_ALGORITHM_EXECUTIVE_STATIC_NEW(T)                              \
                                                                            \
static p_##T New()                                                          \
{                                                                           \
    return p_##T(new T);                                                    \
}                                                                           \
                                                                            \
using enable_shared_from_this<teca_algorithm_executive>::shared_from_this;  \
                                                                            \
std::shared_ptr<T> shared_from_this()                                       \
{                                                                           \
    return std::static_pointer_cast<T>(shared_from_this());                 \
}                                                                           \
                                                                            \
std::shared_ptr<T const> shared_from_this() const                           \
{                                                                           \
    return std::static_pointer_cast<T const>(shared_from_this());           \
}

#endif
