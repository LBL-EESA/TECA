#ifndef teca_algorithm_executive_h
#define teca_algorithm_executive_h

#include "teca_algorithm_executive_fwd.h"
#include "teca_metadata.h"

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
    virtual int initialize(const teca_metadata &)
    { return 0; }

    // get the next request until all requests have been
    // processed. an empty request is returned.
    virtual teca_metadata get_next_request()
    { return teca_metadata(); }

protected:
    teca_algorithm_executive() {}
    teca_algorithm_executive(const teca_algorithm_executive &);
    void operator=(const teca_algorithm_executive &);
};

#endif
