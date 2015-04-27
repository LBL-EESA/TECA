#ifndef teca_algorithm_executive_h
#define teca_algorithm_executive_h

#include "teca_algorithm_executive_fwd.h"
#include "teca_metadata.h"

// base class and default implementation for executives. algorithm
// executives can control pipeline execution by providing a series
// of requests. this allows for the executive to act as a load
// balancer. the executive can for example partition requests across
// spatial data, time steps, or file names. in an MPI parallel
// setting the executive could coordinate this partitioning amongst
// the ranks. However, the only requirement of an algorithm executive
// is that it provide at least one non-empty request.
//
// the default implementation creates a single trivially non-empty
// request containing the key "__request_empty = 0". This will cause
// the pipeline to be executed once but will result in no data being
// requested. Therefore when the default implementation is used
// upstream algorithms must fill in the requests further to pull
// data as needed.
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
    virtual int initialize(const teca_metadata &md);

    // get the next request until all requests have been
    // processed. an empty request is returned.
    virtual teca_metadata get_next_request();

protected:
    teca_algorithm_executive() = default;
    teca_algorithm_executive(const teca_algorithm_executive &) = default;
    teca_algorithm_executive(teca_algorithm_executive &&) = default;
    teca_algorithm_executive &operator=(const teca_algorithm_executive &) = default;
    teca_algorithm_executive &operator=(teca_algorithm_executive &&) = default;

private:
    teca_metadata m_md;
};

#endif
