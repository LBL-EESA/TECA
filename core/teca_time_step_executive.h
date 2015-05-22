#ifndef teca_time_step_executive_h
#define teca_time_step_executive_h

#include "teca_shared_object.h"
#include "teca_algorithm_executive.h"
#include "teca_metadata.h"

#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_time_step_executive)

///
/**
An executive that generates a request for a series of
timesteps. an extent can be optionally set.
*/
class teca_time_step_executive : public teca_algorithm_executive
{
public:
    TECA_ALGORITHM_EXECUTIVE_STATIC_NEW(teca_time_step_executive)

    virtual int initialize(const teca_metadata &md);
    virtual teca_metadata get_next_request();

    // set the time step to process
    void set_step(long s);

    // set the first time step in the series to process.
    // default is 0.
    void set_first_step(long s);

    // set the last time step in the series to process.
    // default is -1. negative number results in the last
    // available time step being used.
    void set_last_step(long s);

    // set the stride to process time steps at. default
    // is 1
    void set_stride(long s);

    // set the extent to process. the default is the
    // whole_extent.
    void set_extent(unsigned long *ext);
    void set_extent(const std::vector<unsigned long> &ext);

    // set the list of arrays to process
    void set_arrays(const std::vector<std::string> &arrays);

protected:
    teca_time_step_executive();

private:
    std::vector<teca_metadata> requests;
    long first_step;
    long last_step;
    long stride;
    std::vector<unsigned long> extent;
    std::vector<std::string> arrays;
};

#endif
