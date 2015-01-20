#ifndef teca_threaded_algorithm_h
#define teca_threaded_algorithm_h

#include "teca_algorithm.h"
#include "teca_threaded_algorithm_fwd.h"
#include "teca_dataset.h"
class teca_meta_data;
class teca_threaded_algorithm_internals;

#include "teca_algorithm_output_port.h"

// this is the base class defining a threaded algorithm.
// the stratgey employed is to parallelize over upstream
// data requests using a thread pool.
class teca_threaded_algorithm : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_threaded_algorithm)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_threaded_algorithm)
    virtual ~teca_threaded_algorithm();

    // set/get the number of threads in the pool. default
    // is to use 1 - the number of cores.
    void set_thread_pool_size(unsigned int n_threads);
    unsigned int get_thread_pool_size();

protected:
    teca_threaded_algorithm();

    // driver function that manages execution of the given
    // requst on the named port. each upstream request issued
    // will be executed by the thread pool.
    virtual
    p_teca_dataset request_data(
        teca_algorithm_output_port &port,
        const teca_meta_data &request) override;

private:
    teca_threaded_algorithm_internals *internals;
    friend class data_request;
};

#endif
