#ifndef teca_threaded_algorithm_h
#define teca_threaded_algorithm_h

#include "teca_algorithm.h"
#include "teca_threaded_algorithm_fwd.h"
#include "teca_dataset.h"
class teca_metadata;
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
    virtual ~teca_threaded_algorithm() noexcept;

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set/get the number of threads in the pool. setting
    // to less than 1 results in 1 - the number of cores.
    // the default is 1.
    void set_thread_pool_size(int n_threads);
    unsigned int get_thread_pool_size() const noexcept;

protected:
    teca_threaded_algorithm();

    // driver function that manages execution of the given
    // requst on the named port. each upstream request issued
    // will be executed by the thread pool.
    virtual
    const_p_teca_dataset request_data(
        teca_algorithm_output_port &port,
        const teca_metadata &request) override;

private:
    teca_threaded_algorithm_internals *internals;
};

#endif
