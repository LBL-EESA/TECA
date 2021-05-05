#ifndef teca_threaded_algorithm_h
#define teca_threaded_algorithm_h

#include "teca_algorithm.h"
#include "teca_dataset.h"
#include "teca_shared_object.h"

#include <thread>
#include <future>

template <typename task_t, typename data_t>
class teca_thread_pool;

class teca_metadata;
class teca_threaded_algorithm_internals;

TECA_SHARED_OBJECT_FORWARD_DECL(teca_threaded_algorithm)

/// Task type for tasks returing a pointer to teca_dataset
using teca_data_request_task = std::packaged_task<const_p_teca_dataset()>;

class teca_data_request;

/// A thread pool for processing teca_data_request_task
using teca_data_request_queue =
    teca_thread_pool<teca_data_request_task, const_p_teca_dataset>;

/// A pointer to teca_data_request_queue
using p_teca_data_request_queue = std::shared_ptr<teca_data_request_queue>;

/** Allocate and initialize a new thread pool.
 * comm [in] The communicator to allocate thread across
 * n [in] The number of threads to create per MPI rank. Use -1 to
 *        map one thread per physical core on each node.
 * bind [in] If set then thread will be bound to a specific core.
 * verbose [in] If set then the mapping is sent to the stderr
 */
p_teca_data_request_queue new_teca_data_request_queue(MPI_Comm comm,
    int n, bool bind, bool verbose);

/// This is the base class defining a threaded algorithm.
/**
 * The strategy employed is to parallelize over upstream
 * data requests using a thread pool.
 */
class teca_threaded_algorithm : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_threaded_algorithm)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_threaded_algorithm)
    TECA_ALGORITHM_CLASS_NAME(teca_threaded_algorithm)
    virtual ~teca_threaded_algorithm() noexcept;

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** Set the number of threads in the pool. setting to -1 results in a
     * thread per core factoring in all MPI ranks running on the node.
     */
    void set_thread_pool_size(int n_threads);

    /// Get the number of threads in the pool.
    unsigned int get_thread_pool_size() const noexcept;

    /** @name verbose
     * set/get the verbosity level.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, verbose)
    ///@}

    /** @name bind_threads
     * set/get thread affinity mode. When 0 threads are not bound CPU cores,
     * allowing for migration among all cores. This will likely degrade
     * performance. Default is 1.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, bind_threads)
    ///@}

    /** @name stream_size
     * set the smallest number of datasets to gather per call to execute. the
     * default (-1) results in all datasets being gathered. In practice more
     * datasets will be returned if ready
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, stream_size)
    ///@}

    /** @name poll_interval
     * set the duration in nanoseconds to wait between checking for completed
     * tasks
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(long long, poll_interval)
    ///@}

    // explicitly set the thread pool to submit requests to
    void set_data_request_queue(const p_teca_data_request_queue &queue);

protected:
    teca_threaded_algorithm();

    // streaming execute. streaming flag will be set when there is more
    // data to process. it is not safe to use MPI when the streaming flag
    // is set. on the last call streaming flag will not be set, at that
    // point MPI may be used.
    virtual
    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request, int streaming);

    // forward to streaming execute
    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    // driver function that manages execution of the given
    // request on the named port. each upstream request issued
    // will be executed by the thread pool.
    const_p_teca_dataset request_data(teca_algorithm_output_port &port,
        const teca_metadata &request) override;

private:
    int bind_threads;
    int stream_size;
    long long poll_interval;

    teca_threaded_algorithm_internals *internals;
};

#endif
