#ifndef teca_threaded_algorithm_h
#define teca_threaded_algorithm_h

#include "teca_config.h"
#include "teca_algorithm.h"
#include "teca_dataset.h"
#include "teca_shared_object.h"

#include <thread>
#include <future>

#if defined(TECA_HAS_CUDA)
template <typename task_t, typename data_t>
class teca_cuda_thread_pool;
#else
template <typename task_t, typename data_t>
class teca_cpu_thread_pool;
#endif

class teca_metadata;
class teca_threaded_algorithm_internals;

TECA_SHARED_OBJECT_FORWARD_DECL(teca_threaded_algorithm)

/// Task type for tasks returing a pointer to teca_dataset
using teca_data_request_task = std::packaged_task<const_p_teca_dataset(int)>;

class teca_data_request;

/// A thread pool for processing teca_data_request_task
#if defined(TECA_HAS_CUDA)
using teca_data_request_queue =
    teca_cuda_thread_pool<teca_data_request_task, const_p_teca_dataset>;
#else
using teca_data_request_queue =
    teca_cpu_thread_pool<teca_data_request_task, const_p_teca_dataset>;
#endif

/// A pointer to teca_data_request_queue
using p_teca_data_request_queue = std::shared_ptr<teca_data_request_queue>;

/** Allocate and initialize a new thread pool.
 *
 * @param comm[in]      The communicator to allocate thread across
 * @param n_threads[in] The number of threads to create per MPI rank. Use -1 to
 *                      map one thread per physical core on each node.
 * @param threads_per_device[in] The number of threads to assign to servicing
 *                               each GPU/device.
 * @param ranks_per_device[in] The number of ranks allowed to access each GPU/device.
 * @param bind[in]      If set then thread will be bound to a specific core.
 * @param verbose[in]   If set then the mapping is sent to the stderr
 */
TECA_EXPORT
p_teca_data_request_queue new_teca_data_request_queue(MPI_Comm comm,
    int n_threads, int threads_per_device, int ranks_per_device, bool bind,
    bool verbose);

/// This is the base class defining a threaded algorithm.
/** The strategy employed is to parallelize over upstream data requests using a
 * thread pool. Implementations override teca_algorithm::get_output_metadata,
 * teca_algorithm::get_upstream_request, and teca_algorithm::execute.  Pipeline
 * execution is parallelized over the set of requests returned from the
 * teca_algorithm::get_upstream_request override. The generated data is then
 * fed incrementally to the teca_algorithm::execute override as it arrives in
 * at least stream_size increments.  Alternatively the generated data can be
 * collected and fed to the execute override in one call. However, processing
 * the data in one call is both slower and has a higher memory footprint making
 * it prohibitive in many situations.
 */
class TECA_EXPORT teca_threaded_algorithm : public teca_algorithm
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

    /** @name threads_per_device
     * Set the number of threads to service each GPU/device. Other threads will
     * use the CPU.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, threads_per_device)
    ///@}

    /** @name ranks_per_device
     * Set the number of ranks that have access to each GPU/device. Other ranks
     * will use the CPU.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, ranks_per_device)
    ///@}

    /// explicitly set the thread pool to submit requests to
    void set_data_request_queue(const p_teca_data_request_queue &queue);

    /** @name propagate_device_assignment
     * When set device assignment is taken from down stream request.
     * Otherwise the thread executing the pipeline will provide the assignment.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, propagate_device_assignment)
    ///@}

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
    int threads_per_device;
    int ranks_per_device;
    int propagate_device_assignment;

    teca_threaded_algorithm_internals *internals;
};

#endif
