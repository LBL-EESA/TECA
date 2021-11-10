#ifndef teca_cuda_thread_pool_h
#define teca_cuda_thread_pool_h

#include "teca_common.h"
#include "teca_algorithm.h"
#include "teca_thread_util.h"
#include "teca_threadsafe_queue.h"
#include "teca_mpi.h"

#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <future>
#include <chrono>
#include <algorithm>
#if defined(_GNU_SOURCE)
#include <pthread.h>
#include <sched.h>
#include <deque>
#endif

namespace teca_cuda_util
{
int get_local_cuda_devices(MPI_Comm comm, std::deque<int> &local_dev);
}

template <typename task_t, typename data_t>
class teca_cuda_thread_pool;

template <typename task_t, typename data_t>
using p_teca_cuda_thread_pool = std::shared_ptr<teca_cuda_thread_pool<task_t, data_t>>;

/// A class to manage a fixed size pool of threads that dispatch work.
template <typename task_t, typename data_t>
class teca_cuda_thread_pool
{
public:
    teca_cuda_thread_pool() = delete;
    ~teca_cuda_thread_pool() noexcept;

    /** construct/destruct the thread pool.
     *
     *   @param[in] comm      communicator over which to map threads. Use
     *                        MPI_COMM_SELF for local mapping.
     *
     *   @param[in] n_threads number of threads to create for the pool. -1 will
     *                        create 1 thread per physical CPU core.  all MPI
     *                        ranks running on the same node are taken into
     *                        account, resulting in 1 thread per core node wide.
     *
     *   @param[in[ n_threads_per_device number of threads to assign to each CUDA
     *                                   device. -1 for all threads assigned.
     *
     *   @param[in] bind      bind each thread to a specific core.
     *
     *   @param[in] verbose   print a report of the thread to core bindings
     */
    teca_cuda_thread_pool(MPI_Comm comm, int n_threads,
        int n_threads_per_device, bool bind, bool verbose);

    // get rid of copy and asignment
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_cuda_thread_pool)

    /** add a data request task to the queue, returns a future from which the
     * generated dataset can be accessed.
     */
    void push_task(task_t &task);

    /** wait for all of the requests to execute and transfer datasets in the
     * order that corresponding requests were added to the queue.
     */
    template <template <typename ... > class container_t, typename ... args>
    void wait_all(container_t<data_t, args ...> &data);

    /** wait for some of the requests to execute. datasets will be retruned as
     * they become ready. n_to_wait specifies how many datasets to gather but
     * there are three cases when the number of datasets returned differs from
     * n_to_wait.  when n_to_wait is larger than the number of tasks remaining,
     * datasets from all of the remaining tasks is returned. when n_to_wait is
     * smaller than the number of datasets ready, all of the currenttly ready
     * data are returned. finally, when n_to_wait is < 1 the call blocks until
     * all of the tasks complete and all of the data is returned.
     */
    template <template <typename ... > class container_t, typename ... args>
    int wait_some(long n_to_wait, long long poll_interval,
        container_t<data_t, args ...> &data);

    /// get the number of threads
    unsigned int size() const noexcept
    { return m_threads.size(); }

private:
    /// create n threads for the pool
    void create_threads(MPI_Comm comm, int n_threads,
        int n_threads_per_device, bool bind, bool verbose);

private:
    std::atomic<bool> m_live;
    teca_threadsafe_queue<task_t> m_queue;
    std::vector<std::future<data_t>> m_futures;
    std::vector<std::thread> m_threads;
};

// --------------------------------------------------------------------------
template <typename task_t, typename data_t>
teca_cuda_thread_pool<task_t, data_t>::teca_cuda_thread_pool(MPI_Comm comm,
    int n_threads, int n_threads_per_device, bool bind, bool verbose) : m_live(true)
{
    this->create_threads(comm, n_threads, n_threads_per_device, bind, verbose);
}

// --------------------------------------------------------------------------
template <typename task_t, typename data_t>
void teca_cuda_thread_pool<task_t, data_t>::create_threads(MPI_Comm comm,
    int n_requested, int n_threads_per_device, bool bind, bool verbose)
{
#if defined(TECA_HAS_MPI)
    // this rank is excluded from computations
    if (comm == MPI_COMM_NULL)
        return;
#endif

    int n_threads = n_requested;

    // determine available CPU cores
    std::deque<int> core_ids;

    if (teca_thread_util::thread_parameters(comm, -1,
        n_requested, bind, verbose, n_threads, core_ids))
    {
        TECA_WARNING("Failed to detetermine thread parameters."
            " Falling back to 1 thread, affinity disabled.")

        n_threads = 1;
        bind = false;
    }

    // determine the available CUDA GPUs
#if defined(TECA_HAS_CUDA)
    std::deque<int> cuda_devices;
    if (teca_cuda_util::get_local_cuda_devices(comm, cuda_devices))
    {
        TECA_WARNING("Failed to determine the local CUDA devices."
            " Falling back to the default device.")
        cuda_devices.resize(1, 0);
    }
    int n_cuda_devices = cuda_devices.size();

    if (n_threads < n_cuda_devices)
    {
        TECA_WARNING(<< n_threads
            << " threads is insufficient to service " << n_cuda_devices
            << " CUDA devices. " << n_cuda_devices - n_threads
            << " CUDA devices will not be utilized.")
    }
#endif

    int n_device_threads = n_threads_per_device < 1 ?
        n_threads : n_cuda_devices * n_threads_per_device;

    for (int i = 0; i < n_threads; ++i)
    {
        // select the CUDA device [0, n_cuda_devices) that this thread will
        // utilize. Once all devices are assigned a thread the remaining
        // threads will make use of CPU cores, specfied by setting cuda_device
        // to -1
        int device_id = -1;
        if (i < n_device_threads)
            device_id = cuda_devices[i % n_cuda_devices];

         m_threads.push_back(std::thread([this, device_id]()
         {
             // "main" for each thread in the pool
             while (m_live.load())
             {
                 task_t task;
                 if (m_queue.try_pop(task))
                     task(device_id);
                 else
                     std::this_thread::yield();
             }
         }));
#if defined(_GNU_SOURCE)
        // bind each to a hyperthread
        if (bind)
        {
            int core_id = core_ids.front();
            core_ids.pop_front();

            cpu_set_t core_mask;
            CPU_ZERO(&core_mask);
            CPU_SET(core_id, &core_mask);

            if (pthread_setaffinity_np(m_threads[i].native_handle(),
                sizeof(cpu_set_t), &core_mask))
            {
                TECA_WARNING("Failed to set thread affinity.")
            }
        }
#endif
    }
}

// --------------------------------------------------------------------------
template <typename task_t, typename data_t>
teca_cuda_thread_pool<task_t, data_t>::~teca_cuda_thread_pool() noexcept
{
    m_live = false;
    std::for_each(m_threads.begin(), m_threads.end(),
        [](std::thread &t) { t.join(); });
}

// --------------------------------------------------------------------------
template <typename task_t, typename data_t>
void teca_cuda_thread_pool<task_t, data_t>::push_task(task_t &task)
{
    m_futures.push_back(task.get_future());
    m_queue.push(std::move(task));
}

// --------------------------------------------------------------------------
template <typename task_t, typename data_t>
template <template <typename ... > class container_t, typename ... args>
int teca_cuda_thread_pool<task_t, data_t>::wait_some(long n_to_wait,
    long long poll_interval, container_t<data_t, args ...> &data)
{
    long n_tasks = m_futures.size();

    // wait for all
    if (n_to_wait < 1)
    {
        this->wait_all(data);
        return 0;
    }
    // wait for at most the number of queued tasks
    else if (n_to_wait > n_tasks)
        n_to_wait = n_tasks;


    // gather the requested number of datasets
    while (1)
    {
        // scan the tasks once. capture any data that is ready
        auto it = m_futures.begin();
        while (it != m_futures.end())
        {
            std::future_status stat = it->wait_for(std::chrono::seconds::zero());
            if (stat == std::future_status::ready)
            {
                data.push_back(it->get());
                it = m_futures.erase(it);
            }
            else
            {
                ++it;
            }
        }

        // if we have not accumulated the requested number of datasets
        // wait for the user supplied duration before re-scanning
        if (data.size() < static_cast<unsigned int>(n_to_wait))
            std::this_thread::sleep_for(std::chrono::nanoseconds(poll_interval));
        else
            break;
    }

    // return the number of tasks remaining
    return m_futures.size();
}

// --------------------------------------------------------------------------
template <typename task_t, typename data_t>
template <template <typename ... > class container_t, typename ... args>
void teca_cuda_thread_pool<task_t, data_t>::wait_all(container_t<data_t, args ...> &data)
{
    // wait on all pending requests and gather the generated
    // datasets
    std::for_each(m_futures.begin(), m_futures.end(),
        [&data] (std::future<data_t> &f)
        {
            data.push_back(f.get());
        });
    m_futures.clear();
}

#endif
