#ifndef teca_cpu_thread_pool_h
#define teca_cpu_thread_pool_h

#include "teca_config.h"
#include "teca_common.h"
#include "teca_algorithm.h"
#include "teca_thread_util.h"
#include "teca_threadsafe_queue.h"
#include "teca_owned_future.h"
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

template <typename task_t, typename data_t>
class teca_cpu_thread_pool;

template <typename task_t, typename data_t>
using p_teca_cpu_thread_pool = std::shared_ptr<teca_cpu_thread_pool<task_t, data_t>>;

/// A class to manage a fixed size pool of threads that dispatch work.
template <typename task_t, typename data_t>
class TECA_EXPORT teca_cpu_thread_pool
{
public:
    teca_cpu_thread_pool() = delete;

    /** construct/destruct the thread pool.
     * arguments:
     *   @param[in] comm     communicator over which to map threads. Use
     *                       MPI_COMM_SELF for local mapping.
     *
     *   @param[in] n        number of threads to create for the pool. -1 will
     *                       create 1 thread per physical CPU core.  all MPI
     *                       ranks running on the same node are taken into
     *                       account, resulting in 1 thread per core node wide.
     *
     *   @param[in] bind     bind each thread to a specific core.
     *
     *   @param[in] verbose  print a report of the thread to core bindings
     */
    teca_cpu_thread_pool(MPI_Comm comm, int n, bool bind, bool verbose);
    ~teca_cpu_thread_pool() noexcept;

    // get rid of copy and asignment
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_cpu_thread_pool)

    /** add a data request task to the queue, returns a future from which the
     * generated dataset can be accessed.
     */
    void push_task(task_t &task);

    /** wait for all of the requests to execute and transfer datasets in the
     * order that corresponding requests were added to the queue.
     */
    template <template <typename ... > class container_t, typename ... args>
    void wait_all(container_t<data_t, args ...> &data);

    /** wait for some of the requests to execute. datasets will be returned as
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
    void create_threads(MPI_Comm comm, int n_threads, bool bind, bool verbose);

private:
    long m_num_futures;
    std::mutex m_mutex;
    std::atomic<bool> m_live;
    teca_threadsafe_queue<task_t> m_queue;
    std::vector<owned_future<data_t>> m_futures;
    std::vector<std::thread> m_threads;
};

// --------------------------------------------------------------------------
template <typename task_t, typename data_t>
teca_cpu_thread_pool<task_t, data_t>::teca_cpu_thread_pool(MPI_Comm comm, int n,
    bool bind, bool verbose) : m_live(true)
{
    this->create_threads(comm, n, bind, verbose);
}

// --------------------------------------------------------------------------
template <typename task_t, typename data_t>
void teca_cpu_thread_pool<task_t, data_t>::create_threads(MPI_Comm comm,
    int n_requested, bool bind, bool verbose)
{
#if defined(TECA_HAS_MPI)
    // this rank is excluded from computations
    if (comm == MPI_COMM_NULL)
        return;
#endif

    m_num_futures = 0;

    int n_threads = n_requested;

    std::deque<int> core_ids;
    std::vector<int> device_ids;

    if (teca_thread_util::thread_parameters(comm, -1,
        n_requested, -1, -1, bind, verbose, n_threads, core_ids,
        device_ids))
    {
        TECA_WARNING("Failed to detetermine thread parameters."
            " Falling back to 1 thread, affinity disabled.")

        n_threads = 1;
        bind = false;
    }

    // allocate the CPU processing threads
    for (int i = 0; i < n_threads; ++i)
    {
        m_threads.push_back(std::thread([this]()
        {
            // "main" for each thread in the pool
            while (m_live.load())
            {
                task_t task;
                if (m_queue.try_pop(task))
                    task(-1);
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
teca_cpu_thread_pool<task_t, data_t>::~teca_cpu_thread_pool() noexcept
{
    m_live = false;
    std::for_each(m_threads.begin(), m_threads.end(),
        [](std::thread &t) { t.join(); });
}

// --------------------------------------------------------------------------
template <typename task_t, typename data_t>
void teca_cpu_thread_pool<task_t, data_t>::push_task(task_t &task)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    ++m_num_futures;

    m_futures.emplace_back(owned_future(task.get_future()));
    m_queue.push(std::move(task));
}

// --------------------------------------------------------------------------
template <typename task_t, typename data_t>
template <template <typename ... > class container_t, typename ... args>
int teca_cpu_thread_pool<task_t, data_t>::wait_some(long n_to_wait,
    long long poll_interval, container_t<data_t, args ...> &data)
{
    // wait for all
    if (n_to_wait < 1)
    {
        this->wait_all(data);
        return 0;
    }

    // gather the requested number of datasets
    size_t thread_valid = 1;
    while (thread_valid)
    {
        {
        // scan the tasks once. capture any data that is ready
        thread_valid = 0;
        std::lock_guard<std::mutex> lock(m_mutex);
        for (auto it = m_futures.begin(); it != m_futures.end(); ++it)
        {
            if (it->owner() && it->m_future.valid())
            {
                if ((it->m_future.wait_for(std::chrono::seconds::zero())
                    == std::future_status::ready))
                {
                    data.push_back(it->m_future.get());
                    --m_num_futures;
                }
                else
                {
                    ++thread_valid;
                }
            }
        }
        }

        // we have the requested number of datasets
        if (data.size() >= static_cast<unsigned int>(n_to_wait))
            break;

        // wait for the user supplied duration before re-scanning
        if (thread_valid)
            std::this_thread::sleep_for(std::chrono::nanoseconds(poll_interval));
    }

    // last one finished clears the futures
    if (!m_num_futures)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_num_futures)
            m_futures.clear();
    }

    // return the number of tasks remaining
    return thread_valid;
}

// --------------------------------------------------------------------------
template <typename task_t, typename data_t>
template <template <typename ... > class container_t, typename ... args>
void teca_cpu_thread_pool<task_t, data_t>::wait_all(container_t<data_t, args ...> &data)
{
    {
    std::lock_guard<std::mutex> lock(m_mutex);
    // wait on all pending requests and gather the generated datasets
    std::for_each(m_futures.begin(), m_futures.end(),
        [&data,this] (owned_future<data_t> &f)
        {
            if (f.owner())
            {
                data.push_back(f.m_future.get());
                --m_num_futures;
            }
        });
    }

    // last one finished clears the futures
    if (!m_num_futures)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_num_futures)
            m_futures.clear();
    }
}

#endif
