#include "teca_threaded_algorithm.h"
#include "teca_meta_data.h"
#include "teca_threadsafe_queue.h"

#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <future>

using std::vector;
using std::thread;
using std::lock_guard;
using std::atomic;
using std::mutex;
using std::future;
using std::packaged_task;
using std::ref;
using std::cref;
using std::dynamic_pointer_cast;


// function that executes the data request and returns the
// requested dataset
class teca_data_request
{
public:
    teca_data_request(
        const p_teca_algorithm &alg,
        const teca_algorithm_output_port up_port,
        const teca_meta_data &up_req)
        : m_alg(alg), m_up_port(up_port), m_up_req(up_req)
    {}

    p_teca_dataset operator()()
    { return m_alg->request_data(m_up_port, m_up_req); }

public:
    p_teca_algorithm m_alg;
    teca_algorithm_output_port m_up_port;
    teca_meta_data m_up_req;
};

// task
typedef
packaged_task<p_teca_dataset()> teca_data_request_task;

class teca_thread_pool;
typedef
std::shared_ptr<teca_thread_pool> p_teca_thread_pool;

// a class to manage a fixed size pool of threads that dispatch
// data requests to teca_algorithm
class teca_thread_pool
{
public:
    // construct/destruct the thread pool. If the size of the
    // pool is unspecified the number of cores - 1 threads
    // will be used.
    teca_thread_pool();
    teca_thread_pool(unsigned int n);
    ~teca_thread_pool() TECA_NOEXCEPT;
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_thread_pool)

    // add a data request task to the queue, returns a future
    // from which the generated dataset can be accessed.
    void push_data_request(
        const p_teca_algorithm &alg,
        const teca_algorithm_output_port &up_port,
        const teca_meta_data &up_req);

    // wait for all of the requests to execute and transfer
    // datasets in the order that corresponding requests
    // were added to the queue.
    void pop_datasets(vector<p_teca_dataset> &data);

    // get the number of threads
    unsigned int size() const TECA_NOEXCEPT
    { return m_threads.size(); }

private:
    // create n threads for the pool
    void create_threads(unsigned int n_threads);

private:
    atomic<bool> m_live;
    teca_threadsafe_queue<teca_data_request_task> m_queue;
    vector<future<p_teca_dataset>> m_dataset_futures;
    vector<thread> m_threads;
};


// --------------------------------------------------------------------------
teca_thread_pool::teca_thread_pool() : m_live(true)
{
    unsigned int n = std::max(1u, thread::hardware_concurrency()-1);
    this->create_threads(n);
}

// --------------------------------------------------------------------------
teca_thread_pool::teca_thread_pool(unsigned int n) : m_live(true)
{
    this->create_threads(n);
}

// --------------------------------------------------------------------------
void teca_thread_pool::create_threads(unsigned int n_threads)
{
    for (unsigned int i = 0; i < n_threads; ++i)
    {
        m_threads.push_back(thread([this]()
        {
            // "main" for each thread in the pool
            while (m_live.load())
            {
                teca_data_request_task task;
                if (m_queue.try_pop(task))
                    task();
                else
                    std::this_thread::yield();
            }
        }));
    }
}

// --------------------------------------------------------------------------
teca_thread_pool::~teca_thread_pool()
{
    m_live = false;
    std::for_each(
        m_threads.begin(), m_threads.end(),
        [](thread &t) { t.join(); });
}

// --------------------------------------------------------------------------
void teca_thread_pool::push_data_request(
        const p_teca_algorithm &alg,
        const teca_algorithm_output_port &up_port,
        const teca_meta_data &up_req)
{
    teca_data_request dreq(alg, up_port, up_req);
    teca_data_request_task task(dreq);
    m_dataset_futures.push_back(task.get_future());
    m_queue.push(std::move(task));
}

// --------------------------------------------------------------------------
void teca_thread_pool::pop_datasets(vector<p_teca_dataset> &data)
{
    // wait on all pending requests and gather the generated
    // datasets
    size_t n = m_dataset_futures.size();
    std::for_each(
        m_dataset_futures.begin(),
        m_dataset_futures.end(),
        [&data] (future<p_teca_dataset> &f)
        {
            data.push_back(f.get());
        });
    m_dataset_futures.clear();
}




// internals for teca threaded algorithm
class teca_threaded_algorithm_internals
{
public:
    teca_threaded_algorithm_internals()
        : thread_pool(new teca_thread_pool(1)) {}

    void thread_pool_resize(unsigned int n);

    unsigned int get_thread_pool_size() const TECA_NOEXCEPT
    { return this->thread_pool->size(); }

public:
    p_teca_thread_pool thread_pool;
};

// --------------------------------------------------------------------------
void teca_threaded_algorithm_internals::thread_pool_resize(unsigned int n)
{
    if (this->thread_pool->size() != n)
        this->thread_pool = std::make_shared<teca_thread_pool>(n);
}











// --------------------------------------------------------------------------
teca_threaded_algorithm::teca_threaded_algorithm()
    : internals(new teca_threaded_algorithm_internals)
{}

// --------------------------------------------------------------------------
teca_threaded_algorithm::~teca_threaded_algorithm() TECA_NOEXCEPT
{
    delete this->internals;
}

// --------------------------------------------------------------------------
void teca_threaded_algorithm::set_thread_pool_size(unsigned int n)
{
    this->internals->thread_pool_resize(n);
}

// --------------------------------------------------------------------------
unsigned int teca_threaded_algorithm::get_thread_pool_size() const TECA_NOEXCEPT
{
    return this->internals->get_thread_pool_size();
}

// --------------------------------------------------------------------------
p_teca_dataset teca_threaded_algorithm::request_data(
    teca_algorithm_output_port &current,
    const teca_meta_data &request)
{
    // execute current algorithm to fulfill the request.
    // return the data
    p_teca_algorithm alg = get_algorithm(current);
    unsigned int port = get_port(current);

    // check for cached data
    teca_meta_data key = alg->get_cache_key(port, request);
    p_teca_dataset out_data = alg->get_output_data(port, key);
    if (!out_data)
    {
        // determine what data is available on our inputs
        unsigned int n_inputs = alg->get_number_of_input_connections();
        vector<teca_meta_data> input_md(n_inputs);
        for (unsigned int i = 0; i < n_inputs; ++i)
        {
            input_md[i]
              = alg->get_output_meta_data(alg->get_input_connection(i));
        }

        // get requests for upstream data
        vector<teca_meta_data> up_reqs
            = alg->get_upstream_request(port, input_md, request);

        // push data requests on to the thread pool's work
        // queue. mapping the requests round-robbin on to
        // the inputs
        size_t n_up_reqs = up_reqs.size();

        p_teca_thread_pool &
            work_queue = this->internals->thread_pool;

        for (unsigned int i = 0; i < n_up_reqs; ++i)
        {
            if (!up_reqs[i].empty())
            {
                teca_algorithm_output_port &up_port
                    = alg->get_input_connection(i%n_inputs);

                work_queue->push_data_request(
                    get_algorithm(up_port), up_port, up_reqs[i]);
            }
        }

        // get the requested data. will block until it's ready.
        vector<p_teca_dataset> input_data;
        work_queue->pop_datasets(input_data);

        // execute override
        out_data = alg->execute(port, input_data, request);

        // cache the output
        alg->cache_output_data(port, key, out_data);
    }

    return out_data;
}
