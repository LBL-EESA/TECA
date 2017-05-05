#include "teca_threaded_algorithm.h"
#include "teca_metadata.h"
#include "teca_thread_pool.h"


#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <future>
#include <cstdlib>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

// function that executes the data request and returns the
// requested dataset
class teca_data_request
{
public:
    teca_data_request(const p_teca_algorithm &alg,
        const teca_algorithm_output_port up_port,
        const teca_metadata &up_req) : m_alg(alg),
        m_up_port(up_port), m_up_req(up_req)
    {}

    const_p_teca_dataset operator()()
    { return m_alg->request_data(m_up_port, m_up_req); }

public:
    p_teca_algorithm m_alg;
    teca_algorithm_output_port m_up_port;
    teca_metadata m_up_req;
};

// task
using teca_data_request_task = std::packaged_task<const_p_teca_dataset()>;

using teca_data_request_queue =
    teca_thread_pool<teca_data_request_task, const_p_teca_dataset>;

using p_teca_data_request_queue = std::shared_ptr<teca_data_request_queue>;

// internals for teca threaded algorithm
class teca_threaded_algorithm_internals
{
public:
    teca_threaded_algorithm_internals() :
        thread_pool(new teca_data_request_queue(-1, false, true, false))
     {}

    void thread_pool_resize(int n, bool local, bool bind, bool verbose);

    unsigned int get_thread_pool_size() const noexcept
    { return this->thread_pool->size(); }

public:
    p_teca_data_request_queue thread_pool;
};

// --------------------------------------------------------------------------
void teca_threaded_algorithm_internals::thread_pool_resize(int n, bool local,
    bool bind, bool verbose)
{
    this->thread_pool = std::make_shared<teca_data_request_queue>(n,
        local, bind, verbose);
}




// --------------------------------------------------------------------------
teca_threaded_algorithm::teca_threaded_algorithm() : verbose(0),
    bind_threads(1), internals(new teca_threaded_algorithm_internals)
{
}

// --------------------------------------------------------------------------
teca_threaded_algorithm::~teca_threaded_algorithm() noexcept
{
    delete this->internals;
}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_threaded_algorithm::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_threaded_algorithm":prefix));

    opts.add_options()
        TECA_POPTS_GET(int, prefix, bind_threads,
            "bind software threads to hardware cores (1)")
        TECA_POPTS_GET(int, prefix, verbose,
            "print a run time report of settings (0)")
        TECA_POPTS_GET(int, prefix, thread_pool_size,
            "number of threads in pool. When n == -1, 1 thread per core is created (-1)")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_threaded_algorithm::set_properties(const std::string &prefix,
    variables_map &opts)
{
    TECA_POPTS_SET(opts, int, prefix, bind_threads)
    TECA_POPTS_SET(opts, int, prefix, verbose)

    std::string opt_name = (prefix.empty()?"":prefix+"::") + "thread_pool_size";
    if (opts.count(opt_name))
        this->set_thread_pool_size(opts[opt_name].as<int>());
}
#endif

// --------------------------------------------------------------------------
void teca_threaded_algorithm::set_thread_pool_size(int n)
{
    this->internals->thread_pool_resize(n, false, this->bind_threads, this->verbose);
}

// --------------------------------------------------------------------------
unsigned int teca_threaded_algorithm::get_thread_pool_size() const noexcept
{
    return this->internals->get_thread_pool_size();
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_threaded_algorithm::request_data(
    teca_algorithm_output_port &current,
    const teca_metadata &request)
{
    // execute current algorithm to fulfill the request.
    // return the data
    p_teca_algorithm alg = get_algorithm(current);
    unsigned int port = get_port(current);

    // check for cached data
    teca_metadata key = alg->get_cache_key(port, request);
    const_p_teca_dataset out_data = alg->get_output_data(port, key);
    if (!out_data)
    {
        // determine what data is available on our inputs
        unsigned int n_inputs = alg->get_number_of_input_connections();
        std::vector<teca_metadata> input_md(n_inputs);
        for (unsigned int i = 0; i < n_inputs; ++i)
        {
            input_md[i]
              = alg->get_output_metadata(alg->get_input_connection(i));
        }

        // get requests for upstream data
        std::vector<teca_metadata> up_reqs
            = alg->get_upstream_request(port, input_md, request);

        // push data requests on to the thread pool's work
        // queue. mapping the requests round-robbin on to
        // the inputs
        size_t n_up_reqs = up_reqs.size();

        for (unsigned int i = 0; i < n_up_reqs; ++i)
        {
            if (!up_reqs[i].empty())
            {
                teca_algorithm_output_port &up_port
                    = alg->get_input_connection(i%n_inputs);

                teca_data_request dreq(get_algorithm(up_port), up_port, up_reqs[i]);
                teca_data_request_task task(dreq);

                this->internals->thread_pool->push_task(task);
            }
        }

        // get the requested data. will block until it's ready.
        std::vector<const_p_teca_dataset> input_data;
        this->internals->thread_pool->wait_data(input_data);

        // execute override
        out_data = alg->execute(port, input_data, request);

        // cache the output
        alg->cache_output_data(port, key, out_data);
    }

    return out_data;
}
