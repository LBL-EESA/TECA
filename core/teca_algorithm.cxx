#include "teca_algorithm.h"
#include "teca_dataset.h"
#include "teca_algorithm_executive.h"
#include "teca_threadsafe_queue.h"

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <utility>
#include <algorithm>
#include <mutex>

using std::vector;
using std::map;
using std::string;
using std::ostream;
using std::istream;
using std::pair;
using std::mutex;


// implementation for managing input connections
// and cached output data
class teca_algorithm_internals
{
public:
    teca_algorithm_internals();
    ~teca_algorithm_internals() noexcept;
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_algorithm_internals)

    // this setsup the output cache. calling
    // this will remove all cached data.
    void set_number_of_inputs(unsigned int n);
    void set_number_of_outputs(unsigned int n);

    unsigned int get_number_of_inputs() const noexcept;
    unsigned int get_number_of_outputs() const noexcept;

    // set/get the input
    teca_algorithm_output_port &get_input(unsigned int i);

    void set_input(
        unsigned int conn,
        const teca_algorithm_output_port &port);

    // insert a dataset into the cache for the
    // given request (thread safe)
    int cache_output_data(
        unsigned int port,
        const teca_metadata &request,
        const_p_teca_dataset data);

    // get a pointer to the cached dataset. if
    // a no dataset is cached for the given request
    // then return is null (thread safe)
    const_p_teca_dataset get_output_data(
        unsigned int port,
        const teca_metadata &request);

    // get a pointer to the "first" cached dataset.
    const_p_teca_dataset get_output_data(unsigned int port);

    // clear the cache (thread safe)
    void clear_data_cache(unsigned int port);

    // remove a dataset at the top or bottom of the cache
    void pop_cache(unsigned int port, int top);

    // sets the maximum nuber of datasets to cache
    // per output port
    void set_data_cache_size(unsigned int n);
    unsigned int get_data_cache_size() const noexcept;

    // set/clear modified flag for the given port
    void set_modified();
    void set_modified(unsigned int port);

    void clear_modified();
    void clear_modified(unsigned int port);

    // get the modified state of the given port
    int get_modified(unsigned int port) const;

    // set/get the executive
    p_teca_algorithm_executive get_executive();
    void set_executive(p_teca_algorithm_executive &exec);

    // set the number of threads
    void set_number_of_threads(unsigned int n_threads);

    // print internal state to the stream
    void to_stream(ostream &os) const;
    void from_stream(istream &is);

    // algorithm description
    string name;

    // links to upstream stages indexed by input
    // connection id
    vector<teca_algorithm_output_port> inputs;

    // cached output data. maps from a request
    // to the cached dataset, one per output port
    unsigned int data_cache_size;
    using req_data_map = map<teca_metadata, const_p_teca_dataset>;
    vector<req_data_map> data_cache;
    mutex data_cache_mutex;

    // flag that indicates if the cache on output port
    // i is invalid
    vector<int> modified;

    // executive
    p_teca_algorithm_executive exec;
};

// --------------------------------------------------------------------------
teca_algorithm_internals::teca_algorithm_internals()
            :
    name("teca_algorithm"),
    data_cache_size(1),
    modified(1),
    exec(teca_algorithm_executive::New())
{
    this->set_number_of_outputs(1);
}

// --------------------------------------------------------------------------
teca_algorithm_internals::~teca_algorithm_internals() noexcept
{}

// --------------------------------------------------------------------------
void teca_algorithm_internals::set_number_of_inputs(unsigned int n)
{
    this->inputs.clear();
    this->inputs.resize(n, teca_algorithm_output_port(nullptr, 0));
}

// --------------------------------------------------------------------------
unsigned int teca_algorithm_internals::get_number_of_inputs() const noexcept
{
    return this->inputs.size();
}

// --------------------------------------------------------------------------
void teca_algorithm_internals::set_input(
    unsigned int conn,
    const teca_algorithm_output_port &port)
{
    this->inputs[conn] = port;
}

// --------------------------------------------------------------------------
teca_algorithm_output_port &teca_algorithm_internals::get_input(
    unsigned int conn)
{
    return this->inputs[conn];
}

// --------------------------------------------------------------------------
void teca_algorithm_internals::set_number_of_outputs(unsigned int n)
{
    if (n < 1)
    {
        TECA_ERROR("invalid number of outputs " << n)
        n = 1;
    }

    // create a chacne for each output
    this->data_cache.clear();
    this->data_cache.resize(n);

    // create a modified flag for each output
    this->modified.clear();
    this->modified.resize(n, 1);
}

// --------------------------------------------------------------------------
unsigned int teca_algorithm_internals::get_number_of_outputs() const noexcept
{
    return this->data_cache.size();
}

// --------------------------------------------------------------------------
void teca_algorithm_internals::set_data_cache_size(unsigned int n)
{
    this->data_cache_size = n;
    unsigned int n_out = this->get_number_of_outputs();
    for (unsigned int i = 0; i < n_out; ++i)
    {
        req_data_map &cache = this->data_cache[i];

        while (cache.size() >  n)
            cache.erase(cache.begin());
    }
}

// --------------------------------------------------------------------------
unsigned int teca_algorithm_internals::get_data_cache_size() const noexcept
{
    return this->data_cache_size;
}

// --------------------------------------------------------------------------
void teca_algorithm_internals::clear_data_cache(unsigned int port)
{
    std::lock_guard<mutex> lock(this->data_cache_mutex);

    this->data_cache[port].clear();
}

// --------------------------------------------------------------------------
void teca_algorithm_internals::pop_cache(unsigned int port, int top)
{
    req_data_map &cache = this->data_cache[port];

    if (cache.empty())
        return;

    if (top)
        cache.erase(--cache.end()); // newest
    else
        cache.erase(cache.begin()); // oldest
}

// --------------------------------------------------------------------------
int teca_algorithm_internals::cache_output_data(
    unsigned int port,
    const teca_metadata &request,
    const_p_teca_dataset data)
{
    if (this->data_cache_size)
    {
        std::lock_guard<mutex> lock(this->data_cache_mutex);

        req_data_map &cache = this->data_cache[port];

        auto res = cache.insert(req_data_map::value_type(request, data));
        if (!res.second)
            res.first->second = data;

        while (cache.size() >= this->data_cache_size)
            cache.erase(cache.begin());
    }
    return 0;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_algorithm_internals::get_output_data(
    unsigned int port,
    const teca_metadata &request)
{
    std::lock_guard<mutex> lock(this->data_cache_mutex);

    req_data_map &cache = this->data_cache[port];

    req_data_map::iterator it = cache.find(request);
    if (it != cache.end())
    {
        return it->second;
    }

    return const_p_teca_dataset();
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_algorithm_internals::get_output_data(unsigned int port)
{
    std::lock_guard<mutex> lock(this->data_cache_mutex);

    req_data_map &cache = this->data_cache[port];

    if (cache.empty())
        return const_p_teca_dataset();

    return cache.rbegin()->second; // newest
}

// --------------------------------------------------------------------------
int teca_algorithm_internals::get_modified(unsigned int port) const
{
    return this->modified[port];
}

// --------------------------------------------------------------------------
void teca_algorithm_internals::set_modified(unsigned int port)
{
    this->modified[port] = 1;
}

// --------------------------------------------------------------------------
void teca_algorithm_internals::set_modified()
{
    std::for_each(
        this->modified.begin(), this->modified.end(),
        [](int &m){ m = 1; });
}

// --------------------------------------------------------------------------
void teca_algorithm_internals::clear_modified(unsigned int port)
{
    this->modified[port] = 0;
}

// --------------------------------------------------------------------------
void teca_algorithm_internals::clear_modified()
{
    std::for_each(
        this->modified.begin(), this->modified.end(),
        [](int &m){ m = 0; });
}

// --------------------------------------------------------------------------
p_teca_algorithm_executive teca_algorithm_internals::get_executive()
{
    return this->exec;
}

// --------------------------------------------------------------------------
void teca_algorithm_internals::set_executive(p_teca_algorithm_executive &e)
{
    this->exec = e;
}

// --------------------------------------------------------------------------
void teca_algorithm_internals::to_stream(ostream &os) const
{
    // TODO
    (void) os;
}

// --------------------------------------------------------------------------
void teca_algorithm_internals::from_stream(istream &is)
{
    // TODO
    (void) is;
}









// --------------------------------------------------------------------------
p_teca_algorithm teca_algorithm::New()
{
    return p_teca_algorithm(new teca_algorithm);
}

// --------------------------------------------------------------------------
teca_algorithm::teca_algorithm() : internals(new teca_algorithm_internals)
{}

// --------------------------------------------------------------------------
teca_algorithm::~teca_algorithm() noexcept
{
    delete this->internals;
}

// --------------------------------------------------------------------------
teca_algorithm_output_port teca_algorithm::get_output_port(
    unsigned int port)
{
    return teca_algorithm_output_port(this->shared_from_this(), port);
}

// --------------------------------------------------------------------------
void teca_algorithm::set_input_connection(
    unsigned int conn,
    const teca_algorithm_output_port &upstream)
{
    this->internals->set_input(conn, upstream);
    this->set_modified();
}

// --------------------------------------------------------------------------
teca_algorithm_output_port &teca_algorithm::get_input_connection(
    unsigned int i)
{
    return this->internals->get_input(i);
}

// --------------------------------------------------------------------------
void teca_algorithm::remove_input_connection(unsigned int id)
{
    this->set_input_connection(id, teca_algorithm_output_port(nullptr, 0));
    this->set_modified();
}

// --------------------------------------------------------------------------
void teca_algorithm::clear_input_connections()
{
    unsigned int n = this->internals->get_number_of_inputs();
    for (unsigned int i=0; i<n; ++i)
    {
        this->remove_input_connection(i);
    }
    this->set_modified();
}

// --------------------------------------------------------------------------
int teca_algorithm::cache_output_data(
    unsigned int port,
    const teca_metadata &key,
    const_p_teca_dataset &data)
{
    return this->internals->cache_output_data(port, key, data);
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_algorithm::get_output_data(
    unsigned int port,
    const teca_metadata &key)
{
    return this->internals->get_output_data(port, key);
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_algorithm::get_output_data(unsigned int port)
{
    return this->internals->get_output_data(port);
}

// --------------------------------------------------------------------------
void teca_algorithm::clear_cache(unsigned int port)
{
    this->internals->clear_data_cache(port);
}

// --------------------------------------------------------------------------
void teca_algorithm::pop_cache(unsigned int port, int top)
{
    this->internals->pop_cache(port, top);
    this->set_modified(port);
}

// --------------------------------------------------------------------------
void teca_algorithm::set_cache_size(unsigned int n)
{
    if (n == this->internals->get_data_cache_size())
        return;

    this->internals->set_data_cache_size(n);
    this->set_modified();
}

// --------------------------------------------------------------------------
void teca_algorithm::set_modified()
{
    this->internals->set_modified();
}

// --------------------------------------------------------------------------
void teca_algorithm::set_modified(unsigned int port)
{
    this->internals->set_modified(port);
}

// --------------------------------------------------------------------------
int teca_algorithm::get_modified(unsigned int port) const
{
    return this->internals->get_modified(port);
}

// --------------------------------------------------------------------------
void teca_algorithm::clear_modified(unsigned int port)
{
    this->internals->clear_modified(port);
}

// --------------------------------------------------------------------------
void teca_algorithm::set_executive(p_teca_algorithm_executive exec)
{
    this->internals->set_executive(exec);
}

// --------------------------------------------------------------------------
p_teca_algorithm_executive teca_algorithm::get_executive()
{
    return this->internals->get_executive();
}

// --------------------------------------------------------------------------
void teca_algorithm::set_number_of_input_connections(unsigned int n)
{
    this->internals->set_number_of_inputs(n);
}

// --------------------------------------------------------------------------
unsigned int teca_algorithm::get_number_of_input_connections()
{
    return internals->get_number_of_inputs();
}

// --------------------------------------------------------------------------
void teca_algorithm::set_number_of_output_ports(unsigned int n)
{
    this->internals->set_number_of_outputs(n);
}

// --------------------------------------------------------------------------
teca_metadata teca_algorithm::get_output_metadata(
    unsigned int port,
    const vector<teca_metadata> &input_md)
{
    (void)port;

    // the default implementation passes meta data through
    if (input_md.size())
        return input_md[0];

    return teca_metadata();
}

// --------------------------------------------------------------------------
vector<teca_metadata> teca_algorithm::get_upstream_request(
    unsigned int port,
    const vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void) port;
    (void) input_md;

    // default implementation forwards request upstream
    return vector<teca_metadata>(
        this->get_number_of_input_connections(), request);
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_algorithm::execute(
        unsigned int port,
        const vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request)
{
    (void) port;
    (void) input_data;
    (void) request;

    // default implementation does nothing
    return p_teca_dataset();
}

// --------------------------------------------------------------------------
teca_metadata teca_algorithm::get_cache_key(
    unsigned int port,
    const teca_metadata &request) const
{
    (void) port;

    // default implementation passes the request through
    return request;
}

// --------------------------------------------------------------------------
teca_metadata teca_algorithm::get_output_metadata(
    teca_algorithm_output_port &current)
{
    p_teca_algorithm alg = get_algorithm(current);
    unsigned int port = get_port(current);

    // gather upstream metadata one per input
    // connection.

    unsigned int n_inputs = alg->get_number_of_input_connections();
    vector<teca_metadata> input_md(n_inputs);
    for (unsigned int i = 0; i < n_inputs; ++i)
    {
        input_md[i]
          = alg->get_output_metadata(alg->get_input_connection(i));
    }

    // now that we have metadata for the algorithm's
    // inputs, call the override to do the actual work
    // of reporting output meta data
    return alg->get_output_metadata(port, input_md);
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_algorithm::request_data(
    teca_algorithm_output_port &current,
    const teca_metadata &request)
{
    // execute current algorithm to fulfill the request.
    // return the data
    p_teca_algorithm &alg = get_algorithm(current);
    unsigned int port = get_port(current);

    // check for cached data
    teca_metadata key = alg->get_cache_key(port, request);
    const_p_teca_dataset out_data = alg->get_output_data(port, key);
    if (!out_data)
    {
        // determine what data is available on our inputs
        unsigned int n_inputs = alg->get_number_of_input_connections();
        vector<teca_metadata> input_md(n_inputs);
        for (unsigned int i = 0; i < n_inputs; ++i)
        {
            input_md[i]
              = alg->get_output_metadata(alg->get_input_connection(i));
        }

        // get requests for upstream data
        vector<teca_metadata> up_reqs
            = alg->get_upstream_request(port, input_md, request);

        // get the upstream data mapping the requests round-robbin
        // on to the inputs
        size_t n_up_reqs = up_reqs.size();
        vector<const_p_teca_dataset> input_data(n_up_reqs);
        for (unsigned int i = 0; i < n_up_reqs; ++i)
        {
            if (!up_reqs[i].empty())
            {
                teca_algorithm_output_port conn
                    = alg->get_input_connection(i%n_inputs);

                input_data[i]
                    = get_algorithm(conn)->request_data(conn, up_reqs[i]);
            }
        }

        // execute override
        out_data = alg->execute(port, input_data, request);

        // cache
        alg->cache_output_data(port, key, out_data);
    }

    return out_data;
}

// --------------------------------------------------------------------------
int teca_algorithm::validate_cache(teca_algorithm_output_port &current)
{
    p_teca_algorithm alg = get_algorithm(current);
    unsigned int port = get_port(current);

    unsigned int n = alg->get_number_of_input_connections();
    for (unsigned int i = 0; i<n; ++i)
    {
        teca_algorithm_output_port upstream = alg->get_input_connection(i);

        if (alg->validate_cache(upstream) || alg->get_modified(port))
        {
            alg->clear_cache(port);
            return 1;
        }
    }

    // only if no upstream has been modified can we
    // report the cache is valid
    return 0;
}

// --------------------------------------------------------------------------
void teca_algorithm::clear_modified(teca_algorithm_output_port current)
{
    p_teca_algorithm alg = get_algorithm(current);
    unsigned int port = get_port(current);

    unsigned int n = alg->get_number_of_input_connections();
    for (unsigned int i = 0; i < n; ++i)
    {
        alg->clear_modified(alg->get_input_connection(i));
    }

    alg->clear_modified(port);
}

// --------------------------------------------------------------------------
int teca_algorithm::update(unsigned int port_id)
{
    teca_algorithm_output_port port = this->get_output_port(port_id);

    // make sure caches are wiped where inputs have changed
    this->validate_cache(port);

    // initialize the executive
    p_teca_algorithm_executive exec = this->internals->get_executive();
    if (exec->initialize(this->get_output_metadata(port)))
    {
        TECA_ERROR("failed to initialize the executive")
        return -1;
    }

    // issue requests for data
    teca_metadata request;
    while ((request = exec->get_next_request()))
        this->request_data(port, request);

    // clear modfied flags
    this->clear_modified(port);

    return 0;
}

// --------------------------------------------------------------------------
int teca_algorithm::update()
{
    // produce data on each of our outputs
    unsigned int n_out = this->internals->get_number_of_outputs();
    for (unsigned int i = 0; i < n_out; ++i)
    {
        if (this->update(i))
        {
            TECA_ERROR("failed to update port " << i)
            return -1;
        }
    }
    return 0;
}

// --------------------------------------------------------------------------
teca_metadata teca_algorithm::update_metadata(unsigned int port_id)
{
    teca_algorithm_output_port port = this->get_output_port(port_id);
    return this->get_output_metadata(port);
}

// --------------------------------------------------------------------------
void teca_algorithm::to_stream(ostream &os) const
{
    this->internals->to_stream(os);
}

// --------------------------------------------------------------------------
void teca_algorithm::from_stream(istream &is)
{
    this->internals->from_stream(is);
}
