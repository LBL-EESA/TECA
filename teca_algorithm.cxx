#include "teca_algorithm.h"

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <utility>
#include <algorithm>

using std::vector;
using std::map;
using std::string;
using std::ostream;
using std::istream;
using std::pair;

// implementation for managing input connections
// and cached output data
class teca_algorithm_internals
{
public:
    teca_algorithm_internals();
    ~teca_algorithm_internals();

    teca_algorithm_internals(const teca_algorithm_internals &other);
    teca_algorithm_internals(teca_algorithm_internals &&other);

    teca_algorithm_internals &operator=(const teca_algorithm_internals &other);
    teca_algorithm_internals &operator=(teca_algorithm_internals &&other);

    // this setsup the output cache. calling
    // this will remove all cached data.
    void set_number_of_inputs(unsigned int n);
    void set_number_of_outputs(unsigned int n);

    unsigned int get_number_of_inputs();
    unsigned int get_number_of_outputs();

    // set/get the input
    teca_algorithm::output_port_t &get_input(unsigned int i);

    void set_input(
        unsigned int conn,
        const teca_algorithm::output_port_t &port);

    // insert a dataset into the cache for the
    // given request
    int cache_output_data(
        unsigned int port,
        const teca_meta_data &request,
        p_teca_dataset data);

    // get a pointer to the cached dataset. if
    // a no dataset is cached for the given request
    // then return is null
    p_teca_dataset get_output_data(
        unsigned int port,
        const teca_meta_data &request);

    // get a pointer to the "first" cached dataset.
    p_teca_dataset get_output_data(unsigned int port);

    // clear the cache
    void clear_data_cache(unsigned int port);

    // sets the maximum nuber of datasets to cache
    // per output port
    void set_data_cache_size(unsigned int n);

    // set/clear modified flag for the given port
    void set_modified();
    void set_modified(unsigned int port);

    void clear_modified();
    void clear_modified(unsigned int port);

    // get the modified state of the given port
    int get_modified(unsigned int port);

    // print internal state to the stream
    void to_stream(ostream &os);
    void from_stream(istream &is);

    // algorithm description
    string name;

    // links to upstream stages indexed by input
    // connection id
    vector<teca_algorithm::output_port_t> inputs;

    // cached output data. maps from a request
    // to the cached dataset, one per output port
    unsigned int data_cache_size;
    typedef map<teca_meta_data, p_teca_dataset> req_data_map;
    vector<req_data_map> data_cache;

    // flag that indicates if the cache on output port
    // i is invalid
    vector<int> modified;
};

// --------------------------------------------------------------------------
teca_algorithm_internals::teca_algorithm_internals()
            :
    name("teca_algorithm"),
    data_cache_size(1),
    modified(1)
{}

// --------------------------------------------------------------------------
teca_algorithm_internals::~teca_algorithm_internals()
{}

// --------------------------------------------------------------------------
teca_algorithm_internals::teca_algorithm_internals(
    const teca_algorithm_internals &other)
            :
    name(other.name),
    inputs(other.inputs),
    data_cache_size(other.data_cache_size),
    data_cache(other.data_cache),
    modified(other.modified)
{}

// --------------------------------------------------------------------------
teca_algorithm_internals::teca_algorithm_internals(
    teca_algorithm_internals &&other)
{
    this->name.swap(other.name);
    this->inputs.swap(other.inputs);
    this->data_cache_size = other.data_cache_size;
    this->data_cache.swap(other.data_cache);
    this->modified.swap(other.modified);
}

// --------------------------------------------------------------------------
teca_algorithm_internals &teca_algorithm_internals::operator=(
    const teca_algorithm_internals &other)
{
    if (&other == this)
        return *this;

    this->name = other.name;
    this->inputs = other.inputs;
    this->data_cache_size = other.data_cache_size;
    this->data_cache = other.data_cache;
    this->modified = other.modified;

    return *this;
}

// --------------------------------------------------------------------------
teca_algorithm_internals &teca_algorithm_internals::operator=(
    teca_algorithm_internals &&other)
{
    if (&other == this) return *this;

    this->name.clear();
    this->name.swap(other.name);

    this->inputs.clear();
    this->inputs.swap(other.inputs);

    this->data_cache_size = other.data_cache_size;

    this->data_cache.clear();
    this->data_cache.swap(other.data_cache);

    this->modified.clear();
    this->modified.swap(other.modified);

    return *this;
}

// --------------------------------------------------------------------------
void teca_algorithm_internals::set_number_of_inputs(unsigned int n)
{
    this->inputs.clear();
    this->inputs.resize(n, teca_algorithm::output_port_t(nullptr, 0));
}

// --------------------------------------------------------------------------
unsigned int teca_algorithm_internals::get_number_of_inputs()
{
    return this->inputs.size();
}

// --------------------------------------------------------------------------
void teca_algorithm_internals::set_input(
    unsigned int conn,
    const teca_algorithm::output_port_t &port)
{
    this->inputs[conn] = port;
}

// --------------------------------------------------------------------------
teca_algorithm::output_port_t &teca_algorithm_internals::get_input(
    unsigned int conn)
{
    return this->inputs[conn];
}

// --------------------------------------------------------------------------
void teca_algorithm_internals::set_number_of_outputs(unsigned int n)
{
    this->data_cache.clear();
    this->data_cache.resize(n);

    this->modified.clear();
    this->modified.resize(n, 1);
}

// --------------------------------------------------------------------------
unsigned int teca_algorithm_internals::get_number_of_outputs()
{
    return this->data_cache.size();
}

// --------------------------------------------------------------------------
void teca_algorithm_internals::clear_data_cache(unsigned int port)
{
    this->data_cache[port].clear();
}

// --------------------------------------------------------------------------
int teca_algorithm_internals::cache_output_data(
    unsigned int port,
    const teca_meta_data &request,
    p_teca_dataset data)
{
    // TODO -- deal with cache size
    req_data_map &cache = this->data_cache[port];
    pair<req_data_map::iterator, bool> res
        = cache.insert(req_data_map::value_type(request, data));
    if (!res.second)
    {
        res.first->second = data;
    }
    return 0;
}

// --------------------------------------------------------------------------
p_teca_dataset teca_algorithm_internals::get_output_data(
    unsigned int port,
    const teca_meta_data &request)
{
    req_data_map &cache = this->data_cache[port];
    req_data_map::iterator it = cache.find(request);
    if (it != cache.end())
    {
        return it->second;
    }
    return p_teca_dataset(nullptr);
}

// --------------------------------------------------------------------------
p_teca_dataset teca_algorithm_internals::get_output_data(unsigned int port)
{
    req_data_map &cache = this->data_cache[port];
    req_data_map::iterator it = cache.begin();
    if (it != cache.end())
    {
        return it->second;
    }
    return p_teca_dataset(nullptr);
}

// --------------------------------------------------------------------------
int teca_algorithm_internals::get_modified(unsigned int port)
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
        this->modified.begin(),
        this->modified.end(),
        [](int &m){ m = 1; });
}

// --------------------------------------------------------------------------
void teca_algorithm_internals::clear_modified(unsigned int port)
{
    this->modified[port] = 1;
}

// --------------------------------------------------------------------------
void teca_algorithm_internals::clear_modified()
{
    std::for_each(
        this->modified.begin(),
        this->modified.end(),
        [](int &m){ m = 0; });
}

// --------------------------------------------------------------------------
void teca_algorithm_internals::to_stream(ostream &os)
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



// helpers
namespace {

p_teca_algorithm &algorithm(teca_algorithm::output_port_t &op)
{ return op.first; }

unsigned int &port(teca_algorithm::output_port_t &op)
{ return op.second; }

};




// --------------------------------------------------------------------------
p_teca_algorithm teca_algorithm::New()
{
    return p_teca_algorithm(new teca_algorithm);
}

// --------------------------------------------------------------------------
teca_algorithm::teca_algorithm()
{
    this->internals = new teca_algorithm_internals;
}

// --------------------------------------------------------------------------
teca_algorithm::~teca_algorithm()
{
    delete this->internals;
}

// --------------------------------------------------------------------------
teca_algorithm::teca_algorithm(const teca_algorithm &src) :
    internals(new teca_algorithm_internals(*src.internals))
{}

// --------------------------------------------------------------------------
teca_algorithm::teca_algorithm(teca_algorithm &&src) :
    internals(new teca_algorithm_internals(std::move(*src.internals)))
{}

// --------------------------------------------------------------------------
teca_algorithm &teca_algorithm::operator=(const teca_algorithm &src)
{
    if (this == &src)
        return *this;

    teca_algorithm_internals *tmp
        = new teca_algorithm_internals(*src.internals);

    delete this->internals;
    this->internals = tmp;

    return *this;
}

// --------------------------------------------------------------------------
teca_algorithm &teca_algorithm::operator=(teca_algorithm &&src)
{
    if (this == &src)
        return *this;

    *this->internals = std::move(*src.internals);

    return *this;
}

// --------------------------------------------------------------------------
teca_algorithm::output_port_t teca_algorithm::get_output_port(
    unsigned int port)
{
    return teca_algorithm::output_port_t(this->shared_from_this(), port);
}

// --------------------------------------------------------------------------
void teca_algorithm::set_input_connection(
    unsigned int conn,
    const output_port_t &upstream)
{
    this->internals->set_input(conn, upstream);
}

// --------------------------------------------------------------------------
void teca_algorithm::remove_input_connection(unsigned int id)
{
    this->set_input_connection(id, output_port_t(nullptr, 0));
}

// --------------------------------------------------------------------------
void teca_algorithm::clear_input_connections()
{
    unsigned int n = this->internals->get_number_of_inputs();
    for (unsigned int i=0; i<n; ++i)
    {
        this->remove_input_connection(i);
    }
}

// --------------------------------------------------------------------------
p_teca_dataset teca_algorithm::get_output_data(
    unsigned int port,
    const teca_meta_data &request)
{
    return this->internals->get_output_data(port, request);
}

// --------------------------------------------------------------------------
p_teca_dataset teca_algorithm::get_output_data(unsigned int port)
{
    return this->internals->get_output_data(port);
}

// --------------------------------------------------------------------------
void teca_algorithm::set_number_of_inputs(unsigned int n)
{
    return this->internals->set_number_of_inputs(n);
}

// --------------------------------------------------------------------------
void teca_algorithm::set_number_of_outputs(unsigned int n)
{
    return this->internals->set_number_of_outputs(n);
}

// --------------------------------------------------------------------------
teca_meta_data teca_algorithm::get_output_meta_data(output_port_t &current)
{
    // gather upstream metadata one per input
    // connection.

    p_teca_algorithm alg = ::algorithm(current) ?
        ::algorithm(current) : this->shared_from_this();

    unsigned int n_in = alg->internals->get_number_of_inputs();
    vector<teca_meta_data> in_md(n_in);
    for (unsigned int i = 0; i < n_in; ++i)
    {
        in_md[i]
          = this->get_output_meta_data(alg->internals->get_input(i));
    }

    // now that we have metadata for the algorithm's
    // inputs, call the override to do the actual work
    // of reporting output meta data
    return this->get_output_meta_data(::port(current), in_md);
}

// --------------------------------------------------------------------------
teca_meta_data teca_algorithm::get_output_meta_data(
    unsigned int port,
    vector<teca_meta_data> &input_md)
{
    // implementations must override. the default implementation
    // passes through upstream metadata.
    if (input_md.size())
    {
        // pass meta data through
        return input_md[0];
    }
    // return an empty meta data
    return teca_meta_data();
}

// --------------------------------------------------------------------------
p_teca_dataset teca_algorithm::request_data(
    output_port_t &current,
    teca_meta_data &request)
{
    // execute current algorithm to fulfill the request.
    // return the data
    p_teca_algorithm &alg = ::algorithm(current);
    unsigned int port = ::port(current);

    // check for cached data
    p_teca_dataset out_data = alg->internals->get_output_data(port);
    if (!out_data)
    {
        // determine what data is available on our inputs
        unsigned int n_in = alg->internals->get_number_of_inputs();
        vector<teca_meta_data> in_md(n_in);
        for (unsigned int i = 0; i < n_in; ++i)
        {
            in_md[i]
              = this->get_output_meta_data(alg->internals->get_input(i));
        }

        // get requests for upstream data
        vector<teca_meta_data> req
            = alg->get_upstream_request(port, in_md, request);

        // get upstream data
        vector<p_teca_dataset> in_data(n_in);
        for (unsigned int i = 0; i < n_in; ++i)
        {
            if (!req[i].empty())
            {
                // input connection i contributes to the output
                in_data[i]
                  = this->request_data(alg->internals->get_input(i), req[i]);
            }
        }

        // execute and cache
        out_data = alg->execute(port, in_data, request);
        alg->internals->cache_output_data(port, request, out_data);
    }

    return out_data;
}

// --------------------------------------------------------------------------
p_teca_dataset teca_algorithm::execute(
        unsigned int port,
        vector<p_teca_dataset> &input_data,
        teca_meta_data &request)
{
    // default implementation does nothing, developers
    // must override
    (void) port;
    (void) input_data;
    (void) request;

    return p_teca_dataset();
}

// --------------------------------------------------------------------------
vector<teca_meta_data> teca_algorithm::get_upstream_request(
    unsigned int port,
    vector<teca_meta_data> &input_md,
    teca_meta_data &request)
{
    // default implementation returns an empty request, developers
    // must override
    (void) port;
    (void) input_md;
    (void) request;

    return vector<teca_meta_data>(this->internals->get_number_of_inputs());
}

// --------------------------------------------------------------------------
int teca_algorithm::validate_cache(output_port_t &current)
{
    p_teca_algorithm alg = ::algorithm(current) ?
        ::algorithm(current) : this->shared_from_this();

    unsigned int n = alg->internals->get_number_of_inputs();
    for (unsigned int i = 0; i<n; ++i)
    {
        output_port_t upstream = alg->internals->get_input(i);

        if (alg->validate_cache(upstream)
            || alg->internals->get_modified(::port(current)))
        {
            this->internals->clear_data_cache(::port(current));
            return 1;
        }
    }

    // only if no upstream has been modified can we
    // report the cache is valid
    return 0;
}

// --------------------------------------------------------------------------
void teca_algorithm::clear_modified(output_port_t &current)
{
    p_teca_algorithm alg = ::algorithm(current) ?
        ::algorithm(current) : this->shared_from_this();

    unsigned int n = alg->internals->get_number_of_inputs();
    for (unsigned int i = 0; i<n; ++i)
    {
        this->clear_modified(alg->internals->get_input(i));
    }

    this->internals->clear_modified(::port(current));
}

// --------------------------------------------------------------------------
int teca_algorithm::update()
{
    // produce data on each of our outputs
    unsigned int n_out = this->internals->get_number_of_outputs();
    for (unsigned int i = 0; i < n_out; ++i)
    {
        output_port_t port = this->get_output_port(i);

        // make sure caches are wiped where inputs have changed
        this->validate_cache(port);

        // execute
        // TODO -- need to do something about where request
        // comes from. maybe use "strategy pattern" or this
        // method should be virtual
        teca_meta_data request;
        this->request_data(port, request);

        // clear modfied flags
        this->clear_modified(this->get_output_port(i));
    }
    return 0;
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
void teca_algorithm::to_stream(ostream &os)
{
    this->internals->to_stream(os);
}

// --------------------------------------------------------------------------
void teca_algorithm::from_stream(istream &is)
{
    this->internals->from_stream(is);
}
