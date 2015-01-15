#ifndef teca_algorithm_h
#define teca_algorithm_h

#include <vector>
#include <memory>
#include <utility>
#include <iosfwd>

// forward delcaration of ref counted types
#include "teca_dataset_fwd.h"
#include "teca_algorithm_fwd.h"
#include "teca_algorithm_executive_fwd.h"
class teca_algorithm_internals;
class teca_algorithm_thread_pool;

#include "teca_meta_data.h"

// interface to teca pipeline architecture. all sources/readers
// filters, sinks/writers will implement this interface
class teca_algorithm : public std::enable_shared_from_this<teca_algorithm>
{
public:
    // convinience method for creating the algorithm
    // in a shared pointer.
    static p_teca_algorithm New();
    virtual ~teca_algorithm();

    // convenience method tp get an output port from
    // the algorithm to be used during pipeline building
    typedef std::pair<p_teca_algorithm, unsigned int> output_port_t;
    output_port_t get_output_port(unsigned int port = 0);

    // set an input to this algorithm
    void set_input_connection(const output_port_t &port)
    { this->set_input_connection(0, port); }

    void set_input_connection(
        unsigned int id,
        const output_port_t &port);

    // remove input connections
    void remove_input_connection(unsigned int id);
    void clear_input_connections();

    // access the cached data produced by this algorithm.
    // when no request is specified the dataset on the
    // top(most recent) of the cache is returned. When
    // a request is specified it may optionally be filtered
    // by the implementations cache key filter. see also
    // get_cache_key
    p_teca_dataset get_output_data(unsigned int port = 0);

    p_teca_dataset get_output_data(
        unsigned int port,
        const teca_meta_data &request,
        int filter = 1);

    // remove a dataset from the top/bottom of the cache. the
    // top of the cache has the most recently created dataset.
    // top or bottom is selected via the boolean argument.
    void pop_cache(unsigned int port = 0, int top = 0);

    // set the cache size. the default is 1.
    void set_cache_size(unsigned int n);

    // execute the pipeline from this instance up.
    virtual int update();

    // set the executive
    void set_executive(p_teca_algorithm_executive exec);

    // set the number of threads to use (default 1)
    void set_number_of_threads(unsigned int);

protected:
    teca_algorithm();

    teca_algorithm(const teca_algorithm &src) = delete;
    teca_algorithm(teca_algorithm &&src) = delete;

    teca_algorithm &operator=(const teca_algorithm &src) = delete;
    teca_algorithm &operator=(teca_algorithm &&src) = delete;

    // implementations should call this from their
    // constructors to setup the internal caches
    // and data structures required for execution.
    void set_number_of_inputs(unsigned int n);
    void set_number_of_outputs(unsigned int n);

    // set the modified flag on the given output
    // port's cache. should be called when user
    // modifies properties on the object
    // that require the output to be regenerated.
    void set_modified();
    void set_modified(unsigned int port);

private:
    // implementations must override this method to provide
    // information to downstream consumers about what data
    // will be produced on each output port. The port to
    // provide information about is named in the first argument
    // the second argument contains a list of the metadata
    // describing data on all of the inputs.
    virtual
    teca_meta_data get_output_meta_data(
        unsigned int port,
        const std::vector<teca_meta_data> &input_md);

    // implementations must override this method and
    // generate a set of requests describing the data
    // required on the inputs to produce data for the
    // named output port, given the upstream meta data
    // and request. If no data is needed on an input
    // then the list should contain a null request.
    virtual
    std::vector<teca_meta_data> get_upstream_request(
        unsigned int port,
        const std::vector<teca_meta_data> &input_md,
        const teca_meta_data &request);

    // implementations must override this method and
    // produce the output dataset for the port named
    // in the first argument. The second argument is
    // a list of all of the input datasets. See also
    // get_request. The third argument contains a request
    // from the consumer which can spcify information
    // such as arrays, subset region, timestep etc.
    // The implementation is free to handle the request
    // as it sees fit.
    virtual
    p_teca_dataset execute(
        unsigned int port,
        const std::vector<p_teca_dataset> &input_data,
        const teca_meta_data &request);

    // implementations may choose to override this method
    // to gain control of keys used in the cache. By default
    // the passed in request is used as the key. This overide
    // gives implementor the chance to filter the passed in
    // request.
    virtual
    teca_meta_data get_cache_key(
        unsigned int port,
        const teca_meta_data &request) const;

    // driver function that manage meta data reporting  phase
    // of pipeline execution.
    static
    teca_meta_data get_output_meta_data(output_port_t &current);

    // driver function that manages execution of the given
    // requst on the named port
    static
    p_teca_dataset request_data(
        output_port_t &port,
        const teca_meta_data &request);

    // driver function that clears the output data cache
    //  where modified flag has been set
    static
    int validate_cache(output_port_t &current);

    // clear the modified flag after execution
    static
    void clear_modified(output_port_t current);

    // serialize the configuration to a stream
    virtual void to_stream(std::ostream &os) const;
    virtual void from_stream(std::istream &is);

private:
    teca_algorithm_internals *internals;

    friend class teca_algorithm_thread_pool;
};

#endif
