#ifndef teca_algorithm_h
#define teca_algorithm_h

#include "teca_config.h"

// forward delcaration of ref counted types
#include "teca_dataset_fwd.h"
#include "teca_algorithm_fwd.h"
#include "teca_algorithm_executive_fwd.h"
class teca_algorithm_internals;

// for types used in the api
#include "teca_metadata.h"
#include "teca_algorithm_output_port.h"
#include "teca_program_options.h"

#include <vector>
#include <utility>
#include <iosfwd>

// interface to teca pipeline architecture. all sources/readers
// filters, sinks/writers will implement this interface
class teca_algorithm : public std::enable_shared_from_this<teca_algorithm>
{
public:
    // construct/destruct
    static p_teca_algorithm New();
    virtual ~teca_algorithm() noexcept;
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_algorithm)

#if defined(TECA_HAS_BOOST)
    // initialize the given options description
    // with algorithm's properties
    virtual void get_properties_description(const std::string &, options_description &)
    {}

    // initialize the algorithm from the given options
    // variable map.
    virtual void set_properties(const std::string &, variables_map &)
    {}
#endif

    // get an output port from the algorithm. to be used
    // during pipeline building
    virtual
    teca_algorithm_output_port get_output_port(unsigned int port = 0);

    // set an input to this algorithm
    void set_input_connection(const teca_algorithm_output_port &port)
    { this->set_input_connection(0, port); }

    virtual
    void set_input_connection(unsigned int id,
        const teca_algorithm_output_port &port);

    // remove input connections
    virtual
    void remove_input_connection(unsigned int id);

    // remove all input connections
    void clear_input_connections();

    // access the cached data produced by this algorithm. when no
    // request is specified the dataset on the top(most recent) of
    // the cache is returned. When a request is specified it may
    // optionally be filtered by the implementations cache key filter.
    // see also get_cache_key (threadsafe)
    const_p_teca_dataset get_output_data(unsigned int port = 0);

    // remove a dataset from the top/bottom of the cache. the
    // top of the cache has the most recently created dataset.
    // top or bottom is selected via the boolean argument.
    // (threadsafe)
    void pop_cache(unsigned int port = 0, int top = 0);

    // set the cache size. the default is 1. (threadsafe)
    void set_cache_size(unsigned int n);

    // execute the pipeline from this instance up.
    virtual int update();
    virtual int update(unsigned int port);

    // get meta data considering this instance up.
    virtual teca_metadata update_metadata(unsigned int port = 0);

    // set the executive
    void set_executive(p_teca_algorithm_executive exe);
    p_teca_algorithm_executive get_executive();

    // serialize the configuration to a stream. this should
    // store the public user modifiable properties so that
    // runtime configuration may be saved and restored..
    virtual void to_stream(std::ostream &s) const;
    virtual void from_stream(std::istream &s);

protected:
    teca_algorithm();

    // implementations should call this from their constructors
    // to setup the internal caches and data structures required
    // for execution.
    void set_number_of_input_connections(unsigned int n);
    void set_number_of_output_ports(unsigned int n);

    // set the modified flag on the given output port's cache.
    // should be called when user modifies properties on the
    // object that require the output to be regenerated.
    virtual void set_modified();
    void set_modified(unsigned int port);

protected:
// this section contains methods that developers
// typically need to override when implementing
// teca_algorithm's such as reader, filters, and
// writers.

    // implementations must override this method to provide
    // information to downstream consumers about what data
    // will be produced on each output port. The port to
    // provide information about is named in the first argument
    // the second argument contains a list of the metadata
    // describing data on all of the inputs.
    virtual
    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md);

    // implementations must override this method and
    // generate a set of requests describing the data
    // required on the inputs to produce data for the
    // named output port, given the upstream meta data
    // and request. If no data is needed on an input
    // then the list should contain a null request.
    virtual
    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request);

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
    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request);

    // implementations may choose to override this method
    // to gain control of keys used in the cache. By default
    // the passed in request is used as the key. This overide
    // gives implementor the chance to filter the passed in
    // request.
    virtual
    teca_metadata get_cache_key(unsigned int port,
        const teca_metadata &request) const;

protected:
// this section contains methods that control the
// pipeline's behavior. these would typically only
// need to be overriden when designing a new class
// of algorithms.

    // driver function that manage meta data reporting  phase
    // of pipeline execution.
    virtual
    teca_metadata get_output_metadata(
        teca_algorithm_output_port &current);

    // driver function that manages execution of the given
    // requst on the named port
    virtual
    const_p_teca_dataset request_data(
        teca_algorithm_output_port &port,
        const teca_metadata &request);

    // driver function that clears the output data cache
    // where modified flag has been set from the current
    // port upstream.
    virtual
    int validate_cache(teca_algorithm_output_port &current);

    // driver function that clears the modified flag on the
    // named port and all of it's upstream connections.
    virtual
    void clear_modified(teca_algorithm_output_port current);

protected:
// api exposing internals for use in driver methods

    // search the given port's cache for the dataset associated
    // with the given request. see also get_cache_key. (threadsafe)
    const_p_teca_dataset get_output_data(unsigned int port,
        const teca_metadata &request);

    // add or update the given request , dataset pair in the cache.
    // see also get_cache_key. (threadsafe)
    int cache_output_data(unsigned int port,
        const teca_metadata &request, const_p_teca_dataset &data);

    // clear the cache on the given output port
    void clear_cache(unsigned int port);

    // get the number of input connections
    unsigned int get_number_of_input_connections();

    // get the output port associated with this algorithm's
    // i'th input connection.
    teca_algorithm_output_port &get_input_connection(unsigned int i);

    // clear the modified flag on the i'th output
    void clear_modified(unsigned int port);

    // return the output port's modified flag value
    int get_modified(unsigned int port) const;

private:
    teca_algorithm_internals *internals;

    friend class teca_threaded_algorithm;
    friend class teca_data_request;
};

#endif
