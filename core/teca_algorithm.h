#ifndef teca_algorithm_h
#define teca_algorithm_h

#include "teca_config.h"
#include "teca_shared_object.h"
#include "teca_dataset.h"
#include "teca_algorithm_executive.h"
#include "teca_metadata.h"
#include "teca_program_options.h"
#include "teca_mpi.h"

#include <vector>
#include <utility>
#include <iosfwd>
#include <initializer_list>
#include <functional>

class teca_algorithm_internals;

TECA_SHARED_OBJECT_FORWARD_DECL(teca_algorithm)

/// An output port packages an algorithm and a port number
using teca_algorithm_output_port
    = std::pair<p_teca_algorithm, unsigned int>;

/// get the algorithm from the output port
inline
p_teca_algorithm &get_algorithm(teca_algorithm_output_port &op)
{ return op.first; }

/// get port number from the output port
inline
unsigned int &get_port(teca_algorithm_output_port &op)
{ return op.second; }

/* this is a convenience macro to be used to declare a static New method that
 * will be used to construct new objects in shared_ptr's. This manages the
 * details of interoperability with std C++11 shared pointer
 */
#define TECA_ALGORITHM_STATIC_NEW(T)                \
                                                    \
/** Returns an instance of T */                     \
static p_##T New()                                  \
{                                                   \
    return p_##T(new T);                            \
}                                                   \
                                                    \
/** Enables the static constructor */               \
std::shared_ptr<T> shared_from_this()               \
{                                                   \
    return std::static_pointer_cast<T>(             \
        teca_algorithm::shared_from_this());        \
}                                                   \
                                                    \
/** Enables the static constructor */               \
std::shared_ptr<T const> shared_from_this() const   \
{                                                   \
    return std::static_pointer_cast<T const>(       \
        teca_algorithm::shared_from_this());        \
}

#define TECA_ALGORITHM_CLASS_NAME(T)                \
/** returns the name of the class */                \
const char *get_class_name() const override         \
{                                                   \
    return #T;                                      \
}

/** this convenience macro removes copy and assignment operators
 * which generally should not be defined for reference counted types
 */
#define TECA_ALGORITHM_DELETE_COPY_ASSIGN(T)    \
                                                \
    T(const T &src) = delete;                   \
    T(T &&src) = delete;                        \
                                                \
    T &operator=(const T &src) = delete;        \
    T &operator=(T &&src) = delete;

/** convenience macro to declare standard set_NAME/get_NAME methods
 * where NAME is the name of a class member. will manage the
 * algorithm's modified state for the user.
 */
#define TECA_ALGORITHM_PROPERTY(T, NAME)                 \
                                                         \
/** Set the value of the NAME algorithm property */      \
void set_##NAME(const T &v)                              \
{                                                        \
    if (this->NAME != v)                                 \
    {                                                    \
        this->NAME = v;                                  \
        this->set_modified();                            \
    }                                                    \
}                                                        \
                                                         \
/** Get the value of the NAME algorithm property */      \
const T &get_##NAME() const                              \
{                                                        \
    return this->NAME;                                   \
}

/** similar to TECA_ALGORITHM_PROPERTY but prior to setting NAME
 * will call the member function int valididate_NAME(T v). If
 * the value v is valid the fucntion should return 0. If the value
 * is not zero the function should invoke TECA_ERROR with a
 * descriptive message and return non-zero.
 */
#define TECA_ALGORITHM_PROPERTY_V(T, NAME)               \
                                                         \
/** Set the value of the NAME algorithm property */      \
void set_##NAME(const T &v)                              \
{                                                        \
    if (this->validate_ ## NAME (v))                     \
        return;                                          \
                                                         \
    if (this->NAME != v)                                 \
    {                                                    \
        this->NAME = v;                                  \
        this->set_modified();                            \
    }                                                    \
}                                                        \
                                                         \
/** Get the value of the NAME algorithm property */      \
const T &get_##NAME() const                              \
{                                                        \
    return this->NAME;                                   \
}

/** convenience macro to declare standard set_NAME/get_NAME methods
 * where NAME is the name of a class member. will manage the
 * algorithm's modified state for the user.
 */
#define TECA_ALGORITHM_VECTOR_PROPERTY(T, NAME)                           \
                                                                          \
/** get the size of the NAME algorithm vector property */                 \
size_t get_number_of_##NAME##s ()                                         \
{                                                                         \
    return this->NAME##s.size();                                          \
}                                                                         \
                                                                          \
/** append to the NAME algorithm vector property */                       \
void append_##NAME(const T &v)                                            \
{                                                                         \
    this->NAME##s.push_back(v);                                           \
    this->set_modified();                                                 \
}                                                                         \
                                                                          \
/** set the NAME algorithm vector property to a single value */          \
void set_##NAME(const T &v)                                               \
{                                                                         \
    this->set_##NAME##s({v});                                             \
}                                                                         \
                                                                          \
/** set the i-th element of the NAME algorithm vector property */         \
void set_##NAME(size_t i, const T &v)                                     \
{                                                                         \
    if (this->NAME##s[i] != v)                                            \
    {                                                                     \
        this->NAME##s[i] = v;                                             \
        this->set_modified();                                             \
    }                                                                     \
}                                                                         \
                                                                          \
/** set the  NAME algorithm vector property */                            \
void set_##NAME##s(const std::vector<T> &v)                               \
{                                                                         \
    if (this->NAME##s != v)                                               \
    {                                                                     \
        this->NAME##s = v;                                                \
        this->set_modified();                                             \
    }                                                                     \
}                                                                         \
                                                                          \
/** set the  NAME algorithm vector property */                            \
void set_##NAME##s(const std::initializer_list<T> &&l)                    \
{                                                                         \
    std::vector<T> v(l);                                                  \
    if (this->NAME##s != v)                                               \
    {                                                                     \
        this->NAME##s = v;                                                \
        this->set_modified();                                             \
    }                                                                     \
}                                                                         \
                                                                          \
/** get the i-th element of the NAME algorithm vector property */         \
const T &get_##NAME(size_t i) const                                       \
{                                                                         \
    return this->NAME##s[i];                                              \
}                                                                         \
                                                                          \
/** get the  NAME algorithm vector property */                            \
const std::vector<T> &get_##NAME##s() const                               \
{                                                                         \
    return this->NAME##s;                                                 \
}                                                                         \
                                                                          \
/** clear the  NAME algorithm vector property */                          \
void clear_##NAME##s()                                                    \
{                                                                         \
    this->NAME##s.clear();                                                \
}

/// helper that allows us to use std::function as a TECA_ALGORITHM_PROPERTY
template<typename T>
bool operator!=(const std::function<T> &lhs, const std::function<T> &rhs)
{
    return &rhs != &lhs;
}

/** This is a work around for older versions of Apple clang
 * Apple LLVM version 4.2 (clang-425.0.28) (based on LLVM 3.2svn)
 * Target: x86_64-apple-darwin12.6.0
 */
#define TECA_ALGORITHM_CALLBACK_PROPERTY(T, NAME)   \
                                                    \
/** Set the NAME algorithm property */              \
void set_##NAME(const T &v)                         \
{                                                   \
    /*if (this->NAME != v)*/                        \
    /*{*/                                           \
        this->NAME = v;                             \
        this->set_modified();                       \
    /*}*/                                           \
}                                                   \
                                                    \
/** Get the NAME algorithm property */              \
const T &get_##NAME() const                         \
{                                                   \
    return this->NAME;                              \
}                                                   \
                                                    \
/** Get the NAME algorithm property */              \
T &get_##NAME()                                     \
{                                                   \
    return this->NAME;                              \
}


/// The interface to TECA pipeline architecture.
/**
 * All sources/readers filters, sinks/writers will implement this interface.
 */
class TECA_EXPORT teca_algorithm : public std::enable_shared_from_this<teca_algorithm>
{
public:
    // construct/destruct
    virtual ~teca_algorithm() noexcept;

    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_algorithm)

    /// return the name of the class.
    virtual const char *get_class_name() const = 0;

    /** set the communicator to use at this stage of the pipeline this has
     * no influence on other stages. We duplicate the passed in communicator
     * providing an isolated communication space for subsequent operations. By
     * default the communicator is initialized to MPI_COMM_WORLD, here it is not
     * duplicated. Thus to put an algorithm into a unique communication space
     * one should explicitly set a communicator. When an algorithm should not
     * use MPI, for instance when it is in a nested pipeline, one may set the
     * communicator to MPI_COMM_SELF.
     */
    void set_communicator(MPI_Comm comm);

    /// get the active communicator
    MPI_Comm get_communicator();

#if defined(TECA_HAS_BOOST)
    /** initialize the given options description with algorithm's properties
     * implementors should call the base implementation when overriding.
     * this should be called after the override adds its options.
     */
    virtual void get_properties_description(const std::string &prefix,
        options_description &opts);

    /** initialize the algorithm from the given options variable map.
     * implementors should call the base implementation when overriding.
     * this should be called before the override sets its properties.
     */
    virtual void set_properties(const std::string &prefix,
        variables_map &opts);
#endif

    /** @name verbose
     * if set to a non-zero value, rank 0 will send status information to the
     * terminal. The default setting of zero results in no output.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, verbose)
    ///@}

    /** get an output port from the algorithm. to be used during pipeline
     * building
     */
    virtual
    teca_algorithm_output_port get_output_port(unsigned int port = 0);

    /// set an input to this algorithm
    void set_input_connection(const teca_algorithm_output_port &port)
    { this->set_input_connection(0, port); }

    /// set an input to this algorithm
    virtual
    void set_input_connection(unsigned int id,
        const teca_algorithm_output_port &port);

    /// remove input connections
    virtual
    void remove_input_connection(unsigned int id);

    /// remove all input connections
    void clear_input_connections();

    /** access the cached data produced by this algorithm. when no request is
     * specified the dataset on the top(most recent) of the cache is returned.
     * When a request is specified it may optionally be filtered by the
     * implementations cache key filter.  see also get_cache_key (threadsafe)
     */
    const_p_teca_dataset get_output_data(unsigned int port = 0);

    /** remove a dataset from the top/bottom of the cache. the top of the cache
     * has the most recently created dataset.  top or bottom is selected via
     * the boolean argument.  (threadsafe)
     */
    void pop_cache(unsigned int port = 0, int top = 0);

    /// set the cache size. the default is 1. (threadsafe)
    void set_cache_size(unsigned int n);

    /// execute the pipeline from this instance up.
    virtual int update();

    /// execute the pipeline from this instance up.
    virtual int update(unsigned int port);

    /// get meta data considering this instance up.
    virtual teca_metadata update_metadata(unsigned int port = 0);

    /// set the executive
    void set_executive(p_teca_algorithm_executive exe);

    /// get the executive
    p_teca_algorithm_executive get_executive();

    /** serialize the configuration to a stream. this should store the public
     * user modifiable properties so that runtime configuration may be saved
     * and restored.
     */
    virtual void to_stream(std::ostream &s) const;

    /// deserialize from the stream.
    virtual void from_stream(std::istream &s);

protected:
    teca_algorithm();

    /** Set the number of input connections. implementations should call this
     * from their constructors to setup the internal caches and data structures
     * required for execution.
     */
    void set_number_of_input_connections(unsigned int n);

    /** Set the number of output ports. implementations should call this from
     * their constructors to setup the internal caches and data structures
     * required for execution.
     */
    void set_number_of_output_ports(unsigned int n);

    /** set the modified flag on the given output port's cache.  should be
     * called when user modifies properties on the object that require the
     * output to be regenerated.
     */
    virtual void set_modified();

    /// an overload to set_modified by port
    void set_modified(unsigned int port);

protected:
// this section contains methods that developers
// typically need to override when implementing
// teca_algorithm's such as reader, filters, and
// writers.

    /** implementations must override this method to provide information to
     * downstream consumers about what data will be produced on each output
     * port. The port to provide information about is named in the first
     * argument the second argument contains a list of the metadata describing
     * data on all of the inputs.
     */
    virtual
    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md);

    /** implementations must override this method and generate a set of
     * requests describing the data required on the inputs to produce data for
     * the named output port, given the upstream meta data and request. If no
     * data is needed on an input then the list should contain a null request.
     */
    virtual
    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request);

    /** implementations must override this method and produce the output dataset
     * for the port named in the first argument. The second argument is a list
     * of all of the input datasets. See also get_request. The third argument
     * contains a request from the consumer which can specify information such
     * as arrays, subset region, timestep etc.  The implementation is free to
     * handle the request as it sees fit.
     */
    virtual
    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request);

    /** implementations may choose to override this method to gain control of
     * keys used in the cache. By default the passed in request is used as the
     * key. This override gives implementor the chance to filter the passed in
     * request.
     */
    virtual
    teca_metadata get_cache_key(unsigned int port,
        const teca_metadata &request) const;

protected:
// this section contains methods that control the
// pipeline's behavior. these would typically only
// need to be overridden when designing a new class
// of algorithms.

    /** driver function that manage meta data reporting  phase of pipeline
     * execution.
     */
    virtual
    teca_metadata get_output_metadata(
        teca_algorithm_output_port &current);

    /* driver function that manages execution of the given request on the named
     * port
     */
    virtual
    const_p_teca_dataset request_data(
        teca_algorithm_output_port &port,
        const teca_metadata &request);

    /** driver function that clears the output data cache where modified flag
     * has been set from the current port upstream.
     */
    virtual
    int validate_cache(teca_algorithm_output_port &current);

    /** driver function that clears the modified flag on the named port and all
     * of it's upstream connections.
     */
    virtual
    void clear_modified(teca_algorithm_output_port current);

protected:
// api exposing internals for use in driver methods

    /** search the given port's cache for the dataset associated
     * with the given request. see also get_cache_key. (threadsafe)
     */
    const_p_teca_dataset get_output_data(unsigned int port,
        const teca_metadata &request);

    /** add or update the given request , dataset pair in the cache.  see also
     * get_cache_key. (threadsafe)
     */
    int cache_output_data(unsigned int port,
        const teca_metadata &request, const_p_teca_dataset &data);

    /// clear the cache on the given output port
    void clear_cache(unsigned int port);

    /// get the number of input connections
    unsigned int get_number_of_input_connections();

    /** get the output port associated with this algorithm's i'th input
     * connection.
     */
    teca_algorithm_output_port &get_input_connection(unsigned int i);

    /// clear the modified flag on the i'th output
    void clear_modified(unsigned int port);

    /// return the output port's modified flag value
    int get_modified(unsigned int port) const;

protected:
    int verbose;

private:
    teca_algorithm_internals *internals;

    friend class teca_threaded_algorithm;
    friend class teca_data_request;
};

#endif
