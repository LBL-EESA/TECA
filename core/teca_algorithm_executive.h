#ifndef teca_algorithm_executive_h
#define teca_algorithm_executive_h

#include "teca_config.h"
#include "teca_metadata.h"
#include "teca_mpi.h"
#include "teca_shared_object.h"

TECA_SHARED_OBJECT_FORWARD_DECL(teca_algorithm_executive)

/* this is a convenience macro to be used to declare a static
 * New method that will be used to construct new objects in
 * shared_ptr's. This manages the details of interoperability
 * with std C++11 shared pointer
 */
#define TECA_ALGORITHM_EXECUTIVE_STATIC_NEW(T)                      \
                                                                    \
                                                                    \
/** Allocate a new T */                                             \
static p_##T New()                                                  \
{                                                                   \
    return p_##T(new T);                                            \
}                                                                   \
                                                                    \
std::shared_ptr<T> shared_from_this()                               \
{                                                                   \
    return std::static_pointer_cast<T>(                             \
        teca_algorithm_executive::shared_from_this());              \
}                                                                   \
                                                                    \
std::shared_ptr<T const> shared_from_this() const                   \
{                                                                   \
    return std::static_pointer_cast<T const>(                       \
        teca_algorithm_executive::shared_from_this());              \
}



/// Base class and default implementation for executives.
/**
 * Algorithm executives can control pipeline execution by providing a series of
 * requests. this allows for the executive to act as a load balancer. the
 * executive can for example partition requests across spatial data, time
 * steps, or file names. in an MPI parallel setting the executive could
 * coordinate this partitioning amongst the ranks. However, the only
 * requirement of an algorithm executive is that it provide at least one
 * non-empty request.
 *
 * The default implementation creates a single request for the first index
 * specified by the "index_initializer_key" using the "index_request_key" with
 * "device_id" set to execute on the CPU or GPU (if GPU's are available).
 */
class TECA_EXPORT teca_algorithm_executive
    : public std::enable_shared_from_this<teca_algorithm_executive>
{
public:
    static p_teca_algorithm_executive New()
    { return p_teca_algorithm_executive(new teca_algorithm_executive); }

    virtual ~teca_algorithm_executive() {}

    // initialize requests from the given metadata object.
    // this is a place where work partitioning across MPI
    // ranks can occur
    virtual int initialize(MPI_Comm comm, const teca_metadata &md);

    // get the next request until all requests have been
    // processed. an empty request is returned.
    virtual teca_metadata get_next_request();

    // set/get verbosity level
    void set_verbose(int a_verbose) { this->verbose = a_verbose; }
    int get_verbose() const { return this->verbose; }

protected:
    teca_algorithm_executive() : verbose(0) {}
    teca_algorithm_executive(const teca_algorithm_executive &) = default;
    teca_algorithm_executive(teca_algorithm_executive &&) = default;
    teca_algorithm_executive &operator=(const teca_algorithm_executive &) = default;
    teca_algorithm_executive &operator=(teca_algorithm_executive &&) = default;

private:
    int verbose;
    teca_metadata m_request;
};

#endif
