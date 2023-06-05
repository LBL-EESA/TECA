#ifndef teca_programmable_algorithm_h
#define teca_programmable_algorithm_h

#include "teca_config.h"
#include "teca_algorithm.h"
#include "teca_shared_object.h"
#include "teca_metadata.h"
#include "teca_dataset.h"

TECA_SHARED_OBJECT_FORWARD_DECL(teca_programmable_algorithm)
TECA_SHARED_OBJECT_FORWARD_DECL(teca_threaded_programmable_algorithm)

#ifdef SWIG
typedef void* report_callback_t;
typedef void* request_callback_t;
typedef void* execute_callback_t;
typedef void* threaded_execute_callback_t;
#else
/// A callable implementing the report phase of pipeline execution
using report_callback_t = std::function<teca_metadata(
        unsigned int, const std::vector<teca_metadata>&)>;

/// A callable implementing the request phase of pipeline execution
using request_callback_t = std::function<std::vector<teca_metadata>(
        unsigned int, const std::vector<teca_metadata> &,
        const teca_metadata &)>;

/// A callable implementing the execute phase of pipeline execution
using execute_callback_t = std::function<const_p_teca_dataset(
        unsigned int, const std::vector<const_p_teca_dataset> &,
        const teca_metadata &)>;

/// A callable implementing the streaming execute phase of pipeline execution
using threaded_execute_callback_t = std::function<const_p_teca_dataset(
        unsigned int, const std::vector<const_p_teca_dataset> &,
        const teca_metadata &, int)>;
#endif

/// An algorithm implemented with  user provided callbacks.
/**
 * The user can provide a callback for each of the three phases
 * of pipeline execution. The number of input and output ports
 * can also be set for filters (1 or more inputs, 1 or more outputs)
 * sources, (no  inputs, 1 or more outputs), or sinks (1 or more
 * inputs, no outputs).
 *
 * 1. report phase. the report callback returns metadata
 *    describing data that can be produced. The report callback
 *    is optional. It's only needed if the algorithm will produce
 *    new data or transform metadata.
 *
 *    the report callback must be callable with signature:
 *    teca_metadata(unsigned int)
 *
 * 2. request phase. the request callback generates a vector
 *    of requests(metadata objects) that inform the upstream of
 *    what data to generate. The request callback is optional.
 *    It's only needed if the algorithm needs data from the
 *    upstream or transform metadata.
 *
 *    the request callback must be callable with the signature:
 *    std::vector<teca_metadata>(
 *        unsigned int,
 *        const std::vector<teca_metadata> &,
 *        const teca_metadata &)
 *
 * 3. execute phase. the execute callback is used to do useful
 *    work on incoming or outgoing data. Examples include
 *    generating new datasets, processing datasets, reading
 *    and writing data to/from disk, and so on. The execute
 *    callback is  optional.
 *
 *    the execute callback must be callable with the signature:
 *    const_p_teca_dataset(
 *        unsigned int, const std::vector<const_p_teca_dataset> &,
 *        const teca_metadata &)
 *
 * see also:
 *
 * set_number_of_input_connections
 * set_number_of_output_ports
 * set_report_callback
 * set_request_callback
 * set_execute_callback
 */
class TECA_EXPORT teca_programmable_algorithm : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_programmable_algorithm)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_programmable_algorithm)
    ~teca_programmable_algorithm();

    /// @name
    /** set/get the class name.
     */
    /// @{
    /// set the name differntiating this instance in the pipeline.
    virtual int set_name(const std::string &name);

    /// get the name differntiating this instance in the pipeline.
    const char *get_class_name() const override
    { return this->class_name; }
    /// @}

    /// set the number of input and outputs
    using teca_algorithm::set_number_of_input_connections;
    using teca_algorithm::set_number_of_output_ports;

    /// @name default actions
    /** Default implemenations for the phases of pipeline execution. Note that
     * these generally do not do all that is needed but may suffice for simple
     * pipelines.
     */
    /// @{
    /** Install the default implementation for the report phase which forwards
     * the incoming report downstream.
     */
    void use_default_report_action();

    /** Install the default implementation for the request phase which forwards
     * the incoming request upstream.
     */
    void use_default_request_action();

    /** Install the default implementation for the execute phase which
     * returns a nullptr.
     */
    void use_default_execute_action();
    /// @}

    /// @name report_callback
    /// @{
    /** Set callback that responds to reporting stage of pipeline execution.
     * the report callback must be callable with signature:
     *
     * ```C++
     * teca_metadata (unsigned int, const std::vector<teca_metadata> &)
     * ```
     */
    TECA_ALGORITHM_CALLBACK_PROPERTY(report_callback_t, report_callback)
    /// @}

    /// @name request_callback
    /// @{
    /** Set the callback that responds to the requesting stage of pipeline
     * execution. the request callback must be callable with the signature:
     *
     * ```C++
     * std::vector<teca_metadata> (unsigned int,
     *    const std::vector<teca_metadata> &, const teca_metadata &)
     * ```
     */
    TECA_ALGORITHM_CALLBACK_PROPERTY(request_callback_t, request_callback)
    /// @}

    /// @name execute_callback
    /// @{
    /** Set the callback that responds to the execution stage of pipeline
     * execution. the execute callback must be callable with the signature:
     *
     * ```C++
     * const_p_teca_dataset (
     *     unsigned int, const std::vector<const_p_teca_dataset> &,
     *     const teca_metadata &)
     * ```
     */
    TECA_ALGORITHM_CALLBACK_PROPERTY(execute_callback_t, execute_callback)

protected:
    teca_programmable_algorithm();

    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    report_callback_t report_callback;
    request_callback_t request_callback;
    execute_callback_t execute_callback;
    char class_name[128];
};

#endif
