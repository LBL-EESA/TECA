#ifndef teca_threaded_programmable_algorithm_h
#define teca_threaded_programmable_algorithm_h

#include "teca_config.h"
#include "teca_metadata.h"
#include "teca_dataset.h"
#include "teca_threaded_algorithm.h"
#include "teca_programmable_algorithm.h"

/// An threaded algorithm implemented with user provided callbacks.
/**
 * This version of the teca_programmable_algorithm is threaded. A thread pool
 * (call set_thread_pool_size to initialize) executes the upstream pipeline
 * asynchronously for each request made. Hence, this version of the
 * programmable algorithm is most useful when there are multiple requests to
 * be processed. Data from the set of requests can be processed incrementally
 * when streaming (see set_stream_size to initialize).  If one doesn't need
 * these features it is better to use the teca_programmable_algorithm instead.
 * See teca_threaded_algorithm for more details about threaded execution.
 *
 * The user can provide a callback for each of the three phases
 * of pipeline execution. The number of input and output ports
 * can also be set for filters (1 or more inputs, 1 or more outputs)
 * sources, (no  inputs, 1 or more outputs), or sinks (1 or more
 * inputs, no outputs).
 *
 * 1) report phase. the report callback returns metadata
 *     describing data that can be produced. The report callback
 *     is optional. It's only needed if the algorithm will produce
 *     new data or transform metadata.
 *
 *     the report callback must be callable with signature:
 *     teca_metadata(unsigned int)
 *
 * 2) request phase. the request callback generates a vector of
 *     requests(metadata objects) that inform the upstream of
 *     what data to generate. The request callback is optional.
 *     It's only needed if the algorithm needs data from the
 *     upstream or transform metadata.
 *
 *     the request callback must be callable with the signature:
 *     std::vector<teca_metadata>(
 *         unsigned int,
 *         const std::vector<teca_metadata> &,
 *         const teca_metadata &)
 *
 * 3) execute phase. the execute callback is used to do useful
 *     work on incoming or outgoing data. Examples include
 *     generating new datasets, processing datasets, reading
 *     and writing data to/from disk, and so on. The execute
 *     callback is  optional.
 *
 *     the execute callback must be callable with the signature:
 *     const_p_teca_dataset(
 *         unsigned int, const std::vector<const_p_teca_dataset> &,
 *         const teca_metadata &, int)
 *
 * see also:
 *
 * set_number_of_input_connections
 * set_number_of_output_ports
 * set_report_callback
 * set_request_callback
 * set_execute_callback
 */
class TECA_EXPORT teca_threaded_programmable_algorithm : public teca_threaded_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_threaded_programmable_algorithm)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_threaded_programmable_algorithm)
    ~teca_threaded_programmable_algorithm();

    // set/get the class name.
    virtual int set_name(const std::string &name);

    const char *get_class_name() const override
    { return this->class_name; }

    // set the number of input and outputs
    using teca_algorithm::set_number_of_input_connections;
    using teca_algorithm::set_number_of_output_ports;

    // set the number of threads. The default is -1.
    using teca_threaded_algorithm::set_thread_pool_size;

    // set the stream size. the default -1 disables streaming.
    // you should set this (2 is a good choice) if your algorithm
    // can stream data.
    using teca_threaded_algorithm::set_stream_size;

    // install the default implementation
    void use_default_report_action();
    void use_default_request_action();
    void use_default_execute_action();

    // set callback that responds to reporting stage
    // of pipeline execution. the report callback must
    // be callable with signature:
    //
    //   teca_metadata (unsigned int,
    //      const std::vector<teca_metadata> &)
    //
    // the default implementation forwards downstream
    TECA_ALGORITHM_CALLBACK_PROPERTY(
        report_callback_t, report_callback)

    // set the callback that responds to the requesting
    // stage of pipeline execution. the request callback
    // must be callable with the signature:
    //
    //  std::vector<teca_metadata> (
    //    unsigned int,
    //    const std::vector<teca_metadata> &,
    //    const teca_metadata &)
    //
    // the default implementation forwards upstream
    TECA_ALGORITHM_CALLBACK_PROPERTY(
        request_callback_t, request_callback)

    // set the callback that responds to the execution stage
    // of pipeline execution. the execute callback must be
    // callable with the signature:
    //
    //  const_p_teca_dataset (
    //    unsigned int, const std::vector<const_p_teca_dataset> &,
    //    const teca_metadata &, int)
    //
    // the default implementation returns a nullptr
    TECA_ALGORITHM_CALLBACK_PROPERTY(
        threaded_execute_callback_t, execute_callback)

protected:
    teca_threaded_programmable_algorithm();

private:
    using teca_algorithm::get_output_metadata;
    using teca_threaded_algorithm::execute;

    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request, int streaming) override;

protected:
    report_callback_t report_callback;
    request_callback_t request_callback;
    threaded_execute_callback_t execute_callback;
    char class_name[96];
};

#endif
