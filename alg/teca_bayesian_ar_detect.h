#ifndef teca_bayesian_ar_detect_h
#define teca_bayesian_ar_detect_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_bayesian_ar_detect)

/// The TECA BARD atmospheric river detector.
/**
 * Given a point wise IVT (integrated vapor transport) field and a training
 * parameter table computes the point wise probability of an atmospheric river
 * using the TECA BARD algorithm.
 *
 * Required inputs:
 *
 *     1. IVT (integrated vapor transport) array on a Cartesian nesh.
 *     2. a compatible parameter table. columns of which are : min IVT,
 *        component area, HWHM lattitude
 *
 * The names of the input varibale and columns can be specified at run time
 * through algorithm properties.
 *
 * Produces:
 *
 *     A Cartesian mesh with probability of an AR stored in the point centered
 *     array named "ar_probability". The diagnostic quantites "ar_count" amd
 *     "parameter_table_row" are stored in information arrays.
 *
 * For more information see:
 *
 * O’Brien, T. A., Risser, M. D., Loring, B., Elbashandy, A. A., Krishnan, H.,
 * Johnson, J., Patricola, C. M., O’Brien, J. P., Mahesh, A., Arriaga Ramirez,
 * S., Rhoades, A. M., Charn, A., Inda Díaz, H., & Collins, W. D. (2020).
 * Detection of atmospheric rivers with inline uncertainty quantification:
 * TECA-BARD v1.0.1. Geoscientific Model Development, 13(12), 6131–6148.
 * https://doi.org/10.5194/gmd-13-6131-2020
*/
class TECA_EXPORT teca_bayesian_ar_detect : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_bayesian_ar_detect)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_bayesian_ar_detect)
    TECA_ALGORITHM_CLASS_NAME(teca_bayesian_ar_detect)
    ~teca_bayesian_ar_detect();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name ivt_variable
     * Sets the name of the array containing the IVT field to detect ARs in.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, ivt_variable)
    ///@}

    /** @name min_ivt_variable
     * Set the names of the minimum IVT column in the parameter table.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, min_ivt_variable)
    ///@}

    /** @name min_component_area_variable
     * Set the names of the minimum area column in the parameter table.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, min_component_area_variable)
    ///@}

    /** @name hwhm_latitude_variable
     * Set the names of the HWHM column in the parameter table.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, hwhm_latitude_variable)
    ///@}

    /** @name probability variable
     * Set the name of the variable to store output probability as.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, ar_probability_variable)
    ///@}

    /** Set the number of threads in the pool. Setting to -1 results in a
     * thread per core factoring in all MPI ranks running on the node. the
     * default is -1.
     * @note this constructs the thread pool. properties that affect the thread
     * pool must be set before construction.
     */
    void set_thread_pool_size(int n_threads);

    /// Get the number of threads in the pool.
    unsigned int get_thread_pool_size() const noexcept;

    /** @name bind_threads
     * set/get thread affinity mode. When 0 threads are not bound CPU cores,
     * allowing for migration among all cores. This will likely degrade
     * performance. Default is 1.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, bind_threads)
    ///@}

    /** @name stream_size
     * set the smallest number of datasets to gather per call to execute. the
     * default (-1) results in all datasets being gathered. In practice more
     * datasets will be returned if ready
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, stream_size)
    ///@}

    /** @name poll_interval
     * set the duration in nanoseconds to wait between checking for completed
     * tasks
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(long long, poll_interval)
    ///@}

    /** @name threads_per_device
     * Set the number of threads to service each GPU/device. Other threads will
     * use the CPU.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, threads_per_device)
    ///@}

    /** @name ranks_per_device
     * Set the number of ranks that have access to each GPU/device. Other ranks
     * will use the CPU.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, ranks_per_device)
    ///@}

    /** @name propagate_device_assignment
     * When set device assignment is taken from down stream request.
     * Otherwise the thread executing the pipeline will provide the assignment.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, propagate_device_assignment)
    ///@}

    /** override the input connections because we are going to take the first
     * input and use it to generate metadata.  the second input then becomes
     * the only one the pipeline knows about.
     */
    void set_input_connection(unsigned int id,
        const teca_algorithm_output_port &port) override;

protected:
    teca_bayesian_ar_detect();

    std::string get_label_variable(const teca_metadata &request);

private:
    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    void set_modified() override;

private:
    std::string ivt_variable;
    std::string min_component_area_variable;
    std::string min_ivt_variable;
    std::string hwhm_latitude_variable;
    std::string ar_probability_variable;
    int thread_pool_size;
    int bind_threads;
    int stream_size;
    long long poll_interval;
    int threads_per_device;
    int ranks_per_device;
    int propagate_device_assignment;

    struct internals_t;
    internals_t *internals;
};

#endif
