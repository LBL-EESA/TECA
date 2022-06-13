#ifndef teca_index_reduce_h
#define teca_index_reduce_h

#include "teca_config.h"
#include "teca_dataset.h"
#include "teca_metadata.h"
#include "teca_shared_object.h"
#include "teca_threaded_algorithm.h"

#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_index_reduce)

/// Base class for MPI+threads+GPUs map reduce reduction over a set of indices.
/**
 * The available indices are partitioned across MPI ranks and threads. Threads
 * are assigned to service GPUs or CPU cores. One can restrict operation to a
 * range of time steps by setting first and last indices to process.
 *
 * ### metadata keys:
 *
 *  | key                   | description |
 *  | ----                  | ----------- |
 *  | index_initializer_key | Holds the name of the key that tells how many |
 *  |                       | indices are available. The named key must also |
 *  |                       | be present and should contain the number of |
 *  |                       | indices available. |
 *  | index_request_key     | Holds the name of the key used to request a |
 *  |                       | specific index. Requests are generated with this |
 *  |                       | name set to a specific index to be processed |
 *  |                       | some upstream algorithm is expected to produce |
 *  |                       | the data associated with the given index. |
 */
class TECA_EXPORT teca_index_reduce : public teca_threaded_algorithm
{
public:
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_index_reduce)
    virtual ~teca_index_reduce(){}

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name start_index
     * set the first index to process. Indices go from 0 to n-1. The default
     * start_index is 0.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(long, start_index)
    ///@}

    /** @name end_index
     * set the last index to process. Indices go from 0 to n-1. The default
     * end_index is -1, which is used to indicate all indices should be
     * processed.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(long, end_index)
    ///@}

protected:
    teca_index_reduce();

protected:

    /** An override that implements the reduction. given two datasets a left
     * and right, reduce into a single dataset and return.
     *
     * @param[in] device_id The device that should be used for the reduction.
     *                      A value of -1 indicates the CPU should be used.
     * @param[in] left a dataset to reduce
     * @param[in] right a dataset to reduce
     *
     * @returns the reduced dataset
     */
    virtual p_teca_dataset reduce(int device_id,
        const const_p_teca_dataset &left,
        const const_p_teca_dataset &right) = 0;

    /** An override that is called when the reduction is complete.  The default
     * implementation passes data through. This might be used for instance to
     * complete an averaging operation where the ::reduce override sums the
     * data and the ::finalize override scales by 1/N, where N is the number of
     * datasets summed.
     *
     * @param[in] device_id The device that should be used for the reduction.
     *                      A value of -1 indicates the CPU should be used.
     * @param[in] ds the reduced dataset
     *
     * @returns a dataset that has been finalized.
     */
    virtual p_teca_dataset finalize(int device_id,
        const const_p_teca_dataset &ds)
    {
        (void) device_id;
        return std::const_pointer_cast<teca_dataset>(ds);
    }

    /** An override that allows derived classes to generate upstream requests
     * that will be applied over all time steps. derived classes implement this
     * method instead of ::get_upstream_request, which here is already
     * implemented to handle the application of requests over the index set.
     */
    virtual std::vector<teca_metadata> initialize_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) = 0;

    /** An override that allows derived classes to report what they can
     * produce. this will be called from ::get_output_metadata which will strip
     * out time and partition time across MPI ranks.
     */
    virtual teca_metadata initialize_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) = 0;

protected:
// customized pipeline behavior and parallel code.
// most derived classes won't need to override these.

    using teca_threaded_algorithm::get_output_metadata;
    using teca_threaded_algorithm::execute;

    /** Generates an upstream request for each index. will call
     * initialize_upstream_request and apply the results to all time steps.
     */
    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    /** Uses MPI communication to collect remote data for required for the
     * reduction. Calls ::reduce with each pair of datasets until the datasets
     * across all threads and ranks are reduced into a single dataset, which is
     * returned.
     */
    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request, int streaming) override;

    /// consumes index metadata, and partitions indices across MPI ranks.
    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

private:
    /** The driver for reducing the local datasets. Calls the ::reduce override as
     * needed.
     */
    const_p_teca_dataset reduce_local(int device_id,
        std::vector<const_p_teca_dataset> local_data);

    /** The driver for reducing the remote datasets. Calls the ::reduce override
     * as needed.
     */
    const_p_teca_dataset reduce_remote(int device_id,
        const_p_teca_dataset local_data);

private:
    long start_index;
    long end_index;
};

#endif
