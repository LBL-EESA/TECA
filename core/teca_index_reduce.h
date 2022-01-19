#ifndef teca_index_reduce_h
#define teca_index_reduce_h

#include "teca_config.h"
#include "teca_dataset.h"
#include "teca_metadata.h"
#include "teca_shared_object.h"
#include "teca_threaded_algorithm.h"

#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_index_reduce)

/// Base class for MPI + threads map reduce reduction over an index.
/**
 * The available indices are partitioned across MPI ranks and threads. One can
 * restrict operation to a range of time steps by setting first and last
 * indices to process.
 *
 * metadata keys:
 *
 *      requires:
 *
 *      index_initializer_key -- holds the name of the key that tells how
 *                               many indices are available. the named key
 *                               must also be present and should contain the
 *                               number of indices available
 *
 *      index_request_key -- holds the name of the key used to request
 *                           a specific index. request are generated with this
 *                           name set to a specific index to be processed some
 *                           upstream algorithm is expected to produce the
 *                           data associated with the given index
 *
 *      consumes:
 *
 *      The key named by index_request_key
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

    // set the range of time steps to process.
    // setting first_step=0 and last_step=-1 results
    // in processing all available steps. this is
    // the default.
    TECA_ALGORITHM_PROPERTY(long, start_index)
    TECA_ALGORITHM_PROPERTY(long, end_index)

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

    // override that allows derived classes to generate upstream
    // requests that will be applied over all time steps. derived
    // classes implement this method instead of get_upstream_request,
    // which here is already implemented to handle the application
    // of requests over all timesteps.
    virtual std::vector<teca_metadata> initialize_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) = 0;

    // override that allows derived classes to report what they can
    // produce. this will be called from get_output_metadata which
    // will strip out time and partition time across MPI ranks.
    virtual teca_metadata initialize_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) = 0;

protected:
// customized pipeline behavior and parallel code.
// most derived classes won't need to override these.

    using teca_threaded_algorithm::get_output_metadata;
    using teca_threaded_algorithm::execute;

    // generates an upstream request for each timestep. will
    // call initialize_upstream_request and apply the results to
    // all time steps.
    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    // uses MPI communication to collect remote data for
    // required for the reduction. calls "reduce" with
    // each pair of datasets until the datasets across
    // all threads and ranks are reduced into a single
    // dataset, which is returned.
    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request, int streaming) override;

    // consumes time metadata, partitions time's across
    // MPI ranks.
    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

private:
    // drivers for reducing the local and remote datasets.
    // calls reduce override as needed.
    const_p_teca_dataset reduce_local(
        std::vector<const_p_teca_dataset> local_data);

    const_p_teca_dataset reduce_remote(const_p_teca_dataset local_data);

private:
    long start_index;
    long end_index;
};

#endif
