#ifndef teca_temporal_reduction_h
#define teca_temporal_reduction_h

#include "teca_dataset_fwd.h"
#include "teca_temporal_reduction_fwd.h"

#include "teca_threaded_algorithm.h"
#include "teca_metadata.h"

#include <vector>

// base class for MPI+threads temporal reduction over
// time. the available time steps  are partitioned
// across MPI ranks and threads. one can restrict
// operation to a range of time steps by setting
// first and last steps to process.
//
// meta data keys:
//      requires:
//          number_of_time_steps - the number of time steps available
//
//      consumes:
//          time_step
class teca_temporal_reduction : public teca_threaded_algorithm
{
public:
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_temporal_reduction)
    virtual ~teca_temporal_reduction(){}

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the range of time steps to process.
    // setting first_step=0 and last_step=-1 results
    // in processing all available steps. this is
    // the default.
    TECA_ALGORITHM_PROPERTY(long, first_step)
    TECA_ALGORITHM_PROPERTY(long, last_step)

protected:
    teca_temporal_reduction();

protected:
    // override that implements the reduction. given two datasets
    // a left and right, reduce into a single dataset and return.
    virtual p_teca_dataset reduce(const const_p_teca_dataset &left,
        const const_p_teca_dataset &right) = 0;

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
        const teca_metadata &request) override;

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
    long first_step;
    long last_step;
};

#endif
