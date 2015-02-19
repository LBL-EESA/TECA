#ifndef teca_temporal_reduction_h
#define teca_temporal_reduction_h

#include "teca_dataset_fwd.h"
#include "teca_temporal_reduction_fwd.h"

#include "teca_threaded_algorithm.h"
#include "teca_metadata.h"

#include <vector>

// base class for MPI+threads temporal reduction over
// time. times are partitioned across MPI ranks and
// threads.
//
// meta data keys:
//      requires:
//          time
//      consumes:
//          time
//
// TODO -- api for setting MPI communicator
class teca_temporal_reduction : public teca_threaded_algorithm
{
public:
    //TECA_THREADED_ALGORITHM_STATIC_NEW(teca_temporal_reduction)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_temporal_reduction)
    virtual ~teca_temporal_reduction(){}

protected:
    teca_temporal_reduction(){}

protected:
// overrides that derived classes need to implement.

    // override that implements the reduction. given two datasets
    // a left and right, reduce into a single dataset and return.
    virtual p_teca_dataset reduce(
        const p_teca_dataset &left,
        const p_teca_dataset &right) = 0;

    // override that allows derived classes to generate upstream
    // requests that will be applied over all time steps. derived
    // classes implement this method instead of get_upstream_request,
    // which here is already implemented to handle the application
    // of requests over all timesteps.
    virtual
    std::vector<teca_metadata> initialize_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) = 0;

    // override that allows derived classes to report what they can
    // produce. this will be called from get_output_metadata which
    // will strip out time and partition time across MPI ranks.
    virtual
    teca_metadata initialize_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) = 0;


protected:
// customized pipeline behavior and parallel code.
// most derived classes won't need to override these.

    // generates an upstream request for each timestep. will
    // call initialize_upstream_request and apply the results to
    // all time steps.
    virtual
    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    // uses MPI communication to collect remote data for
    // required for the reduction. calls "reduce" with
    // each pair of datasets until the datasets across
    // all threads and ranks are reduced into a single
    // dataset, which is returned.
    virtual
    p_teca_dataset execute(
        unsigned int port,
        const std::vector<p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    // consumes time metadata, partitions time's across
    // MPI ranks.
    virtual
    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

private:
    // drivers for reducing the local and remote datasets.
    // calls reduce override as needed.
    p_teca_dataset reduce_local(
        std::vector<p_teca_dataset> local_data);

    p_teca_dataset reduce_remote(p_teca_dataset local_data);

};

#endif
