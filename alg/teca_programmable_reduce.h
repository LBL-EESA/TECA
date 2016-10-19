#ifndef teca_programmable_reduce_h
#define teca_programmable_reduce_h

#include "teca_programmable_reduce_fwd.h"
#include "teca_programmable_algorithm_fwd.h"
#include "teca_temporal_reduction.h"
#include "teca_dataset_fwd.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

// callbacks implement a user defined reduction over time steps
/**
callbacks implement a reduction on teca_datasets over time steps.
user provides reduce callable that takes 2 datasets and produces
a thrid reduced dataset. callbacks should be threadsafe as this is
a parallel operation. see teca_temporal_reduction for details of
parallelization.
*/
class teca_programmable_reduce : public teca_temporal_reduction
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_programmable_reduce)
    ~teca_programmable_reduce(){}

    // set the callback that initializes the output metadata during
    // report phase of the pipeline. The callback must be a callable
    // with the signature:
    //
    // teca_metadata report_callback(unsigned int port,
    //    const std::vector<teca_metadata> &input_md);
    //
    // the default implementation forwards downstream
    TECA_ALGORITHM_CALLBACK_PROPERTY(report_callback_t, report_callback)

    // set the callback that initializes the upstream request.
    // The callback must be a callable with the signature:
    //
    // std::vector<teca_metadata> request(
    //    unsigned int port, const std::vector<teca_metadata> &input_md,
    //    const teca_metadata &request) override;
    //
    // the default implementation forwards upstream
    TECA_ALGORITHM_CALLBACK_PROPERTY(request_callback_t, request_callback)

    // set the callback that performs the reduction on 2 datasets
    // returning the reduced dataset. The callback must be a callable
    // with the signature:
    //
    // p_teca_dataset reduce(const const_p_teca_dataset &left,
    //    const const_p_teca_dataset &right);
    //
    // the default implementation returns a nullptr
    TECA_ALGORITHM_CALLBACK_PROPERTY(reduce_callback_t, reduce_callback)

protected:
    teca_programmable_reduce();

    // overrides
    p_teca_dataset reduce(const const_p_teca_dataset &left,
        const const_p_teca_dataset &right) override;

    std::vector<teca_metadata> initialize_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    teca_metadata initialize_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

private:
    reduce_callback_t reduce_callback;
    request_callback_t request_callback;
    report_callback_t report_callback;
};

#endif
