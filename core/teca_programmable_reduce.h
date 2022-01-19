#ifndef teca_programmable_reduce_h
#define teca_programmable_reduce_h

#include "teca_config.h"
#include "teca_dataset.h"
#include "teca_metadata.h"
#include "teca_shared_object.h"
#include "teca_programmable_algorithm.h"
#include "teca_index_reduce.h"

#include <string>
#include <vector>
#include <functional>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_programmable_reduce)

#ifdef SWIG
typedef void* reduce_callback_t;
typedef void* finalize_callback_t;
#else
/// A callable that can reduce two datasets into one.
using reduce_callback_t = std::function<p_teca_dataset(
    int, const const_p_teca_dataset &, const const_p_teca_dataset &)>;

/// A callable that can finalize the reduction.
using finalize_callback_t = std::function<p_teca_dataset(
    int, const const_p_teca_dataset &)>;
#endif

/// Callbacks implement a user defined reduction over time steps.
/**
 * Callbacks implement a reduction on teca_datasets over time steps.
 * User provides reduce callable that takes 2 datasets and produces
 * a third reduced dataset. Callbacks should be threadsafe as this is
 * a parallel operation. See teca_index_reduce for details of
 * parallelization.
 */
class TECA_EXPORT teca_programmable_reduce : public teca_index_reduce
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_programmable_reduce)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_programmable_reduce)
    ~teca_programmable_reduce(){}

    // set the implementation name, this is used in logging to
    // identify the specific instance of programmable reduce
    int set_name(const std::string &name);

    const char *get_class_name() const override
    { return this->class_name; }

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

    // set the callback that finalizes the reduction.
    // The callback must be a callable with the signature:
    //
    // p_teca_dataset reduce(const const_p_teca_dataset &ds);
    //
    // the default implementation passes the input dataset
    // through
    TECA_ALGORITHM_CALLBACK_PROPERTY(finalize_callback_t, finalize_callback)

protected:
    teca_programmable_reduce();

    // overrides
    p_teca_dataset reduce(int device_id, const const_p_teca_dataset &left,
        const const_p_teca_dataset &right) override;

    p_teca_dataset finalize(int device_id,
        const const_p_teca_dataset &input) override;

    std::vector<teca_metadata> initialize_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    teca_metadata initialize_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

private:
    reduce_callback_t reduce_callback;
    finalize_callback_t finalize_callback;
    request_callback_t request_callback;
    report_callback_t report_callback;
    char class_name[64];
};

#endif
