#ifndef teca_programmable_source_h
#define teca_programmable_source_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_dataset_fwd.h"
#include "teca_metadata.h"

TECA_SHARED_OBJECT_FORWARD_DECL(teca_programmable_source)

// helper that allows us to use std::function
// as a TECA_ALGORITHM_PROPERTY
template<typename T>
bool operator!=(const std::function<T> &lhs, const std::function<T> &rhs)
{
    return &rhs != &lhs;
}

/// serves up a user provided dataset
/**
An algorithm that serves up a user provided
dataset to the pipeline. metadata for the
reporting phase must also be provided. this
provides a way to inject arbitrary data into
a pipeline without i/o.
*/
class teca_programmable_source : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_programmable_source)
    ~teca_programmable_source();

    // set function that responds to reporting stage
    // of pipeline execution. the function must return
    // a metedata object describing the data that could
    // be produced during the execution stage.
    TECA_ALGORITHM_PROPERTY(
        std::function<teca_metadata()>,
        report_function)

    // set the function that responds to the execution stage
    // of pipeline execution. the function must return
    // a dataset containing the requested data.
    TECA_ALGORITHM_PROPERTY(
        std::function<const_p_teca_dataset(const teca_metadata &)>,
        execute_function)

protected:
    teca_programmable_source();

private:
    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::function<teca_metadata()> report_function;
    std::function<const_p_teca_dataset(const teca_metadata &)> execute_function;
};

#endif
