#ifndef teca_derived_quantity_h
#define teca_derived_quantity_h

#include "teca_programmable_algorithm.h"
#include "teca_metadata.h"
#include "teca_dataset_fwd.h"
#include "teca_shared_object.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_derived_quantity)

/// a programmable algorithm specialized for simple array based computations
/**
A programmable algorithm specialized for simple array based
computations. A user provided callable(see set execute_callback)
which operates on one or more arrays(the dependent variables) to
produce a new array (the derived quantity). The purpose of this
class is to implement the request and report phases of the pipeline
consistently for this common use case. An implementation specific
context(operation_name) differentiates between multiple instances
in the same pipeline.
*/
class teca_derived_quantity : public teca_programmable_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_derived_quantity)
    ~teca_derived_quantity();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set/get the list of arrays that are needed to produce
    // the derived quantity
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, dependent_variable)

    // set/get the array that is produced
    TECA_ALGORITHM_PROPERTY(std::string, derived_variable)

    // set/get the contextual name that differentiates this
    // instance from others in the same pipeline.
    TECA_ALGORITHM_PROPERTY(std::string, operation_name)

    // set the callable that implements to derived quantity
    // computation
    using teca_programmable_algorithm::set_execute_callback;
    using teca_programmable_algorithm::get_execute_callback;

protected:
    teca_derived_quantity();

private:
    // specialized report and request implementations that
    // process the input and output array lists in a standardized
    // manner.
    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    // hide the programmable algorithm report request callbacks
    using teca_programmable_algorithm::set_report_callback;
    using teca_programmable_algorithm::get_report_callback;

    using teca_programmable_algorithm::set_request_callback;
    using teca_programmable_algorithm::get_request_callback;

    // extracts variable from incoming request if the property
    // is not set
    std::string get_derived_variable(const teca_metadata &request);

    // extracts dependent variables from the incoming request
    // if the coresponding property is not set
    void get_dependent_variables(const teca_metadata &request,
        std::vector<std::string> &dep_vars);

private:
    std::string operation_name;   
    std::vector<std::string> dependent_variables;
    std::string derived_variable;
};

#endif
