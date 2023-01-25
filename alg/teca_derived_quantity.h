#ifndef teca_derived_quantity_h
#define teca_derived_quantity_h

#include "teca_programmable_algorithm.h"
#include "teca_metadata.h"
#include "teca_dataset.h"
#include "teca_shared_object.h"
#include "teca_array_attributes.h"

#include <string>
#include <vector>

#if defined(__CUDACC__)
#pragma nv_diag_suppress = partial_override
#endif

TECA_SHARED_OBJECT_FORWARD_DECL(teca_derived_quantity)

/// a programmable algorithm specialized for simple array based computations
/**
 * A programmable algorithm specialized for simple array based computations. A
 * user provided callable(see set execute_callback) which operates on one or more
 * arrays(the dependent variables) to produce a new array (the derived quantity).
 * The purpose of this class is to implement the request and report phases of the
 * pipeline consistently for this common use case. An implementation specific
 * context(operation_name) differentiates between multiple instances in the same
 * pipeline.
 */
class TECA_EXPORT teca_derived_quantity : public teca_programmable_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_derived_quantity)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_derived_quantity)
    ~teca_derived_quantity();

    //using teca_programmable_algorithm::get_class_name;

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /// @name dependent_variables
    /** Set/get the list of arrays that are needed to produce the derived
     * quantity. This should include valid value masks if needed.  See
     * teca_valid_value_mask for more information.
     */
    /// @{
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, dependent_variable)
    /// @}

    /// @name derived_variable
    /** Set/get the names of the arrays and their attributes that are produced.
     */
    /// @{
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, derived_variable)
    TECA_ALGORITHM_VECTOR_PROPERTY(teca_array_attributes, derived_variable_attribute)

    /** Add the names and attributes of the derived variables. Array attributes
     * must be provided for parallel I/O using the teca_cf_writer.
     */
    void append_derived_variable(const std::string &name,
        const teca_array_attributes &atts)
    {
        this->append_derived_variable(name);
        this->append_derived_variable_attribute(atts);
    }

    /** Set the name and attributes of the derived variable. Array attributes
     * must be provided for parallel I/O using the teca_cf_writer.
     */
    void set_derived_variables(const std::vector<std::string> &names,
        const std::vector<teca_array_attributes> &atts)
    {
        this->set_derived_variables(names);
        this->set_derived_variable_attributes(atts);
    }
    /// @}

    /// @name operation name
    /** Set the contextual name that differentiates this instance from others
     * in the same pipeline.
     */
    /// @{
    /// set the operation name
    void set_operation_name(const std::string &op_name);

    /// get the operation name
    const std::string &get_operation_name() const
    { return this->operation_name; }

    /// get the operation name
    std::string &get_operation_name()
    { return this->operation_name; }
    /// @}

    // set the callable that implements to derived quantity
    // computation
    using teca_programmable_algorithm::set_execute_callback;
    using teca_programmable_algorithm::get_execute_callback;

protected:
    teca_derived_quantity();

private:
    //using teca_algorithm::get_output_metadata;

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

    // set the name used in log files.
    int set_name(const std::string &name) override;

private:
    std::string operation_name;
    std::vector<std::string> dependent_variables;
    std::vector<std::string> derived_variables;
    std::vector<teca_array_attributes> derived_variable_attributes;
};

#if defined(__CUDACC__)
#pragma nv_diag_default = partial_override
#endif
#endif
