#ifndef teca_rename_variables_h
#define teca_rename_variables_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_rename_variables)

/// An algorithm that renames variables.
class TECA_EXPORT teca_rename_variables : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_rename_variables)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_rename_variables)
    TECA_ALGORITHM_CLASS_NAME(teca_rename_variables)
    ~teca_rename_variables();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name original_variable_names
     * Set the list of variables to rename. For each variable to rename a new
     * name must be specified at the same index in the new_variable_names
     * list. The two lists must be the same length.
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, original_variable_name)
    ///@}

    /** @name new_variable_names
     * Set the names of the renamed variables. The new names are applied to the
     * list of variables to rename in the same order and the two lists must be
     * the same length.
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, new_variable_name)
    ///@}

protected:
    teca_rename_variables();

private:
    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::vector<std::string> original_variable_names;
    std::vector<std::string> new_variable_names;
};

#endif
