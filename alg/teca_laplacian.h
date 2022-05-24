#ifndef teca_laplacian_h
#define teca_laplacian_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_laplacian)

/// An algorithm that computes the Laplacian from a vector field.
class TECA_EXPORT teca_laplacian : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_laplacian)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_laplacian)
    TECA_ALGORITHM_CLASS_NAME(teca_laplacian)
    ~teca_laplacian();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name scalar_field_name
     * Set the arrays that contain the vector components to compute laplacian
     * from.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, scalar_field_name)
    ///@}

    /** @name laplacian_variable
     * Set the name of the array to store the result in.  the default is
     * "laplacian".
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, laplacian_variable)
    ///@}

protected:
    teca_laplacian();

    std::string get_scalar_field_name(const teca_metadata &request);
    std::string get_laplacian_variable(const teca_metadata &request);

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
    std::string scalar_field_name;
    std::string laplacian_variable;
};

#endif
