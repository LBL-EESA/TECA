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

    /** @name component_0_variable
     * Set the arrays that contain the vector components to compute laplacian
     * from.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, component_0_variable)
    ///@}

    /** @name component_1_variable
     * Set the arrays that contain the vector components to compute laplacian
     * from.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, component_1_variable)
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

    std::string get_component_0_variable(const teca_metadata &request);
    std::string get_component_1_variable(const teca_metadata &request);
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
    std::string component_0_variable;
    std::string component_1_variable;
    std::string laplacian_variable;
};

#endif
