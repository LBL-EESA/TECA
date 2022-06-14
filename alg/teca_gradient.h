#ifndef teca_gradient_h
#define teca_gradient_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_gradient)

/// An algorithm that computes gradient from a vector field.
class TECA_EXPORT teca_gradient : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_gradient)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_gradient)
    TECA_ALGORITHM_CLASS_NAME(teca_gradient)
    ~teca_gradient();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name scalar field
     * set the array that contains the scalar field to compute gradient
     * from.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, scalar_field)
    ///@}

    /** @name gradient_field_x
     * set the name of the array to store the x-component of the 
     * gradient in.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, gradient_field_x)
    ///@}

    /** @name gradient_field_y
     * set the name of the array to store the y-component of the 
     * gradient in.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, gradient_field_y)
    ///@}


protected:
    teca_gradient();

    std::string get_scalar_field(const teca_metadata &request);
    std::string get_gradient_field_x(const teca_metadata &request);
    std::string get_gradient_field_y(const teca_metadata &request);

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
    std::string scalar_field;
    std::string gradient_field_x;
    std::string gradient_field_y;
};

#endif
