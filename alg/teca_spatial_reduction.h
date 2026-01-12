#ifndef teca_spatial_reduction_h
#define teca_spatial_reduction_h

#include "teca_shared_object.h"
#include "teca_threaded_algorithm.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"

#include <string>
#include <vector>
#include <map>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_spatial_reduction)

class TECA_EXPORT teca_spatial_reduction : public teca_threaded_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_spatial_reduction)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_spatial_reduction)
    TECA_ALGORITHM_CLASS_NAME(teca_spatial_reduction)
    ~teca_spatial_reduction();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name point_arrays
     * Set the list of arrays to reduce
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, point_array)
    ///@}

    /** @name operation
     * Set the reduction operation
     * default average
     */
    ///@{
    enum {
        average ///< Set the reduction operation to average
    };

    TECA_ALGORITHM_PROPERTY(int, operation)

    int set_operation(const std::string &operation);

    std::string get_operation_name();
    ///@}

    /** @name fill_value
     * Set the _FillValue attribute for the output data
     * default -1
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, fill_value)
    ///@}

    /** @name land_weights
     * Set land weights for global average
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, land_weights)
    ///@}

    /** @name land_weights_norm
     * Set land weights norm for global average
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, land_weights_norm)
    ///@}


protected:
    teca_spatial_reduction();

private:
    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &md_in,
        const teca_metadata &req_in) override;

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &data_in,
        const teca_metadata &req_in,
        int streaming) override;

    using teca_algorithm::get_output_metadata;
    using teca_threaded_algorithm::execute;

private:
    std::vector<std::string> point_arrays;
    int operation;
    double fill_value;
    std::string land_weights;
    double land_weights_norm;

    class internals_t;
    internals_t *internals;
};

#endif
