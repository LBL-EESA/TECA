#ifndef teca_unpack_data_h
#define teca_unpack_data_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_variant_array.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_unpack_data)

/// an algorithm that unpacks NetCDF packed values
/**
 * Applies a data transform according to the NetCDF attribute conventions for
 * packed data values.
 * https://www.unidata.ucar.edu/software/netcdf/docs/attribute_conventions.html
 *
 * Variables in the input dataset are scanned for the presence
 * of the `scale_factor` and `add_offset` attributes. When both are present
 * an element wise transformation is applied such that
 *
 * out[i] = scale_factor * in[i] + add_offset
 *
 * The input array is expected to be an integer type while the type of the output
 * array may be either float or double. Valid value masks may be necessary for
 * correct results, see `teca_valid_value_mask`.
*/
class TECA_EXPORT teca_unpack_data : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_unpack_data)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_unpack_data)
    TECA_ALGORITHM_CLASS_NAME(teca_unpack_data)
    ~teca_unpack_data();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name output_data_type
     * set the output data type.  use teca_variant_array_code<T>::get() to get
     * the numeric code corresponding to the data type T. The default output
     * data type is single precision floating point.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY_V(int, output_data_type)

    /// set the output data type to double precision floating point
    void set_output_data_type_to_float()
    { this->set_output_data_type(teca_variant_array_code<float>::get()); }

    /// set the output data type to single precision floating point
    void set_output_data_type_to_double()
    { this->set_output_data_type(teca_variant_array_code<double>::get()); }
    ///@}

protected:
    teca_unpack_data();

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

    int validate_output_data_type(int val);

private:
    int output_data_type;
};

#endif
