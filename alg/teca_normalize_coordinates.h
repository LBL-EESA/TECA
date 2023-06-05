#ifndef teca_normalize_coordinates_h
#define teca_normalize_coordinates_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_normalize_coordinates)

/** @brief
 * An algorithm to ensure that Cartesian mesh coordinates follow conventions
 *
 * @details
 * When enabled, transformations of coordinates and data are applied such that
 * Cartesian meshes are follow the conventions:
 *
 * 1. the x-axis coordinates are in the range of 0 to 360.
 * 2. the y-axis coordinate are in ascending order.
 * 3. the z-axis units are in Pa. Conversion from hPa is supported.
 *
 * These transformations are automatically applied and can be enabled or
 * disbaled as needed. The properties enable_unit_conversions, enable_periodic_shift
 * and enable_y_axis_ascending provide a way to enable/disable the transforms.
 *
 * Subset requests are not implemented when the periodic shift is enabled. When
 * a request is made for data that crosses the periodic boundary, the request
 * is modified to request the entire x-axis.
 *
 * If data point opn the periodic boundary is duplicated, the data at 180 is
 * dropped and a warning is issued.g
 */
class TECA_EXPORT teca_normalize_coordinates : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_normalize_coordinates)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_normalize_coordinates)
    TECA_ALGORITHM_CLASS_NAME(teca_normalize_coordinates)
    ~teca_normalize_coordinates();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name enable_periodic_shift_x
     * If set, this  enables an automatic transformation of the x-axis
     * coordinates and data from [-180, 180] to [0, 360]. When enabled, the
     * transformation is applied if the lowest x coordinate is less than 0 and
     * skipped otherwise.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, enable_periodic_shift_x)
    ///@}

    /** @name enable_y_axis_ascending
     * If set, this enables an automatic transformation of the y-axis
     * coordinates and data from descending to ascending order. The
     * transformation is applied if the lowest y coordinate is greater than the
     * highest y coordinate skipped otherwise. Many TECA algorithms are written
     * to process data with y-axis coordinates in ascending order, thus the
     * transform is enabled by default. Setting this to 0 disables the
     * transform for cases where it is desirable to pass data through
     * unmodified.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, enable_y_axis_ascending)
    ///@}

    /** @name enable_unit_conversions
     * If set, this enables an automatic conversions of units of the axes.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, enable_unit_conversions)
    ///@}

protected:
    teca_normalize_coordinates();

private:
    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    int enable_periodic_shift_x;
    int enable_y_axis_ascending;
    int enable_unit_conversions;
};

#endif
