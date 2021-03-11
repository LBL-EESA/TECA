#ifndef teca_normalize_coordinates_h
#define teca_normalize_coordinates_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_normalize_coordinates)

/// an algorithm that ensures that Cartesian mesh coordinates follow conventions
/**
 * Transformations of coordinates and data to/from ascending order
 * are made as data and information pass up and down stream through
 * the algorithm. See @ref axis_order
 *
 * An optional translation to each axis can be applied by setting
 * one or more of translate_x, translate_y, or translate_z to a
 * non-zero value. See @ref translate_axis
 *
 * Use this algorithm when downstream processing depends on coordinate
 * conventions. For instance differentials or integrals may require spatial
 * coordinate be in ascending or descending order. Similarly regriding
 * operations may require data in the same coordinate system.
*/
class teca_normalize_coordinates : public teca_algorithm
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

    /** @anchor axis_order
     * @name axis_order
     * Set the desired order of the output for each coordinate
     * axis. Use ORDER_ASCENDING(0) to ensure the output is in
     * ascending order, and ORDER_DESCENDING(1) to ensure the
     * output is in descending order. By default the x and y
     * axes are put in ascending order and the z axis is put
     * into descending order.
     */
    ///@{
    enum {ORDER_ASCENDING = 0, ORDER_DESCENDING = 1};
    TECA_ALGORITHM_PROPERTY_V(int, x_axis_order)
    TECA_ALGORITHM_PROPERTY_V(int, y_axis_order)
    TECA_ALGORITHM_PROPERTY_V(int, z_axis_order)
    ///@}

    /** @anchor translate_axis
     * @name translate_axis
     * Set the amount to translate the x, y, or z axis by.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, translate_x)
    TECA_ALGORITHM_PROPERTY(double, translate_y)
    TECA_ALGORITHM_PROPERTY(double, translate_z)
    ///@}

protected:
    teca_normalize_coordinates();

    int validate_x_axis_order(int val);
    int validate_y_axis_order(int val);
    int validate_z_axis_order(int val);

private:
    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    int x_axis_order;
    int y_axis_order;
    int z_axis_order;
    double translate_x;
    double translate_y;
    double translate_z;

    struct internals_t;
    internals_t *internals;
};

#endif
