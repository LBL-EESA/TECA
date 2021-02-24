#ifndef teca_normalize_coordinates_h
#define teca_normalize_coordinates_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_normalize_coordinates)

/// an algorithm to ensure that coordinates are in ascending order
/**
Transformations of coordinates and data to/from ascending order
are made as data and information pass up and down stream through
the algorithm.
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

    /** @anchor x,y,z_axis_order
     * @name x,y,z_axis_order
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

    struct internals_t;
    internals_t *internals;
};

#endif
