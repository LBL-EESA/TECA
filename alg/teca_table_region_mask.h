#ifndef teca_table_region_mask_h
#define teca_table_region_mask_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_table_region_mask)

/**
 an algorithm that identifies rows in the table that are
inside the list of regions provided. a new column, called
the mask column is created. It has 1's if the row is in
the set of regions, otherwise 0's. The invert property
can be used to invert the result.
*/
class teca_table_region_mask : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_table_region_mask)
    ~teca_table_region_mask();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the name of the columns to use as coordinates
    // defaults are empty which disables space based filtering
    TECA_ALGORITHM_PROPERTY(std::string, x_coordinate_column)
    TECA_ALGORITHM_PROPERTY(std::string, y_coordinate_column)

    // set the name of the column to store the mask in
    // the mask is a column of 1 and 0 indicating if the
    // row satsifies the criteria or not. the default is
    // "region_mask"
    TECA_ALGORITHM_PROPERTY(std::string, result_column);

    // the following group of properties define a set of
    // polygons describing arbitrary regions. events are removed
    // when outside of the regions. note: must explicitly include
    // end point.
    TECA_ALGORITHM_VECTOR_PROPERTY(unsigned long, region_size)
    TECA_ALGORITHM_VECTOR_PROPERTY(unsigned long, region_start)
    TECA_ALGORITHM_VECTOR_PROPERTY(double, region_x_coordinate)
    TECA_ALGORITHM_VECTOR_PROPERTY(double, region_y_coordinate)

    // clear the list of region definitions.
    void clear_regions();

    // load a predefined basin region by name. one can use
    // teca_geography::get_cyclone_basin_names to obtain
    // the list of basin names. the basin region definition
    // is appended to the current list of regions.
    int load_cyclone_basin(const std::string &name);

    // invert the test. when true the result will be true
    // if the point is outside the regions
    TECA_ALGORITHM_PROPERTY(int, invert)

protected:
    teca_table_region_mask();

private:
    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string x_coordinate_column;
    std::string y_coordinate_column;
    std::string result_column;
    std::vector<unsigned long> region_sizes;
    std::vector<unsigned long> region_starts;
    std::vector<double> region_x_coordinates;
    std::vector<double> region_y_coordinates;
    int invert;
};

#endif
