#ifndef teca_tc_classify_h
#define teca_tc_classify_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_tc_classify)

/// an algorithm that classifies storms using Saphire-Simpson scale
/**
An algorithm that classifies storms using Saphire-Simpson scale
a column containing the classification is added to the output

An algorithm that sorts the storms by geographic region
and category. tyhe output is a table where regions is
mapped to columns and category is mapped to rows.
*/
class teca_tc_classify : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_tc_classify)
    ~teca_tc_classify();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the name pf the column containing track id.
    // default is "track_id"
    TECA_ALGORITHM_PROPERTY(std::string, track_id_column)

    // set the name of the column containing the time
    // axis. default is "time"
    TECA_ALGORITHM_PROPERTY(std::string, time_column)

    // set the name of the column containing surface wind speed
    // for Saphire-Simpson scale. default is "surface_wind"
    TECA_ALGORITHM_PROPERTY(std::string, surface_wind_column)

    // set the name of the column containing sea level pressure
    // default is "sea_level_pressure"
    TECA_ALGORITHM_PROPERTY(std::string, sea_level_pressure_column)

    // set the name of the columns to use as coordinates
    // defaults are "lon" and "lat"
    TECA_ALGORITHM_PROPERTY(std::string, x_coordinate_column)
    TECA_ALGORITHM_PROPERTY(std::string, y_coordinate_column)

    // the following group of properties define a set of
    // polygons describing arbitrary regions to count
    // storms by. note: must explicitly include end point.
    TECA_ALGORITHM_VECTOR_PROPERTY(unsigned long, region_size)
    TECA_ALGORITHM_VECTOR_PROPERTY(unsigned long, region_start);
    TECA_ALGORITHM_VECTOR_PROPERTY(double, region_x_coordinate);
    TECA_ALGORITHM_VECTOR_PROPERTY(double, region_y_coordinate);
    TECA_ALGORITHM_VECTOR_PROPERTY(int, region_id);
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, region_name);
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, region_long_name);

protected:
    teca_tc_classify();

private:
    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string track_id_column;
    std::string time_column;
    std::string x_coordinate_column;
    std::string y_coordinate_column;
    std::string surface_wind_column;
    std::string sea_level_pressure_column;

    std::vector<unsigned long> region_sizes;
    std::vector<unsigned long> region_starts;
    std::vector<double> region_x_coordinates;
    std::vector<double> region_y_coordinates;
    std::vector<int> region_ids;
    std::vector<std::string> region_names;
    std::vector<std::string> region_long_names;
};

#endif
