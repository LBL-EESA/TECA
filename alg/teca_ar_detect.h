#ifndef teca_ar_detect_h
#define teca_ar_detect_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_ar_detect)

/**
Suren and Junmin's atmospheric river detector.

The algorithm searches for atmospheric rivers that
end on the California coast in water vapor data over
a specific subset of the input data. A river is detected
based on it's length, width, and percent area of the
search space. The algorithm can optionally use a
land-sea mask to increase accuracy of the California
coast. Without the land-sea mask a box is used.
*/
class teca_ar_detect : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_ar_detect)
    ~teca_ar_detect();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set/get the name of the integrated water vapor variable
    TECA_ALGORITHM_PROPERTY(std::string, water_vapor_variable)

    // set/get threshold on water vapor variable used
    // to segment the data
    TECA_ALGORITHM_PROPERTY(double, low_water_vapor_threshold)
    TECA_ALGORITHM_PROPERTY(double, high_water_vapor_threshold)

    // set/get the region of interest in lat lon coordinate system
    // defaults are 19 56 180 250
    TECA_ALGORITHM_PROPERTY(double, search_lat_low)
    TECA_ALGORITHM_PROPERTY(double, search_lon_low)
    TECA_ALGORITHM_PROPERTY(double, search_lat_high)
    TECA_ALGORITHM_PROPERTY(double, search_lon_high)

    // set/get the river source region in lat lon coordinate system
    // defaults are 18 180
    TECA_ALGORITHM_PROPERTY(double, river_start_lat_low)
    TECA_ALGORITHM_PROPERTY(double, river_start_lon_low)

    // set/get the river ladfall region in lat lon coordinate system
    // defaults are  29 233 56 238
    TECA_ALGORITHM_PROPERTY(double, river_end_lat_low)
    TECA_ALGORITHM_PROPERTY(double, river_end_lon_low)
    TECA_ALGORITHM_PROPERTY(double, river_end_lat_high)
    TECA_ALGORITHM_PROPERTY(double, river_end_lon_high)

    // set/get the area as a percent of the search space that
    // a potential river must occupy
    TECA_ALGORITHM_PROPERTY(double, percent_in_mesh)

    // set/get the minimum river width and length. defaults
    // are 1250 2000
    TECA_ALGORITHM_PROPERTY(double, river_width)
    TECA_ALGORITHM_PROPERTY(double, river_length)

    // set/get the land-sea mask variable. this array
    // will be used to identify land from ocean using
    // land_threshold properties.
    TECA_ALGORITHM_PROPERTY(std::string, land_sea_mask_variable)

    // set/get the land classification range [low high). defaults
    // are [1.0 DOUBLE_MAX)
    TECA_ALGORITHM_PROPERTY(double, land_threshold_low)
    TECA_ALGORITHM_PROPERTY(double, land_threshold_high)

    // send humand readable representation to the
    // stream
    virtual void to_stream(std::ostream &os) const override;

protected:
    teca_ar_detect();

    // helper that computes the output extent
    int get_active_extent(
        p_teca_variant_array lat,
        p_teca_variant_array lon,
        std::vector<unsigned long> &extent) const;

private:
    virtual
    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    virtual
    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    virtual
    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string water_vapor_variable;
    std::string land_sea_mask_variable;
    double low_water_vapor_threshold;
    double high_water_vapor_threshold;
    double search_lat_low;
    double search_lon_low;
    double search_lat_high;
    double search_lon_high;
    double river_start_lat_low;
    double river_start_lon_low;
    double river_end_lat_low;
    double river_end_lon_low;
    double river_end_lat_high;
    double river_end_lon_high;
    double percent_in_mesh;
    double river_width;
    double river_length;
    double land_threshold_low;
    double land_threshold_high;
};

#endif
