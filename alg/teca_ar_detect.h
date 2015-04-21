#ifndef teca_ar_detect_h
#define teca_ar_detect_h

#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <memory>
#include <string>
#include <vector>

class teca_ar_detect;

using p_teca_ar_detect = std::shared_ptr<teca_ar_detect>;
using const_p_teca_ar_detect = std::shared_ptr<const teca_ar_detect>;

/**
an example implementation of a teca_algorithm

meta data keys:

    consumes:

    requests:
*/
class teca_ar_detect : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_ar_detect)
    ~teca_ar_detect();

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
    // defaults are 180 19
    TECA_ALGORITHM_PROPERTY(double, river_start_lon_low)
    TECA_ALGORITHM_PROPERTY(double, river_start_lat_low)

    // set/get the river ladfall region in lat lon coordinate system
    // defaults are 233 238 29 56
    TECA_ALGORITHM_PROPERTY(double, river_end_lon_low)
    TECA_ALGORITHM_PROPERTY(double, river_end_lon_high)
    TECA_ALGORITHM_PROPERTY(double, river_end_lat_low)
    TECA_ALGORITHM_PROPERTY(double, river_end_lat_high)

    // set/get the area as a percent of the search space that
    // a potential river must occupy
    TECA_ALGORITHM_PROPERTY(double, percent_in_mesh)

    // set/get the minimum river width and length. defaults
    // are 1250 2000
    TECA_ALGORITHM_PROPERTY(double, river_width)
    TECA_ALGORITHM_PROPERTY(double, river_length)


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
        const std::vector<teca_metadata> &input_md);

    virtual
    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request);

    virtual
    p_teca_dataset execute(
        unsigned int port,
        const std::vector<p_teca_dataset> &input_data,
        const teca_metadata &request);

private:
    std::string water_vapor_variable;
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
};

#endif
