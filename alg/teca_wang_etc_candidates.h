#ifndef teca_wang_etc_candidates_h
#define teca_wang_etc_candidates_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_wang_etc_candidates)

/**
ETC candidate stage based on algorithm described in

  Climatology and Changes of Extratropical Cyclone Activity:
  Comparison of ERA-40 with NCEP–NCAR Reanalysis for 1958–2001

  XIAOLAN L. WANG, VAL R. SWAIL, AND FRANCIS W. ZWIERS

  JOURNAL OF CLIMATE, VOLUME 19, 2006

We have modified the approach in the following ways:
1) add test to exclude high elevations
2) remove test for equal pressure on 3x3 grid of neighbors
*/
class teca_wang_etc_candidates : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_wang_etc_candidates)
    ~teca_wang_etc_candidates();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set/get the name of input variables
    TECA_ALGORITHM_PROPERTY(std::string, vorticity_variable)
    TECA_ALGORITHM_PROPERTY(std::string, pressure_variable)
    TECA_ALGORITHM_PROPERTY(std::string, elevation_variable)

    // a candidate is defined as having:
    // 1. a local pressure minimum centered on the search window
    // 2. and pressure increases from the minimum by at least
    //    min_pressure_delta over the max_pressure_radius
    // 3. and surface elevation less than max_elevation
    // 4. and optionally, voricity greater than min_vorticity.
    TECA_ALGORITHM_PROPERTY(double, search_window)
    TECA_ALGORITHM_PROPERTY(double, min_vorticity)
    TECA_ALGORITHM_PROPERTY(double, min_pressure_delta)
    TECA_ALGORITHM_PROPERTY(double, max_pressure_radius)
    TECA_ALGORITHM_PROPERTY(double, max_elevation)

    // set/get the bounding box to search for storms
    // in units of degreees lat,lon
    TECA_ALGORITHM_PROPERTY(double, search_lat_low)
    TECA_ALGORITHM_PROPERTY(double, search_lat_high)
    TECA_ALGORITHM_PROPERTY(double, search_lon_low)
    TECA_ALGORITHM_PROPERTY(double, search_lon_high)

protected:
    teca_wang_etc_candidates();

    // helper that computes the output extent
    int get_active_extent(p_teca_variant_array lat,
        p_teca_variant_array lon, std::vector<unsigned long> &extent)
        const;

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
    std::string vorticity_variable;
    std::string pressure_variable;
    std::string elevation_variable;

    double min_vorticity;
    double min_pressure_delta;
    double max_pressure_radius;
    double max_elevation;
    double search_window;

    double search_lat_low;
    double search_lat_high;
    double search_lon_low;
    double search_lon_high;
};

#endif
