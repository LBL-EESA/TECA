#ifndef teca_latitude_range_filter_h
#define teca_latitude_range_filter_h

#include "teca_config.h"
#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_latitude_range_filter)

/// Box‐filter for scalar fields based on latitude_range.
/// Passes values unchanged for latitude_ranges in [min_lat,max_lat] and
/// zeroes everything else.
///
/// Request keys:
///   teca_latitude_range_filter::filtered_variables
///   teca_latitude_range_filter::min_lat
///   teca_latitude_range_filter::max_lat
///
/// User‑specified values take precedence over request keys.
class TECA_EXPORT teca_latitude_range_filter : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_latitude_range_filter)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_latitude_range_filter)
    TECA_ALGORITHM_CLASS_NAME(tecaitude_filter)
    ~teca_latitude_range_filter();

    // Boost program‑options description
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // latitude_range limits (degrees)
    TECA_ALGORITHM_PROPERTY(double, min_lat)
    TECA_ALGORITHM_PROPERTY(double, max_lat)

    // list of variables to filter
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, filtered_variable)

    // postfix for filtered arrays (empty → replace input)
    TECA_ALGORITHM_PROPERTY(std::string, variable_postfix)

protected:
    teca_latitude_range_filter();

    // helpers to obtain filter limits from the request if not set
    int get_min_lat(const teca_metadata &request, double &min);
    int get_max_lat(const teca_metadata &request, double &max);

    // helper to obtain list of variables to filter
    int get_filtered_variables(std::vector<std::string> &vars);

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

private:
    double min_lat;
    double max_lat;
    std::vector<std::string> filtered_variables;
    std::string variable_postfix;
};

#endif