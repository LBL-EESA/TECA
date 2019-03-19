#ifndef teca_component_area_filter_h
#define teca_component_area_filter_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_component_area_filter)

/// an algorithm that filters labels (components) based on
/// the area of labeled regions. It replaces the filtered
/// out labels with a user chosen id, saved in variable
/// "filtered_label_id".

class teca_component_area_filter : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_component_area_filter)
    ~teca_component_area_filter();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the name of the input array
    TECA_ALGORITHM_PROPERTY(std::string, labels_variable)

    // set the name of the unique labels array
    TECA_ALGORITHM_PROPERTY(std::string, unique_labels_variable)

    // set the name of the labels areas array
    TECA_ALGORITHM_PROPERTY(std::string, areas_variable)

    // set this to be the default label id for the filtered
    // out component areas. The default will be '0'
    TECA_ALGORITHM_PROPERTY(int, filtered_label_id)

    // set the range identifying values to area filter.
    // The defaults are (-infinity, infinity].
    TECA_ALGORITHM_PROPERTY(double, low_threshold_value)
    TECA_ALGORITHM_PROPERTY(double, high_threshold_value)

    // a string to be appended to the name of the output variable.
    // setting this to an empty string will result in the damped array
    // replacing the input array in the output. default is an empty
    // string ""
    TECA_ALGORITHM_PROPERTY(std::string, variable_post_fix)

protected:
    teca_component_area_filter();

    std::string get_labels_variable(const teca_metadata &request);
    std::string get_unique_labels_variable(const teca_metadata &request);
    std::string get_areas_variable(const teca_metadata &request);

private:
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
    std::string labels_variable;
    std::string unique_labels_variable;
    std::string areas_variable;
    int filtered_label_id;
    double low_threshold_value;
    double high_threshold_value;
    std::string variable_post_fix;
};

#endif
