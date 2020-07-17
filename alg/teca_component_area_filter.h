#ifndef teca_component_area_filter_h
#define teca_component_area_filter_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_component_area_filter)

/// An algorithm that applies a mask based on connected component area
/**
The filter masks the regions identified by an integer label that are outside
the range bracketed by the 'low_area_threshold' and 'high_area_threshold'
properties. These default to -inf and +inf, hence by default no regions are
masked. The mask value may be set by the 'mask_value' property which defaults
to '0'.

The filter expects an integer field containing connected component labels.
This field is named by the 'component_variable' property. Additionally a list
of label ids and coresponding areas is expected in the dataset metadata. The
properties 'component_ids_key' and 'component_area_key' identify the latter
metadata. These default to the names used by the 'teca_2d_component_area'
algotihm, 'component_ids' and 'component_area'.

Applying the 'teca_connected_component' algorithm followed by the
'teca_2d_component_area' algorithm is the easiest way to get valid inputs for
the 'component_area_filter'.

The filtered coomponent ids are put in the output dataset along with the
updated lists of valid component ids and component area metadata keys. By
default the filtered data replaces the input data in the output. However, the
input data can be retained by setting the 'variable_post_fix' property, a
string that will be appended to the names of the filtered component array and
metadata keys.
*/
class teca_component_area_filter : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_component_area_filter)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_component_area_filter)
    TECA_ALGORITHM_CLASS_NAME(teca_component_area_filter)
    ~teca_component_area_filter();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the name of the input array containing connected
    // component labels
    TECA_ALGORITHM_PROPERTY(std::string, component_variable)

    // set the name of the dataset metadata key holding the number of
    // components left after the filter is applied
    TECA_ALGORITHM_PROPERTY(std::string, number_of_components_key)

    // set the name of the dataset metadata key holding connected component
    // label ids
    TECA_ALGORITHM_PROPERTY(std::string, component_ids_key)

    // set the name of the dataset metadata key holding connected component
    // areas
    TECA_ALGORITHM_PROPERTY(std::string, component_area_key)

    // set this to be the default label id for the filtered out component
    // areas. This will typically correspond to the label used for cells
    // outside of the segmentation (i.e. in the background). One can use this
    // property to override the mask value.  The default mask value is '-1'
    // which results in aquiring the mask value from input metadata key
    // `background_id`. Use -2 to specify no background label.
    TECA_ALGORITHM_PROPERTY(long, mask_value)

    // set the range identifying values to area filter.
    // The defaults are (-infinity, infinity].
    TECA_ALGORITHM_PROPERTY(double, low_area_threshold)
    TECA_ALGORITHM_PROPERTY(double, high_area_threshold)

    // a string to be appended to the name of the output variable.
    // setting this to an empty string will result in the masked array
    // replacing the input array in the output. default is an empty
    // string ""
    TECA_ALGORITHM_PROPERTY(std::string, variable_post_fix)

    // set this only if you know for certain that label ids
    // are contiguous and start at 0. this enables use of a
    // faster implementation.
    TECA_ALGORITHM_PROPERTY(int, contiguous_component_ids)

protected:
    teca_component_area_filter();

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
    std::string component_variable;
    std::string number_of_components_key;
    std::string component_ids_key;
    std::string component_area_key;
    long mask_value;
    double low_area_threshold;
    double high_area_threshold;
    std::string variable_post_fix;
    int contiguous_component_ids;
};

#endif
