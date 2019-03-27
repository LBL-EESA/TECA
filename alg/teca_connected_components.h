#ifndef teca_connected_components_h
#define teca_connected_components_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_connected_components)

/// an algorithm that computes connected component componenting
/**
an algorithm that computes connected component labeling for 1D, 2D, and 3D
data. The components are computed from a binary segmentation provided on the
input.

the segmentation is labeled and stored in a variable named by the
segmentation_variable property. the component ids are added to the output
dataset metadata in an key named 'component_ids'
*/
class teca_connected_components : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_connected_components)
    ~teca_connected_components();

    // set the name of the output array
    TECA_ALGORITHM_PROPERTY(std::string, component_variable)

    // set the array to threshold
    TECA_ALGORITHM_PROPERTY(std::string, segmentation_variable)

protected:
    teca_connected_components();

    std::string get_component_variable(const teca_metadata &request);
    std::string get_segmentation_variable(const teca_metadata &request);

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
    std::string segmentation_variable;
};

#endif
