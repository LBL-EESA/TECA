#ifndef teca_connected_components_h
#define teca_connected_components_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_connected_components)

/// an algorithm that computes connected component labeling
/**
an algorithm that computes connected component labeling for 1D, 2D, and 3D
data. The components are computed from a binary segmentation provided on the
input.

the input binary segmentation is labeled and stored in a variable named by the
component_variable property. the component ids are added to the output
dataset metadata in an key named 'component_ids', and the number of components
is stored in a key named 'number_of_components'. These keys facilitate further
processing as one need not scan the labeled data to get the list of label ids.

The cells outside of the segmentation (i.e. the background) are always assigned
the label 0. The cells belonging to connected regions inside the segmentation
are labeled starting from 1 up to number_of_components - 1.

output keys:

    number_of_components - number of component ids found. this will always be
                           at least 1 long as the cells outside the segmentation
                           are assigned the label 0.

    component_ids - a vector containing the label of each component. This is
                    always starts with 0, where the label 0 identifies cells
                    out side of the segmentation, and ranges up to
                    number_of_components - 1, where the labels from 1 up to
                    number_of_components - 1 identify connected regions of
                    cells inside the segmentation.

    background_id - the label used for cells outside of the segmentation,
                    i.e. the background. always 0.
*/
class teca_connected_components : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_connected_components)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_connected_components)
    TECA_ALGORITHM_CLASS_NAME(teca_connected_components)
    ~teca_connected_components();

    // set the input array containing a binary segmentation
    // see teca_binary_segmentation
    TECA_ALGORITHM_PROPERTY(std::string, segmentation_variable)

    // set the name of the output array to store the component labels in
    TECA_ALGORITHM_PROPERTY(std::string, component_variable)


protected:
    teca_connected_components();

    std::string get_component_variable(const teca_metadata &request);
    std::string get_segmentation_variable(const teca_metadata &request);

private:
    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute( unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string component_variable;
    std::string segmentation_variable;
};

#endif
