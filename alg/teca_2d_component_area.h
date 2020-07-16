#ifndef teca_2d_component_area_h
#define teca_2d_component_area_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_2d_component_area)

/// an algorithm that computes the area of labeled regions
/**
Given a set of labels on a Cartesian mesh, the algorithm computes the area of
each region. Regions are identified by assigning a unique integer value to each
mesh point that belongs in the region. The component_variable property names
the variable containing the region labels.

if the region labels start at 0 and are contiguous then an optimization can be
used. Set contiguous_component_ids property to enable the optimization. Note that
TECA's connected component labeler assigns the background (i.e. cells not inside
the segmentation) the label 0. One can identify the background region and area
via this label. When processing data generated outside of TECA it might be
necessary to supply the background label. Use -2 if there is no background.

the input dataset is passed through and the results of the calculations are
stored in the output dataset metadata in the following keys:

    number_of_components - number of component ids for which area was
                           computed. Note that this can include a background
                           component i.e. for cells outside of the segmentation.

    component_ids - a vector containing the label of each component. This is
                    always starts with 0, where the label 0 identifies cells
                    out side of the segmentation, and ranges up to
                    number_of_components - 1, where the labels from 1 up to
                    number_of_components - 1 identify connected regions of
                    cells inside the segmentation.

    component_area - a vector containing the area for the corresponding entry
                     in the component_ids array.

    background_id - the label used for cells outside of the segmentation,
                    i.e. the background. This can be used to skip processing
                    of the background when desirable.
*/
class teca_2d_component_area : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_2d_component_area)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_2d_component_area)
    TECA_ALGORITHM_CLASS_NAME(teca_2d_component_area)
    ~teca_2d_component_area();

    // report/initialize to/from Boost program options objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the name of the input array
    TECA_ALGORITHM_PROPERTY(std::string, component_variable)

    // set this only if you know for certain that label ids are contiguous and
    // start at 0. this enables use of a faster implementation.
    TECA_ALGORITHM_PROPERTY(int, contiguous_component_ids)

    // set this to override the component label used for background. By default
    // this is set to -1 to indicate that the value should be obtained from the
    // metadata key `background_id`.  Note that TECA's connected component
    // labeler uses the id 0 for the background and passes this in a metadata
    // key and as a result no action is required.
    TECA_ALGORITHM_PROPERTY(long, background_id)

protected:
    teca_2d_component_area();

    int get_component_variable(std::string &component_var);

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
    int contiguous_component_ids;
    long background_id;
};

#endif
