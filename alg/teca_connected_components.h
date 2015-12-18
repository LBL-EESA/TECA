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
an algorithm that computes connected component labeling
for 1D, 2D, and 3D data. The labels are computed form a
binary segmentation which is computed using threshold
operation where values in a range (low, high] are in the
segmentation.
*/
class teca_connected_components : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_connected_components)
    ~teca_connected_components();

    // set the name of the output array
    TECA_ALGORITHM_PROPERTY(std::string, label_variable)

    // set the array to threshold
    TECA_ALGORITHM_PROPERTY(std::string, threshold_variable)

    // Set the threshold range. The defaults are
    // (-infinity, infinity].
    TECA_ALGORITHM_PROPERTY(double, low_threshold_value)
    TECA_ALGORITHM_PROPERTY(double, high_threshold_value)

protected:
    teca_connected_components();

    std::string get_label_variable(const teca_metadata &request);
    std::string get_threshold_variable(const teca_metadata &request);

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
    std::string label_variable;
    std::string threshold_variable;
    double low_threshold_value;
    double high_threshold_value;
};

#endif
