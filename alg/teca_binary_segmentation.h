#ifndef teca_binary_segmentation_h
#define teca_binary_segmentation_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_binary_segmentation)

/// an algorithm that computes a binary segmentation
/**
an algorithm that computes a binary segmentation for 1D, 2D,
and 3D data. The segmentation is computed using threshold
operation where values in a range (low, high] are assigned
1 else 0.
*/
class teca_binary_segmentation : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_binary_segmentation)
    ~teca_binary_segmentation();

    // set the name of the output array
    TECA_ALGORITHM_PROPERTY(std::string, segmentation_variable)

    // set the array to threshold
    TECA_ALGORITHM_PROPERTY(std::string, threshold_variable)

    // Set the threshold range. The defaults are
    // (-infinity, infinity].
    TECA_ALGORITHM_PROPERTY(double, low_threshold_value)
    TECA_ALGORITHM_PROPERTY(double, high_threshold_value)

protected:
    teca_binary_segmentation();

    std::string get_segmentation_variable(const teca_metadata &request);
    std::string get_threshold_variable(const teca_metadata &request);

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
    std::string segmentation_variable;
    std::string threshold_variable;
    double low_threshold_value;
    double high_threshold_value;
};

#endif
