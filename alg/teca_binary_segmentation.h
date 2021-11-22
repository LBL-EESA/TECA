#ifndef teca_binary_segmentation_h
#define teca_binary_segmentation_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_binary_segmentation)

/// An algorithm that computes a binary segmentation.
/**
 * an algorithm that computes a binary segmentation for 1D, 2D, and 3D data.
 * The segmentation is computed using threshold operation where values in a
 * range (low, high] are in the segmentation (assigned 1). Values outside the
 * range are outside of the segmentation (assigned 0).
 *
 * The algorithm has 2 modes, BY_VALUE and BY_PERCENTILE. In the BY_VALUE mode,
 * the test for inclusion is applied on the raw data. In the BY_PERCENTILE mode
 * the range is given in percentiles and each data point is converted to a
 * percentile before applying the test for inclusion.
*/
class TECA_EXPORT teca_binary_segmentation : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_binary_segmentation)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_binary_segmentation)
    TECA_ALGORITHM_CLASS_NAME(teca_binary_segmentation)
    ~teca_binary_segmentation();

    /** @name segmentation_variable
     * set the name of the output array to store the resulting segmentation in
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, segmentation_variable)
    ///@}


    /** @name segmentation_variable_attributes
     * set extra metadata for the segmentation variable
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(teca_metadata, segmentation_variable_attributes)
    ///@}


    /** @name threshold_variable
     * set the name of the input array to segment
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, threshold_variable)
    ///@}

    /** @name low_threshold_value
     * Set the threshold range. The defaults are (-infinity, infinity].
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, low_threshold_value)
    ///@}

    /** @name high_threshold_value
     * Set the threshold range. The defaults are (-infinity, infinity].
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, high_threshold_value)
    ///@}

    /** @name threshold_mode
     * Set the threshold mode. In BY_PERCENTILE mode low and high thresholds
     * define the percentiles (0 to 100) between which data is in the
     * segmentation. default is BY_VALUE.
     */
    ///@{
    /// The threshold_mode modes
    enum {BY_VALUE=0, BY_PERCENTILE=1};

    TECA_ALGORITHM_PROPERTY(int, threshold_mode)

    /// Set the threshold_mode to percentile.
    void set_threshold_by_percentile() { set_threshold_mode(BY_PERCENTILE); }

    /// Set the threshold_mode to value.
    void set_threshold_by_value() { set_threshold_mode(BY_VALUE); }
    ///@}

protected:
    teca_binary_segmentation();

    int get_segmentation_variable(std::string &segmentation_var);
    int get_threshold_variable(std::string &threshold_var);

private:
    using teca_algorithm::get_output_metadata;

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
    teca_metadata segmentation_variable_attributes;
    std::string threshold_variable;
    double low_threshold_value;
    double high_threshold_value;
    int threshold_mode;
};

#endif
