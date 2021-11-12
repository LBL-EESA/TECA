#ifndef teca_time_axis_convolution_h
#define teca_time_axis_convolution_h

#include "teca_config.h"
#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_time_axis_convolution)

/// An algorithm that applies a convolution along the time axis
/** Supports constant, Gaussian, and explicitly provided convolution kernels of
 * arbitrary width applied over forward, backward, or centered stencils.
 *
 * A number of options are provided for specifying kernel weights.
 *
 * * User provided kernels can be explicitly specified via the combination of
 *   ::set_kernel_weights and ::set_kernel_name.
 *
 * * The kernel can be generated at run time by providing a kernel name, width,
 *   and flag selecting either a high or low pass filter via the combination of
 *   ::set_kernel_name, ::set_kernel_width, and ::set_use_high_pass. In this
 *   case default kernel parameters are used.
 *
 * * The kernel can be generated with explicitly provided kernel parameters.
 */
class TECA_EXPORT teca_time_axis_convolution : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_time_axis_convolution)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_time_axis_convolution)
    TECA_ALGORITHM_CLASS_NAME(teca_time_axis_convolution)

    ~teca_time_axis_convolution();

    // report/initialize to/from Boost program options objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name stencil_type
     * Select which time steps are convolved. The default stencil is backward.
     */
    ///@{
    enum {
        backward, /// convolve with time steps before the active step
        centered, /// convolve with time steps centered on the active step
        forward   /// convolve with time steps after the active step
    };

    TECA_ALGORITHM_PROPERTY(int, stencil_type)

    /// convolve with time steps after the active step
    void set_stencil_type_to_forward()
    { this->stencil_type = forward; }

    /// convolve with time steps before the active step
    void set_stencil_type_to_backward()
    { this->stencil_type = backward; }

    /// convolve with time steps centered on the active step
    void set_stencil_type_to_centered()
    { this->stencil_type = centered; }

    /// set the stencil type from a string
    int set_stencil_type(const std::string &type);

    /// get the stencil as a string
    std::string get_stencil_type_name();
    ///@}

    /** @name use_high_pass
     * When set the internally generated weights will be converted to a
     * high_pass filter during weight generation. This option has no affect on
     * user provided kernel weights.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, use_high_pass)
    ///@}

    /** @name kernel_width
     * The number of samples to use when generating kernel weights. This option
     * has no affect on user provided kernel weights.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(unsigned int, kernel_width)
    ///@}

    /** @name kernel_weights
     * Set the kernel weights explicitly. The number of weights defines the
     * stencil width and must be odd for a centered stencil.
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(double, kernel_weight)

    /// generate constant convolution kernel weights with the given filter width
    int set_constant_kernel_weights(unsigned int width);

    /** generate Gaussian convolution kernel weights with the given filter
     * width.
     *
     * @param[in] width     the number of samples in the generated kernel
     * @param[in] high_pass transform the weights for use in a high pass filter.
     * @param[in] a         peak height of the Gaussian
     * @param[in] B         center of the Gaussian (note coordinates range from -1 to 1.
     * @param[in] c         width of the Gaussian
     */
    int set_gaussian_kernel_weights(unsigned int width, int high_pass = 0,
        double a = 1.0, double B = 0.0, double c = 0.55);

    /** set the kernel weights from a string name the kernel type. Default
     * kernel weight generator parameters are used.
     *
     * @param[in] name      the name of the kernel to generate. (gaussian, or constant)
     * @param[in] width     the width of the kernel.
     * @param[in] high_pass convert the weights for use as a high pass filter.
     */
    int set_kernel_weights(const std::string &name,
        unsigned int width, int high_pass);
    ///@}

    /** @name kernel_name
     * set the name of the user provided kernel, or the kernel to generate.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, kernel_name)
    ///@}

    /** @name variable_postfix
     * a string to be appended to the name of each output variable setting this
     * to an empty string will result in the damped array replacing the input
     * array in the output. default is an empty string ""
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, variable_postfix)
    ///@}

protected:
    teca_time_axis_convolution();

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
    int stencil_type;
    std::vector<double> kernel_weights;
    std::string kernel_name;
    std::string variable_postfix;
    int use_high_pass;
    unsigned int kernel_width;
};

#endif
