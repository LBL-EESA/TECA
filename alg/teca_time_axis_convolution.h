#ifndef teca_time_axis_convolution_h
#define teca_time_axis_convolution_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_time_axis_convolution)

/// An algorithm that applies a convolution along the time axis
/** Supports constant, Gaussian, and explicitly provided convolution kernels of
 * arbitarry width applied over forward, backweard or centered stencils.
 */
class teca_time_axis_convolution : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_time_axis_convolution)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_time_axis_convolution)
    TECA_ALGORITHM_CLASS_NAME(teca_time_axis_convolution)
    ~teca_time_axis_convolution();

    // report/initialize to/from Boost program options
    // objects.
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
    void set_stencil_type(const std::string &type);

    /// get the stencil as a string
    std::string get_stencil_type_name();
    ///@}

    /** @name kernel_weights
     * Set the kernel weights excplicitly. The number of weights defines the
     * stencil width and must be odd for a centered stencil.
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(double, kernel_weight)

    /// generate constant convolution kernel weights with the given filter width
    void set_constant_kernel_weights(unsigned int width);

    /** generate Gaussian convolution kernel weights with the given filter
     * width.
     *
     * @param[in] width the filter width
     * @param[in] a     peak height of the Gaussian
     * @param[in] B     center of the Gaussian (note coordinates range from -1 to 1.
     * @param[in] c     width of the Gaussian
     */
    void set_gaussian_kernel_weights(unsigned int width,
        double a = 1.0, double B = 0.0, double c = 0.55);

    /** set the kernel weights from a string name the kernel type. Default
     * kernel weight generator parameters are used.
     *
     * @param[in] name  the name of the kernel to generate. (gaussian, or constant)
     * @param[in] width the width of the kernel.
     */
    void set_kernel_weights(const std::string &name, int width);
    ///@}

    /** @name kernel_name
     * set the name of the user provided kernel. May be used in conjunction
     * with set_kernel_weights to provide additional output metadata.
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

    /** @name use_highpass
     * flags whether the weights should be converted to a highpass filter
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(bool, use_highpass)
    ///@}


protected:
    teca_time_axis_convolution();

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
    int stencil_type;
    std::vector<double> kernel_weights;
    std::string kernel_name;
    std::string variable_postfix;
    bool use_highpass;
};

#endif
