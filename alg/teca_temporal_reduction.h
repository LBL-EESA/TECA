#ifndef teca_cpp_temporal_reduction_h
#define teca_cpp_temporal_reduction_h

#include "teca_shared_object.h"
#include "teca_threaded_algorithm.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"

#include <string>
#include <vector>
#include <map>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_cpp_temporal_reduction)

/**
 * Reduce a mesh across the time dimensions by a defined increment using
 * a defined operation.
 *
 *     time increments: daily, monthly, seasonal, yearly, n_steps, all
 *     reduction operations: average, summation, minimum, maximum
 *
 * The output time axis will be defined using the selected increment.
 * The output data will be accumulated/reduced using the selected
 * operation.
 *
 * By default the fill value will be obtained from metadata stored in the
 * NetCDF CF file (_FillValue). One may override this by explicitly calling
 * set_fill_value method with the desired fill value.
 *
 * For minimum and maximum operations, at given grid point only valid values
 * over the interval are used in the calculation. If there are no valid
 * values over the interval at the grid point it is set to the fill_value.
 *
 * For the averaging operation, during summation missing values are treated
 * as 0.0 and a per-grid point count of valid values over the interval is
 * maintained and used in the average. Grid points with no valid values over
 * the interval are set to the fill value.
 */
class TECA_EXPORT teca_cpp_temporal_reduction : public teca_threaded_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_cpp_temporal_reduction)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_cpp_temporal_reduction)
    TECA_ALGORITHM_CLASS_NAME(teca_cpp_temporal_reduction)
    ~teca_cpp_temporal_reduction();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name point_arrays
     * Set the list of arrays to reduce
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, point_array)
    ///@}

    /** @name operation
     * Set the reduction operation
     * default average
     */
    ///@{
    enum {
        average, ///< Set the reduction operation to average
        summation, ///< Set the reduction operation to summation
        minimum, ///< Set the reduction operation to minimum
        maximum ///< Set the reduction operation to maximum
    };

    TECA_ALGORITHM_PROPERTY(int, operation)

    int set_operation(const std::string &operation);

    std::string get_operation_name();
    ///@}

    /** @name interval
     * Set the type of interval iterator to create
     * default monthly
     */
    ///@{
    enum {
        daily = 2, ///< Set the time increment to daily
        monthly = 3, ///< Set the time increment to monthly
        seasonal = 4, ///< Set the time increment to seasonal
        yearly = 5, ///< Set the time increment to yearly
        n_steps = 6, ///< Set the time increment to n steps
        all = 7 ///< Set the time increment to all
    };

    TECA_ALGORITHM_PROPERTY(int, interval)

    int set_interval(const std::string &interval);

    std::string get_interval_name();
    ///@}

    /** @number_of_steps
     * For n_steps interval iterator,
     * the desired number of steps should be set.
     * default 0
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(long, number_of_steps)
    ///@}

    /** @name fill_value
     * Set the _FillValue attribute for the output data
     * default -1
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, fill_value)
    ///@}

    /** @step_per_request
     * Set the number of time steps per request
     * default 1
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(long, steps_per_request)
    ///@}

protected:
    teca_cpp_temporal_reduction();

private:
    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &md_in) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &md_in,
        const teca_metadata &req_in) override;

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &data_in,
        const teca_metadata &req_in,
        int streaming) override;

    using teca_algorithm::get_output_metadata;
    using teca_threaded_algorithm::execute;

private:
    std::vector<std::string> point_arrays;
    int operation;
    int interval;
    long number_of_steps;
    double fill_value;
    long steps_per_request;

    class internals_t;
    internals_t *internals;
};

#endif
