#ifndef teca_temporal_reduction_h
#define teca_temporal_reduction_h

#include "teca_shared_object.h"
#include "teca_threaded_algorithm.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"

#include <string>
#include <vector>
#include <map>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_temporal_reduction)

class TECA_EXPORT reduction_operator_collection
{
public:
    double fill_value;

    virtual void initialize(double fill_value);

    virtual int update(int device_id,
          const const_p_teca_variant_array &out_array,
          const const_p_teca_variant_array &out_valid,
          const const_p_teca_variant_array &in_array,
          const const_p_teca_variant_array &in_valid,
          p_teca_variant_array &red_array,
          p_teca_variant_array &red_valid) {}

    virtual int finalize(int device_id,
          p_teca_variant_array &out_array,
          const p_teca_variant_array &out_valid,
          p_teca_variant_array &red_array);
};

class TECA_EXPORT average_operator : public reduction_operator_collection
{
public:
    p_teca_variant_array count;

    void initialize(double fill_value) override;

    int update(int device_id,
          const const_p_teca_variant_array &out_array,
          const const_p_teca_variant_array &out_valid,
          const const_p_teca_variant_array &in_array,
          const const_p_teca_variant_array &in_valid,
          p_teca_variant_array &red_array,
          p_teca_variant_array &red_valid) override;

    int finalize(int device_id,
          p_teca_variant_array &out_array,
          const p_teca_variant_array &out_valid,
          p_teca_variant_array &red_array) override;
};

class TECA_EXPORT summation_operator : public reduction_operator_collection
{
public:
    int update(int device_id,
         const const_p_teca_variant_array &out_array,
         const const_p_teca_variant_array &out_valid,
         const const_p_teca_variant_array &in_array,
         const const_p_teca_variant_array &in_valid,
         p_teca_variant_array &red_array,
         p_teca_variant_array &red_valid) override;
};

class TECA_EXPORT minimum_operator : public reduction_operator_collection
{
public:
    int update(int device_id,
          const const_p_teca_variant_array &out_array,
          const const_p_teca_variant_array &out_valid,
          const const_p_teca_variant_array &in_array,
          const const_p_teca_variant_array &in_valid,
          p_teca_variant_array &red_array,
          p_teca_variant_array &red_valid) override;
};

class TECA_EXPORT maximum_operator : public reduction_operator_collection
{
public:
    int update(int device_id,
          const const_p_teca_variant_array &out_array,
          const const_p_teca_variant_array &out_valid,
          const const_p_teca_variant_array &in_array,
          const const_p_teca_variant_array &in_valid,
          p_teca_variant_array &red_array,
          p_teca_variant_array &red_valid) override;
};

using p_reduction_operator = std::shared_ptr<reduction_operator_collection>;

class TECA_EXPORT reduction_operator_factory
{
public:
    /** Allocate and return an instance of the named operator
     * @param[in] op Name of the desired reduction operator.
     *               One of average, summation, minimum, or
     *                                              maximum
     * @returns an instance of reduction_operator_collection
     */
    static p_reduction_operator New(const std::string op);
};

struct TECA_EXPORT time_interval
{
    time_interval(double t, long start, long end) : time(t),
        start_index(start), end_index(end)
    {}

    double time;
    long start_index;
    long end_index;
};

using time_interval_t = time_interval;

/**
 * Reduce a mesh across the time dimensions by a defined increment using
 * a defined operation.
 *
 *     time increments: daily, monthly, seasonal, yearly, n_steps
 *     reduction operators: average, summation, minimum, maximum
 *
 * The output time axis will be defined using the selected increment.
 * The output data will be accumulated/reduced using the selected
 * operation.
 *
 * The set_use_fill_value method controls how invalid or missing values are
 * treated. When set to 1, NetCDF CF fill values are detected and handled.
 * This is the default. If it is known that the dataset has no invalid or
 * missing values one may set this to 0 for faster processing. By default the
 * fill value will be obtained from metadata stored in the NetCDF CF file
 * (_FillValue). One may override this by explicitly calling set_fill_value
 * method with the desired fill value.
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
class TECA_EXPORT teca_temporal_reduction : public teca_threaded_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_temporal_reduction)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_temporal_reduction)
    TECA_ALGORITHM_CLASS_NAME(teca_temporal_reduction)
    ~teca_temporal_reduction();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name point_arrays
     * Set the list of arrays to reduce
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, point_array)

    /** @name operator_name
     * Set the reduction operator
     * default "None"
     * It can be:
     *           average
     *           summation
     *           minimum
     *           maximum
     */
    ///@{
    int set_operator(const std::string &op);
    ///@}

    /** @name interval_name
     * Set the type of interval iterator to create
     * default "None"
     * It can be:
     *           daily
     *           monthly
     *           seasonal
     *           yearly
     *           n_steps
     * For the n_steps iterator replace n with the
     * desired number of steps (e.g. 8_steps)
     */
    ///@{
    int set_interval(const std::string &interval);
    ///@}

    /** @name fill_value
     * Set the _fillValue attribute for the output data
     * default -1
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, fill_value)

    /** @name use_fill_value
     * Control how invalid or missing values are treated
     * default 1
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, use_fill_value)

protected:
    teca_temporal_reduction();

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

private:
    std::vector<std::string> point_arrays;
    std::string operator_name;
    std::string interval_name;
    double fill_value;
    int use_fill_value;
    std::vector<time_interval_t> indices;
    std::map<std::string, p_reduction_operator> op;
};

#endif
