#ifndef array_scalar_multiply_h
#define array_scalar_multiply_h

#include "teca_algorithm.h"
#include "teca_meta_data.h"

#include <memory>
#include <string>
#include <vector>

class array_scalar_multiply;
typedef std::shared_ptr<array_scalar_multiply> p_array_scalar_multiply;

/**
an example implementation of a teca_algorithm
that multiplies arrays by a sclalar value

meta data keys:

    consumes:
        array_names

    requests:
        array_name
*/
class array_scalar_multiply : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(array_scalar_multiply)
    ~array_scalar_multiply();

    // set the scalar to multiply by
    TECA_ALGORITHM_PROPERTY(double, scalar)

    // set the name of the array to multiply (optional)
    TECA_ALGORITHM_PROPERTY(std::string, array_name)

protected:
    array_scalar_multiply();

    // helper that provides a default array to process
    // from the input meta data if the user has not
    // set one.
    int get_active_array(
        const teca_meta_data &input_md,
        std::string &active_array) const;

private:
    virtual
    teca_meta_data get_output_meta_data(
        unsigned int port,
        const std::vector<teca_meta_data> &input_md);

    virtual
    std::vector<teca_meta_data> get_upstream_request(
        unsigned int port,
        const std::vector<teca_meta_data> &input_md,
        const teca_meta_data &request);

    virtual
    p_teca_dataset execute(
        unsigned int port,
        const std::vector<p_teca_dataset> &input_data,
        const teca_meta_data &request);

private:
    std::string array_name;
    double scalar;
};

#endif
