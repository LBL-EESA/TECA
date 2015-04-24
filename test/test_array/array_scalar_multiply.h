#ifndef array_scalar_multiply_h
#define array_scalar_multiply_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(array_scalar_multiply)

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
        const teca_metadata &input_md,
        std::string &active_array) const;

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
    std::string array_name;
    double scalar;
};

#endif
