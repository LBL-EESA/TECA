#ifndef teca_normalize_coordinates_h
#define teca_normalize_coordinates_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_normalize_coordinates)

/// an algorithm to ensure that coordinates are in ascending order
/**
Transformations of coordinates and data to/from ascending order
are made as data and information pass up and down stream through
the algorithm.
*/
class teca_normalize_coordinates : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_normalize_coordinates)
    ~teca_normalize_coordinates();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

protected:
    teca_normalize_coordinates();

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
    struct internals_t;
    internals_t *internals;
};

#endif
