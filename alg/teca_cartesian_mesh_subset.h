#ifndef teca_cartesian_mesh_subset_h
#define teca_cartesian_mesh_subset_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_variant_array_fwd.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_cartesian_mesh_subset)

/// applies a subset given in world coordinates to the upstream request
/**
an algorithm that applies a subset specified in
world coordinates to upstream requests. the subset
is specified as bounding box of the form [x_low to x_high,
y_low to y_high, z_low to z_high]. The subset can be either
the smallest subset containing the bounding box or the
largest set contained by the bounding box, and is controled
by the cover_bounds property.
*/
class teca_cartesian_mesh_subset : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_cartesian_mesh_subset)
    ~teca_cartesian_mesh_subset();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // define the bounding box of the subset
    // this algorithm converts this into an
    // extent into the upstream dataset.
    TECA_ALGORITHM_PROPERTY(std::vector<double>, bounds);

    void set_bounds(double low_x, double high_x,
        double low_y, double high_y, double low_z, double high_z)
    { this->set_bounds({low_x, high_x, low_y, high_y, low_z, high_z}); }

    // control how bounds are converted. if true
    // smallest subset covering the bounding box is
    // used. if false the largest subset contained
    // by the bounding box is used.
    TECA_ALGORITHM_PROPERTY(bool, cover_bounds)

protected:
    teca_cartesian_mesh_subset();

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
    std::vector<double> bounds;
    bool cover_bounds;

    // internals
    std::vector<unsigned long> extent;
};

#endif
