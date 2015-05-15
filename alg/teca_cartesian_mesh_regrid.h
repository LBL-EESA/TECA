#ifndef teca_cartesian_mesh_regrid_h
#define teca_cartesian_mesh_regrid_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_variant_array_fwd.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_cartesian_mesh_regrid)

/// moves arrays from one cartesian mesh to another
/**
an algorithm that moves arrays from one cartesian mesh
to another using nearest or linear interpolation. the
first input is the target mesh. the second input is the
source mesh. the arrays to move from source to target
can be selected using add_array api or in the request
key regrid_source_arrays. arrays may be optionally renamed
as they are moved from one mesh to another. Target names
are specified  using add_array api or requets key
regrid_target_arrays.
*/
class teca_cartesian_mesh_regrid : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_cartesian_mesh_regrid)
    ~teca_cartesian_mesh_regrid();

    // clear the list of arrays to move
    void clear_arrays();

    // set names of arrays to move. any arrays
    // named here will be requested from the source
    // mesh(input 2) and moved onto the target mesh
    // (input 1) with the same name.
    void add_array(const std::string &array);

    // set names of arrays to move. any arrays
    // named in source_array will be requested
    // from the source mesh(input 2) and moved
    // onto the target mesh and renamed to
    // target_array.
    void add_array(
        const std::string &source_array,
        const std::string &target_array);

    // set the interpolation mode used in transfering
    // data between meshes of differing resolution.
    // in nearest mode value at the nearest grid point
    // is used, in linear mode bi/tri linear interpolation
    // is used.
    enum {
        nearest,
        linear
    };
    TECA_ALGORITHM_PROPERTY(int, interpolation_mode)

protected:
    teca_cartesian_mesh_regrid();

private:
    // TODO -- filter cache key for port 1, remove
    // everything but time step, extent, and arrays

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::vector<std::string> source_arrays;
    std::vector<std::string> target_arrays;
    int interpolation_mode;
};

#endif
