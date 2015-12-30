#ifndef teca_vtk_cartesian_mesh_writer_h
#define teca_vtk_cartesian_mesh_writer_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <vector>
#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_vtk_cartesian_mesh_writer)

/**
an algorithm that writes cartesian meshes in VTK format.
when VTK is found then the files are written using the
XML formats. otherwise legacy format is used. Can be
written as raw binary (default) or as ascii.
*/
class teca_vtk_cartesian_mesh_writer : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_vtk_cartesian_mesh_writer)
    ~teca_vtk_cartesian_mesh_writer();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the output filename. for time series the substring
    // %t% is replaced with the current time step.
    TECA_ALGORITHM_PROPERTY(std::string, file_name)

    // set the output type. can be binary or ascii.
    TECA_ALGORITHM_PROPERTY(int, binary)

protected:
    teca_vtk_cartesian_mesh_writer();

private:
    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string file_name;
    int binary;
};

#endif
