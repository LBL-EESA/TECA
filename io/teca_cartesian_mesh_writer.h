#ifndef teca_cartesian_mesh_writer_h
#define teca_cartesian_mesh_writer_h

#include "teca_config.h"
#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <vector>
#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_cartesian_mesh_writer)

/// An algorithm that writes Cartesian meshes in VTK format.
/**
 * When VTK is found then the files are written using the
 * XML formats. Otherwise legacy format is used. Can be
 * written as raw binary (default) or as ASCII.
 */
class TECA_EXPORT teca_cartesian_mesh_writer : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_cartesian_mesh_writer)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_cartesian_mesh_writer)
    TECA_ALGORITHM_CLASS_NAME(teca_cartesian_mesh_writer)
    ~teca_cartesian_mesh_writer();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the output filename. for time series the substring
    // %t% is replaced with the current time step.
    TECA_ALGORITHM_PROPERTY(std::string, file_name)

    // set the output type. can be binary or ASCII.
    TECA_ALGORITHM_PROPERTY(int, binary)

    // Select the output file format. 0:bin, 1:vtr, 2:vtk, 3:auto
    // the default is bin.
    enum {format_bin, format_vtk, format_vtr, format_auto};
    TECA_ALGORITHM_PROPERTY(int, output_format)
    void set_output_format_bin(){ this->set_output_format(format_bin); }
    void set_output_format_vtk(){ this->set_output_format(format_vtk); }
    void set_output_format_vtr(){ this->set_output_format(format_vtr); }
    void set_output_format_auto(){ this->set_output_format(format_auto); }

protected:
    teca_cartesian_mesh_writer();

private:
    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string file_name;
    int binary;
    int output_format;
};

#endif
