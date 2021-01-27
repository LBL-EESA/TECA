#include "teca_cartesian_mesh_writer_factory.h"

#include "teca_config.h"
#include "teca_file_util.h"
#include "teca_cartesian_mesh_writer.h"
#if defined(TECA_HAS_NETCDF)
#include "teca_cf_writer.h"
#endif

#include <string>

// --------------------------------------------------------------------------
p_teca_algorithm teca_cartesian_mesh_writer_factory::New(const std::string &file)
{
    std::string ext = teca_file_util::extension(file);

    if (ext == "nc")
    {
#if !defined(TECA_HAS_NETCDF)
        TECA_ERROR("Failed to construct an instance of teca_cf_writer "
            "because was not compiled with NetCDF features enabled")
        return nullptr;
#else
        p_teca_cf_writer w = teca_cf_writer::New();
        w->set_file_name(file);
        return w;
#endif
    }
    else if (ext == "bin")
    {
        p_teca_cartesian_mesh_writer w = teca_cartesian_mesh_writer::New();
        w->set_file_name(file);
        return w;
    }
    else
    {
        TECA_ERROR("Failed to create a mesh writer from the file \""
            << file << "\" and extension \"" << ext << "\"")
        return nullptr;
    }
}

