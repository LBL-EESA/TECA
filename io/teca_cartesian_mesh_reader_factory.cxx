#include "teca_cartesian_mesh_reader_factory.h"

#include "teca_config.h"
#include "teca_file_util.h"
#include "teca_cartesian_mesh_reader.h"
#if defined(TECA_HAS_NETCDF)
#include "teca_cf_reader.h"
#include "teca_multi_cf_reader.h"
#endif

#include <string>

// --------------------------------------------------------------------------
p_teca_algorithm teca_cartesian_mesh_reader_factory::New(const std::string &file)
{
    std::string ext = teca_file_util::extension(file);

    if (ext == "nc")
    {
#if !defined(TECA_HAS_NETCDF)
        TECA_ERROR("Failed to construct an instance of teca_cf_reader "
            "because was not compiled with NetCDF features enabled")
        return nullptr;
#else
        p_teca_cf_reader r = teca_cf_reader::New();
        r->set_files_regex(file);
        return r;
#endif
    }
    else if (ext == "bin")
    {
        p_teca_cartesian_mesh_reader r = teca_cartesian_mesh_reader::New();
        r->set_file_name(file);
        return r;
    }
    else if (ext == "mcf")
    {
#if !defined(TECA_HAS_NETCDF)
        TECA_ERROR("Failed to construct an instance of teca_multi_cf_reader "
            "because was not compiled with NetCDF features enabled")
        return nullptr;
#else
        p_teca_multi_cf_reader r = teca_multi_cf_reader::New();
        r->set_input_file(file);
        return r;
#endif
    }
    else
    {
        TECA_ERROR("Failed to create a mesh reader from the file \""
            << file << "\" and extension \"" << ext << "\"")
        return nullptr;
    }
}
