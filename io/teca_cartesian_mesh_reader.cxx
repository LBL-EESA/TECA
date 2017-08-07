#include "teca_cartesian_mesh_reader.h"
#include "teca_cartesian_mesh.h"
#include "teca_binary_stream.h"
#include "teca_coordinate_util.h"
#include "teca_file_util.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <errno.h>

using std::string;
using std::endl;
using std::cerr;

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

// PIMPL idiom
struct teca_cartesian_mesh_reader::teca_cartesian_mesh_reader_internals
{
    teca_cartesian_mesh_reader_internals() {}

    void clear();

    static p_teca_cartesian_mesh read_cartesian_mesh(
        const std::string &file_name);

    p_teca_cartesian_mesh cartesian_mesh;
};

// --------------------------------------------------------------------------
void teca_cartesian_mesh_reader::teca_cartesian_mesh_reader_internals::clear()
{
    this->cartesian_mesh = nullptr;
}

// --------------------------------------------------------------------------
p_teca_cartesian_mesh
teca_cartesian_mesh_reader::teca_cartesian_mesh_reader_internals::read_cartesian_mesh(
    const std::string &file_name)
{
    // read the binary representation
    teca_binary_stream stream;
    if (teca_file_util::read_stream(file_name.c_str(),
        "teca_cartesian_mesh", stream))
    {
        TECA_ERROR("Failed to read teca_cartesian_mesh from \""
            << file_name << "\"")
        return nullptr;

    }

    // deserialize the binary rep
    p_teca_cartesian_mesh cartesian_mesh = teca_cartesian_mesh::New();
    cartesian_mesh->from_stream(stream);
    return cartesian_mesh;
}


// --------------------------------------------------------------------------
teca_cartesian_mesh_reader::teca_cartesian_mesh_reader() : generate_original_ids(0)
{
    this->internals = new teca_cartesian_mesh_reader_internals;
}

// --------------------------------------------------------------------------
teca_cartesian_mesh_reader::~teca_cartesian_mesh_reader()
{
    delete this->internals;
}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_cartesian_mesh_reader::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_cartesian_mesh_reader":prefix));

    opts.add_options()
        TECA_POPTS_GET(string, prefix, file_name,
            "a file name to read")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_reader::set_properties(const string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, string, prefix, file_name)
}
#endif

// --------------------------------------------------------------------------
void teca_cartesian_mesh_reader::set_modified()
{
    // clear cached metadata before forwarding on to
    // the base class.
    this->clear_cached_metadata();
    teca_algorithm::set_modified();
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_reader::clear_cached_metadata()
{
    this->internals->clear();
}

// --------------------------------------------------------------------------
teca_metadata teca_cartesian_mesh_reader::get_output_metadata(unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_cf_reader::get_output_metadata" << endl;
#endif
    (void)port;
    (void)input_md;

    // TODO
    // 1 use regex for multi step dataset
    // 2 read metadata without reading mesh

    // read the mesh if we have not already done so
    if (!this->internals->cartesian_mesh)
    {
        if (!(this->internals->cartesian_mesh =
            teca_cartesian_mesh_reader_internals::read_cartesian_mesh(this->file_name)))
        {
            TECA_ERROR("Failed to read the mesh")
            return teca_metadata();
        }

        this->internals->cartesian_mesh->get_metadata().set("number_of_time_steps", 1);
        this->internals->cartesian_mesh->get_metadata().to_stream(cerr);
    }

    return this->internals->cartesian_mesh->get_metadata();
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_cartesian_mesh_reader::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void)port;
    (void)input_data;

    // TODO
    // 1 handle request for specific time step
    // 2 handle spatial subseting

    return this->internals->cartesian_mesh;
}
