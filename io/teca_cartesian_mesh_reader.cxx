#include "teca_cartesian_mesh_reader.h"
#include "teca_cartesian_mesh.h"
#include "teca_binary_stream.h"
#include "teca_coordinate_util.h"
#include "teca_file_util.h"
#include "teca_dataset_util.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <errno.h>

using std::endl;
using std::cerr;

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif


// PIMPL idiom
struct teca_cartesian_mesh_reader::teca_cartesian_mesh_reader_internals
{
    teca_cartesian_mesh_reader_internals() {}

    void clear();

    static p_teca_mesh read_cartesian_mesh(
        const std::string &file_name);

    p_teca_mesh mesh;
};

// --------------------------------------------------------------------------
void teca_cartesian_mesh_reader::teca_cartesian_mesh_reader_internals::clear()
{
    this->mesh = nullptr;
}

// --------------------------------------------------------------------------
p_teca_mesh
teca_cartesian_mesh_reader::teca_cartesian_mesh_reader_internals::read_cartesian_mesh(
    const std::string &file_name)
{
    // read the binary representation
    std::string header;
    teca_binary_stream stream;
    if (teca_file_util::read_stream(file_name.c_str(),
        "teca_cartesian_mesh_writer_v2", stream))
    {
        TECA_ERROR("Failed to read teca_cartesian_mesh from \""
            << file_name << "\"")
        return nullptr;
    }

    // construct the mesh
    int type_code = 0;
    stream.unpack(type_code);

    p_teca_mesh mesh = std::dynamic_pointer_cast<teca_mesh>
        (teca_dataset_factory::New(type_code));

    if (!mesh)
    {
        TECA_ERROR("Failed to construct an appropriate mesh type")
        return nullptr;
    }

    // deserialize the binary rep
    if (mesh->from_stream(stream))
    {
        TECA_ERROR("Failed to deserialize the \""
            << mesh->get_class_name() << "\"")
        return nullptr;
    }

    return mesh;
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
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_cartesian_mesh_reader":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, file_name,
            "a file name to read")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_reader::set_properties(const std::string &prefix,
    variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::string, prefix, file_name)
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
        << "teca_cartesian_mesh_reader::get_output_metadata" << endl;
#endif
    (void) port;
    (void) input_md;

    // TODO
    // 1 use regex for multi step dataset
    // 2 read metadata without reading mesh

    // read the mesh if we have not already done so
    if (!this->internals->mesh)
    {
        if (!(this->internals->mesh =
            teca_cartesian_mesh_reader_internals::read_cartesian_mesh(this->file_name)))
        {
            TECA_FATAL_ERROR("Failed to read the mesh from \"" << this->file_name << "\"")
            return teca_metadata();
        }
    }

    teca_metadata md = this->internals->mesh->get_metadata();
    md.set("index_initializer_key", std::string("number_of_meshes"));
    md.set("index_request_key", std::string("mesh_id"));
    md.set("number_of_meshes", 1l);

    return md;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_cartesian_mesh_reader::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_cartesian_mesh_reader::execute" << endl;
#endif
    (void) port;
    (void) input_data;
    (void) request;

    // TODO
    // 1 handle request for specific index
    // 2 handle spatial subseting
    // 3 Pass only requested arrays

    p_teca_dataset ds = this->internals->mesh->new_instance();
    ds->shallow_copy(this->internals->mesh);
    ds->set_request_index("mesh_id", 0l);

    return ds;
}
