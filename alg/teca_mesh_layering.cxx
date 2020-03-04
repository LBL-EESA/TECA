#include "teca_mesh_layering.h"

#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_cartesian_mesh.h"
#include "teca_coordinate_util.h"
#include "teca_metadata_util.h"

#include <iostream>
#include <set>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#include <complex.h>

using std::cerr;
using std::endl;

namespace {

// layer/stack the input array
template <typename num_t>
void apply_layering(
    num_t *output, const num_t *input,
    size_t nz, size_t ny, size_t nx)
{
    for (size_t k = 0; k < nz; ++k)
    {
        size_t kk = k*ny*nx;
        for (size_t j = 0; j < ny; ++j)
        {
            size_t jj = j*nx;
            for (size_t i = 0; i < nx; ++i)
            {
                output[kk + jj + i] = input[jj + i];
            }
        }
    }
}

};

// --------------------------------------------------------------------------
teca_mesh_layering::teca_mesh_layering() :
    n_layers(0)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_mesh_layering::~teca_mesh_layering()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_mesh_layering::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_mesh_layering":prefix));
    
    opts.add_options()
        TECA_POPTS_GET(unsigned long, prefix, n_layers,
            "set the z-dimenstion n_layers value. Each 2D array "
            "will be layered above itself n_layers times")
        ;
    
    global_opts.add(opts);
}
// --------------------------------------------------------------------------
void teca_mesh_layering::set_properties(const std::string &prefix,
    variables_map &opts)
{
    TECA_POPTS_SET(opts, unsigned long, prefix, n_layers)
}
#endif


// --------------------------------------------------------------------------
teca_metadata teca_mesh_layering::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_mesh_layering::get_output_metadata" << endl;
#endif
    (void)port;

    // add in the array we will generate
    teca_metadata out_md(input_md[0]);

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_mesh_layering::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_mesh_layering::get_upstream_request" << endl;
#endif
    (void) port;
    (void) input_md;

    std::vector<teca_metadata> up_reqs;

    // send up
    up_reqs.push_back(request);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_mesh_layering::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_mesh_layering::execute" << endl;
#endif

    (void)port;
    (void)request;

    // get the input
    const_p_teca_cartesian_mesh in_mesh =
        std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_ERROR("empty input, or not a mesh")
        return nullptr;
    }

    // create output and copy metadata, coordinates, etc
    p_teca_cartesian_mesh out_mesh = teca_cartesian_mesh::New();
    
    out_mesh->copy_metadata(
        std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));

    teca_metadata &out_metadata = out_mesh->get_metadata();


    size_t n_layers = this->get_n_layers();

    if (n_layers > 1)
    {
        // get the coordinate axes
        std::string z_name;
        in_mesh->get_z_coordinate_variable(z_name);

        const_p_teca_variant_array x_crds = in_mesh->get_x_coordinates();
        const_p_teca_variant_array y_crds = in_mesh->get_y_coordinates();
        const_p_teca_variant_array z_crds = in_mesh->get_z_coordinates();

        size_t n_x = x_crds->size();
        size_t n_y = y_crds->size();
        size_t n_z = z_crds->size();

        if (n_z > 1)
        {
            TECA_ERROR("Cannot layer above 2D arrays")
            return nullptr;
        }

        size_t nxyz_new = n_x * n_y * n_layers;


        unsigned long req_whole_extent[6] = {0};
        unsigned long req_extent[6] = {0};

        req_whole_extent[1] = n_x - 1;
        req_whole_extent[3] = n_y - 1;
        req_whole_extent[5] = n_layers - 1;
        req_extent[1] = n_x - 1;
        req_extent[3] = n_y - 1;
        req_extent[5] = n_layers - 1;

        out_mesh->set_whole_extent(req_whole_extent);
        out_mesh->set_extent(req_extent);

        out_metadata.set("whole_extent", req_whole_extent, 6);
        out_metadata.set("extent", req_extent, 6);


        p_teca_variant_array z_crds_new = z_crds->new_instance(n_layers);

        TEMPLATE_DISPATCH_FP(
            teca_variant_array_impl,
            z_crds_new.get(),

            const NT* p_z_crds = static_cast<const TT*>(z_crds.get())->get();
            NT* p_z_crds_new = static_cast<TT*>(z_crds_new.get())->get();

            teca_coordinate_util::expand_dimension(p_z_crds_new, p_z_crds, 0,
                            0, n_layers);

            out_mesh->set_z_coordinates(const_cast<const std::string&>(z_name), z_crds_new);
        )


        size_t n_arrays = in_mesh->get_point_arrays()->size();
        for (size_t i = 0; i < n_arrays; ++i)
        {
            const_p_teca_variant_array field_array = in_mesh->get_point_arrays()->get(i);
            const std::string field_var = in_mesh->get_point_arrays()->get_name(i);
            if (!field_array)
            {
                if (!field_var.empty())
                {
                    TECA_ERROR("Field array \"" << field_var
                        << "\" not present.")
                }
                else
                {
                    TECA_ERROR("Field array at \"" << i
                        << "\" not present.")
                }
                return nullptr;
            }

            p_teca_variant_array layered_array = field_array->new_instance(nxyz_new);

            TEMPLATE_DISPATCH_FP(
                teca_variant_array_impl,
                layered_array.get(),

                const NT* p_field_array = static_cast<const TT*>(field_array.get())->get();
                NT* p_layered_array = static_cast<TT*>(layered_array.get())->get();


                ::apply_layering(p_layered_array, p_field_array,
                                n_layers, n_y, n_x);

                out_mesh->get_point_arrays()->set(field_var, layered_array);
            )
        }
    }

    return out_mesh;
}

