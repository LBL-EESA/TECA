#include "teca_mesh_padding.h"

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

// copy input array into a zeroes padded array
template <typename num_t>
void apply_padding(
    num_t *output, const num_t *input, size_t n_lat, size_t n_lon,
    size_t py_low, size_t px_low, size_t nx_new)
{
    size_t y_extent = py_low + n_lat; 
    size_t x_extent = px_low + n_lon;

    for (size_t j = py_low; j < y_extent; ++j)
    {
        size_t jj = (j - py_low) * n_lon;
        size_t jjj = j * nx_new;
        for (size_t i = px_low; i < x_extent; ++i)
        {
            output[jjj + i] = input[jj + i - px_low];
        }
    }
}

/*
// extend coordinate dimension to fit padding
template <typename crd>
void extend_coordinates(crd *output, const crd* input)
{
    TEMPLATE_DISPATCH_FP(
        teca_variant_array_impl,
        y_crds_new.get(),

        const NT* p_y_crds = static_cast<const TT*>(input.get())->get();
        NT* p_y_crds_new = static_cast<TT*>(y_crds_new.get())->get();

        teca_coordinate_util::expand_dimension(p_y_crds_new, p_y_crds, n_y,
                        py_low, py_high);

        out_mesh->set_y_coordinates(const_cast<const std::string&>(y_name), y_crds_new);
    )
}*/

};

// --------------------------------------------------------------------------
teca_mesh_padding::teca_mesh_padding() :
    py_low(0), py_high(0),
    px_low(0), px_high(0)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_mesh_padding::~teca_mesh_padding()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_mesh_padding::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_mesh_padding":prefix));
    
    opts.add_options()
        TECA_POPTS_GET(size_t, prefix, py_low,
            "set the y-dimenstion pad-low value. low means the negative "
            "side of the dimesnion")
        TECA_POPTS_GET(size_t, prefix, py_high,
            "set the y-dimenstion pad-high value. high means the positive "
            "side of the dimesnion")
        TECA_POPTS_GET(size_t, prefix, px_low,
            "set the x-dimenstion pad-low value. low means the negative "
            "side of the dimesnion")
        TECA_POPTS_GET(size_t, prefix, px_high,
            "set the x-dimenstion pad-high value. high means the positive "
            "side of the dimesnion")
        ;
    
    global_opts.add(opts);
}
// --------------------------------------------------------------------------
void teca_mesh_padding::set_properties(const std::string &prefix,
    variables_map &opts)
{
    TECA_POPTS_SET(opts, size_t, prefix, py_low)
    TECA_POPTS_SET(opts, size_t, prefix, py_high)
    TECA_POPTS_SET(opts, size_t, prefix, px_low)
    TECA_POPTS_SET(opts, size_t, prefix, px_high)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_mesh_padding::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_mesh_padding::get_output_metadata" << endl;
#endif
    (void)port;

    // add in the array we will generate
    teca_metadata out_md(input_md[0]);

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_mesh_padding::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_mesh_padding::get_upstream_request" << endl;
#endif
    (void) port;
    (void) input_md;

    std::vector<teca_metadata> up_reqs;

    // send up
    up_reqs.push_back(request);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_mesh_padding::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_mesh_padding::execute" << endl;
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

    size_t py_low = this->get_py_low();
    size_t py_high = this->get_py_high();
    size_t px_low = this->get_px_low();
    size_t px_high = this->get_px_high();

    // get the coordinate axes
    std::string y_name;
    std::string x_name;
    in_mesh->get_y_coordinate_variable(y_name);
    in_mesh->get_x_coordinate_variable(x_name);

    const_p_teca_variant_array y_crds = in_mesh->get_y_coordinates();
    const_p_teca_variant_array x_crds = in_mesh->get_x_coordinates();

    size_t n_y = y_crds->size();
    size_t n_x = x_crds->size();

    size_t ny_new = py_low + n_y + py_high;
    size_t nx_new = px_low + n_x + px_high;

    size_t nxy_new = nx_new * ny_new;


    unsigned long req_whole_extent[6];
    unsigned long req_extent[6];

    out_mesh->get_whole_extent(req_whole_extent);
    out_mesh->get_extent(req_extent);

    req_whole_extent[1] = nx_new - 1;
    req_whole_extent[3] = ny_new - 1;

    req_extent[0] = px_low + req_extent[0];
    req_extent[1] = px_low + req_extent[1];
    req_extent[2] = py_low + req_extent[2];
    req_extent[3] = py_low + req_extent[3];

    out_mesh->set_whole_extent(req_whole_extent);
    out_mesh->set_extent(req_extent);

    out_metadata.set("whole_extent", req_whole_extent, 6);
    out_metadata.set("extent", req_extent, 6);


    if (px_low || px_high)
    {
        p_teca_variant_array x_crds_new = x_crds->new_instance(px_low + n_x + px_high);

        TEMPLATE_DISPATCH_FP(
            teca_variant_array_impl,
            x_crds_new.get(),

            const NT* p_x_crds = static_cast<const TT*>(x_crds.get())->get();
            NT* p_x_crds_new = static_cast<TT*>(x_crds_new.get())->get();

            teca_coordinate_util::expand_dimension(p_x_crds_new, p_x_crds, n_x,
                            px_low, px_high);

            out_mesh->set_x_coordinates(const_cast<const std::string&>(x_name), x_crds_new);
        )
    }

    if (py_low || py_high)
    {
        p_teca_variant_array y_crds_new = y_crds->new_instance(py_low + n_y + py_high);

        TEMPLATE_DISPATCH_FP(
            teca_variant_array_impl,
            y_crds_new.get(),

            const NT* p_y_crds = static_cast<const TT*>(y_crds.get())->get();
            NT* p_y_crds_new = static_cast<TT*>(y_crds_new.get())->get();

            teca_coordinate_util::expand_dimension(p_y_crds_new, p_y_crds, n_y,
                            py_low, py_high);

            out_mesh->set_y_coordinates(const_cast<const std::string&>(y_name), y_crds_new);
        )
    }


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


        p_teca_variant_array padded_array = field_array->new_instance(nxy_new);

        TEMPLATE_DISPATCH_FP(
            teca_variant_array_impl,
            padded_array.get(),

            const NT* p_field_array = static_cast<const TT*>(field_array.get())->get();
            NT* p_padded_array = static_cast<TT*>(padded_array.get())->get();

            memset(p_padded_array, 0, ny_new*nx_new*sizeof(NT));

            ::apply_padding(p_padded_array, p_field_array,
                            n_y, n_x, py_low, px_low, nx_new);

            out_mesh->get_point_arrays()->set(field_var, padded_array);
        )
    }

    return out_mesh;
}

