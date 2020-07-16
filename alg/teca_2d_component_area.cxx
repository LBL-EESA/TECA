#include "teca_2d_component_area.h"

#include "teca_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_cartesian_mesh.h"

#include <algorithm>
#include <iostream>
#include <deque>
#include <set>
#define _USE_MATH_DEFINES
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

namespace {

template <typename component_t>
component_t get_max_component_id(unsigned long n, const component_t *labels)
{
    component_t max_component_id = std::numeric_limits<component_t>::lowest();
    for (unsigned long i = 0; i < n; ++i)
    {
        component_t label = labels[i];
        max_component_id = label > max_component_id ? label : max_component_id;
    }
    return max_component_id;
}

// visit each node in the mesh, the node is treated as a cell in
// the dual mesh defined by mid points between nodes. the area of
// the cell is added to the corresponding label. this formulation
// requires a layer of ghost nodes
//
// The exact area of the sperical rectangular patch A_i is given by:
//
// A_i = rho^2(cos(phi_0) - cos(phi_1))(theta_1 - theta_0)
//
//     = rho^2(sin(rad_lat_0) - sin(rad_lat_1))(theta_1 - theta_0)
//
// where
//
//   theta = deg_lon * pi/180
//   phi = pi/2 - rad_lat
//   rad_lat = deg_lat * pi/180
//   sin(rad_lat) = cos(pi/2 - rad_lat)
//
template<typename coord_t, typename component_t, typename container_t>
void component_area(unsigned long nlon, unsigned long nlat,
    const coord_t *deg_lon, const coord_t *deg_lat, const component_t *labels,
    container_t &area)
{
    // This calculation is sensative to floating point precision and
    // should be done in double precision
    using calc_t = double;

    calc_t R_e = 6378.1370; // km
    calc_t R_e_sq = R_e*R_e;
    calc_t rad_per_deg = M_PI/180.0;

    unsigned long nlonm1 = nlon - 1;
    unsigned long nlatm1 = nlat - 1;

    // convert to spherical coordinates in units of radians,
    // move to the dual mesh, and pre-compute factors in A
    calc_t *rho_sq_d_theta = (calc_t*)malloc(nlon*sizeof(calc_t));
    rho_sq_d_theta[0] = calc_t();
    for (unsigned long i = 1; i < nlonm1; ++i)
        rho_sq_d_theta[i] = R_e_sq*calc_t(0.5)*(deg_lon[i + 1] - deg_lon[i - 1])*rad_per_deg;
    rho_sq_d_theta[nlonm1] = calc_t();

    calc_t *rad_lat = (calc_t*)malloc(nlat*sizeof(calc_t));
    for (unsigned long j = 0; j < nlat; ++j)
        rad_lat[j] = deg_lat[j]*rad_per_deg;

    calc_t *d_cos_phi = (calc_t*)malloc(nlat*sizeof(calc_t));
    for (unsigned long j = 1; j < nlatm1; ++j)
    {
        calc_t cos_phi_1 = sin(calc_t(0.5)*(rad_lat[j - 1] + rad_lat[j]));
        calc_t cos_phi_0 = sin(calc_t(0.5)*(rad_lat[j] + rad_lat[j + 1]));
        d_cos_phi[j] = cos_phi_0 - cos_phi_1;
    }
    d_cos_phi[0] = calc_t();
    d_cos_phi[nlatm1] = calc_t();

    // finish off the calc by multiplying the factors
    for (unsigned long j = 1; j < nlatm1; ++j)
    {
        calc_t d_cos_phi_j = d_cos_phi[j];
        unsigned long jj = j*nlon;
        for (unsigned long i = 1; i < nlonm1; ++i)
        {
            area[labels[jj + i]] += rho_sq_d_theta[i]*d_cos_phi_j;
        }
    }

    free(rad_lat);
    free(d_cos_phi);
    free(rho_sq_d_theta);
}

}



// --------------------------------------------------------------------------
teca_2d_component_area::teca_2d_component_area() :
    component_variable(""), contiguous_component_ids(0), background_id(-1)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_2d_component_area::~teca_2d_component_area()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_2d_component_area::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_2d_component_area":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, component_variable,
            "name of the varibale containing region labels")
        TECA_POPTS_GET(int, prefix, contiguous_component_ids,
            "when the region label ids start at 0 and are consecutive "
            "this flag enables use of an optimization (0)")
        TECA_POPTS_GET(long, prefix, background_id,
            "the label id that corresponds to the background (-1)")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_2d_component_area::set_properties(const std::string &prefix,
    variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, component_variable)
    TECA_POPTS_SET(opts, int, prefix, contiguous_component_ids)
    TECA_POPTS_SET(opts, long, prefix, background_id)
}
#endif

// --------------------------------------------------------------------------
int teca_2d_component_area::get_component_variable(std::string &component_var)
{
    if (this->component_variable.empty())
        return -1;

    component_var = this->component_variable;
    return 0;
}

// --------------------------------------------------------------------------
teca_metadata teca_2d_component_area::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_2d_component_area::get_output_metadata" << endl;
#endif
    (void) port;

    teca_metadata md = input_md[0];
    return md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_2d_component_area::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_2d_component_area::get_upstream_request" << endl;
#endif
    (void) port;
    (void) input_md;

    std::vector<teca_metadata> up_reqs;

    // get the name of the array to request
    std::string component_var;
    if (this->get_component_variable(component_var))
    {
        TECA_ERROR("component_variable was not specified")
        return up_reqs;
    }

    // pass the incoming request upstream, and
    // add in what we need
    teca_metadata req(request);
    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);
    arrays.insert(component_var);

    req.set("arrays", arrays);

    // send up
    up_reqs.push_back(req);
    return up_reqs;
}


// --------------------------------------------------------------------------
const_p_teca_dataset teca_2d_component_area::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_2d_component_area::execute" << endl;
#endif
    (void)port;
    (void)request;

    // get the input
    const_p_teca_cartesian_mesh in_mesh =
        std::dynamic_pointer_cast<const teca_cartesian_mesh>(
            input_data[0]);
    if (!in_mesh)
    {
        TECA_ERROR("empty input, or not a cartesian_mesh")
        return nullptr;
    }

    // create output and copy metadata, coordinates, etc
    p_teca_cartesian_mesh out_mesh = teca_cartesian_mesh::New();

    out_mesh->shallow_copy(
        std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));

    // get the input array
    std::string component_var;
    if (this->get_component_variable(component_var))
    {
        TECA_ERROR("component_variable was not specified")
        return nullptr;
    }

    const_p_teca_variant_array component_array
        = out_mesh->get_point_arrays()->get(component_var);
    if (!component_array)
    {
        TECA_ERROR("label variable \"" << component_var
            << "\" is not in the input")
        return nullptr;
    }

    // get mesh dimension
    unsigned long extent[6];
    out_mesh->get_extent(extent);

    unsigned long nx = extent[1] - extent[0] + 1;
    unsigned long ny = extent[3] - extent[2] + 1;
    unsigned long nxy = nx*ny;

    unsigned long nz = extent[5] - extent[4] + 1;
    if (nz != 1)
    {
        TECA_ERROR("This calculation requires 2D data. The current dataset "
            "extents are [" << extent[0] << ", " << extent[1] << ", "
            << extent[2] << ", " << extent[3] << ", " << extent[4] << ", "
            << extent[5] << "]")
        return nullptr;
    }

    // get the coordinate axes
    const_p_teca_variant_array xc = in_mesh->get_x_coordinates();
    const_p_teca_variant_array yc = in_mesh->get_y_coordinates();

    // get the input and output metadata
    teca_metadata &in_metadata =
        const_cast<teca_metadata&>(in_mesh->get_metadata());

    teca_metadata &out_metadata = out_mesh->get_metadata();

    // get the background_id, and pass it through
    long bg_id = this->background_id;
    if (this->background_id == -1)
    {
        if (in_metadata.get("background_id", bg_id))
        {
            TECA_ERROR("Metadata is missing the key \"background_id\". "
                "One should specify it via the \"background_id\" algorithm "
                "property")
            return nullptr;
        }
    }
    out_metadata.set("background_id", bg_id);

    // calculate area of components
    NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
        xc.get(),
        _COORD,
        // the calculation is sensative to floating point precision
        // and should be made in double precision
        using calc_t = double;

        const NT_COORD *p_xc = static_cast<TT_COORD*>(xc.get())->get();
        const NT_COORD *p_yc = static_cast<TT_COORD*>(yc.get())->get();

        NESTED_TEMPLATE_DISPATCH_I(const teca_variant_array_impl,
            component_array.get(),
            _LABEL,

            const NT_LABEL *p_labels = static_cast<TT_LABEL*>(component_array.get())->get();

            unsigned int n_labels = 0;

            bool has_component_id = in_metadata.has("component_ids");
            if (this->contiguous_component_ids || has_component_id)
            {
                // use a contiguous buffer to hold the result, only for
                // contiguous lables that start at 0
                p_teca_variant_array component_id;
                if (has_component_id)
                {
                    in_metadata.get("number_of_components", int(0), n_labels);
                    component_id = in_metadata.get("component_ids");
                }
                else
                {
                    NT_LABEL max_component_id = ::get_max_component_id(nxy, p_labels);
                    n_labels = max_component_id + 1;
                    p_teca_variant_array_impl<NT_LABEL> tmp = teca_variant_array_impl<NT_LABEL>::New(n_labels);
                    for (unsigned int i = 0; i < n_labels; ++i)
                        tmp->set(i, NT_LABEL(i));
                    component_id = tmp;
                }
                std::vector<calc_t> component_area(n_labels);
                ::component_area(nx,ny, p_xc,p_yc, p_labels, component_area);

                // transfer the result to the output
                out_metadata.set("number_of_components", n_labels);
                out_metadata.set("component_ids", component_id);
                out_metadata.set("component_area", component_area);
            }
            else
            {
                // use an associative array to handle any labels
                //std::map<NT_LABEL, NT_COORD> result;
                decltype(std::map<NT_LABEL, calc_t>()) result;
                ::component_area(nx,ny, p_xc,p_yc, p_labels, result);

                // transfer the result to the output
                n_labels = result.size();

                p_teca_variant_array_impl<NT_LABEL> component_id =
                    teca_variant_array_impl<NT_LABEL>::New(n_labels);

                p_teca_variant_array_impl<calc_t> component_area =
                    teca_variant_array_impl<calc_t>::New(n_labels);

                //std::map<NT_LABEL,NT_COORD>::iterator it = result.begin();
                auto it = result.begin();
                for (unsigned int i = 0; i < n_labels; ++i,++it)
                {
                    component_id->set(i, it->first);
                    component_area->set(i, it->second);
                }

                out_metadata.set("number_of_components", n_labels);
                out_metadata.set("component_ids", component_id);
                out_metadata.set("component_area", component_area);
            }
            )
        )

    return out_mesh;
}
