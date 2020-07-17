#include "teca_connected_components.h"

#include "teca_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_cartesian_mesh.h"
#include "teca_array_attributes.h"

#include <algorithm>
#include <iostream>
#include <deque>
#include <cmath>
#include <sstream>

using std::cerr;
using std::endl;

//#define TECA_DEBUG
namespace {

/// hold i,j,k index triplet
struct id3
{
    id3() : i(0), j(0), k(0) {}
    id3(unsigned long p, unsigned long q, unsigned long r)
        : i(p), j(q), k(r) {}

    unsigned long i;
    unsigned long j;
    unsigned long k;
};

/// 2D/3D connected component labeler
/**
given seed(i0,j0,k0) that's in a component to label, the
current component(current_component), a binary segmentation(segments),
and a set of components(components) of dimensions nx,ny,nz,nxy,
walk the segmentation from the seed labeling it as we go.
when this function returns this component is completely
labeled. this is the 1 pass algorithm.
*/
template <typename segment_t, typename component_t>
void non_periodic_labeler(unsigned long i0, unsigned long j0, unsigned long k0,
    component_t current_component, unsigned long nx, unsigned long ny,
    unsigned long nz, unsigned long nxy, const segment_t *segments,
    component_t *components)
{
    std::deque<id3> work_queue;
    work_queue.push_back(id3(i0,j0,k0));

    while (work_queue.size())
    {
        id3 ijk = work_queue.back();
        work_queue.pop_back();

        long s0 = ijk.k > 0 ? -1 : 0;
        long s1 = ijk.k < nz-1 ? 1 : 0;
        for (long s = s0; s <= s1; ++s)
        {
            unsigned long ss = ijk.k + s;
            unsigned long kk = ss*nxy;

            long r0 = ijk.j > 0 ? -1 : 0;
            long r1 = ijk.j < ny-1 ? 1 : 0;
            for (long r = r0; r <= r1; ++r)
            {
                unsigned long rr = ijk.j + r;
                unsigned long jj = rr*nx;

                long q0 = ijk.i > 0 ? -1 : 0;
                long q1 = ijk.i < nx-1 ? 1 : 0;
                long q_inc = (r || s) ? 1 : 2;
                for (long q = q0; q <= q1; q += q_inc)
                {
                    unsigned long qq = ijk.i + q;
                    unsigned long w = qq + jj + kk;

                    if (segments[w] && !components[w])
                    {
                        components[w] = current_component;
                        work_queue.push_back(id3(qq,rr,ss));
                    }
                }
            }
        }
    }
}

/// 2D/3D connected component labeler, with periodic boundary in x
/**
given seed(i0,j0,k0) that's in a component to label, the
current component(current_component), a binary segmentation(segments),
and a set of components(components) of dimensions nx,ny,nz,nxy,
walk the segmentation from the seed labeling it as we go.
when this function returns this component is completely
labeled. this is the 1 pass algorithm.

notes:
if we have a periodic bc then neighborhood includes cells -1 to 1, relative
to the current index, else the neighborhood is constrained to 0 to 1, or
-1 to 0.

    long s0 = periodic_in_z ? -1 : ijk.k > 0 ? -1 : 0;
    long s1 = periodic_in_z ? 1 : ijk.k < nz-1 ? 1 : 0;

then when an index goes out of bounds because the neighborhood crosses the
periodic bc

    ss = (ss + nz) % nz;

wraps it around
*/
template <typename segment_t, typename component_t>
void periodic_labeler(unsigned long i0, unsigned long j0, unsigned long k0,
    component_t current_component, unsigned long nx, unsigned long ny,
    unsigned long nz, unsigned long nxy, int periodic_in_x, int periodic_in_y,
    int periodic_in_z, const segment_t *segments,
    component_t *components)
{
    std::deque<id3> work_queue;
    work_queue.push_back(id3(i0,j0,k0));

    while (work_queue.size())
    {
        id3 ijk = work_queue.back();
        work_queue.pop_back();

        long s0 = periodic_in_z ? -1 : ijk.k > 0 ? -1 : 0;
        long s1 = periodic_in_z ? 1 : ijk.k < nz-1 ? 1 : 0;
        for (long s = s0; s <= s1; ++s)
        {
            unsigned long ss = ijk.k + s;
            ss = (ss + nz) % nz;
            unsigned long kk = ss*nxy;

            long r0 = periodic_in_y ? -1 : ijk.j > 0 ? -1 : 0;
            long r1 = periodic_in_y ? 1 : ijk.j < ny-1 ? 1 : 0;
            for (long r = r0; r <= r1; ++r)
            {
                unsigned long rr = ijk.j + r;
                rr = (rr + ny) % ny;
                unsigned long jj = rr*nx;

                long q0 = periodic_in_x ? -1 : ijk.i > 0 ? -1 : 0;
                long q1 = periodic_in_x ? 1 : ijk.i < nx-1 ? 1 : 0;
                long q_inc = (r || s) ? 1 : 2;
                for (long q = q0; q <= q1; q += q_inc)
                {
                    long qq = ijk.i + q;
                    qq = (qq + nx) % nx;
                    unsigned long w = qq + jj + kk;

                    if (segments[w] && !components[w])
                    {
                        components[w] = current_component;
                        work_queue.push_back(id3(qq,rr,ss));
                    }
                }
            }
        }
    }
}

/// 2D/3D connected component labeler driver
/**
given a binary segmentation(segments) and buffer(components), both
with dimensions described by the given exent(ext), compute
the labeling.
*/
template <typename segment_t, typename component_t>
void label(unsigned long *ext, int periodic_in_x, int periodic_in_y,
    int periodic_in_z, const segment_t *segments, component_t *components,
    component_t &max_component)
{
    unsigned long nx = ext[1] - ext[0] + 1;
    unsigned long ny = ext[3] - ext[2] + 1;
    unsigned long nz = ext[5] - ext[4] + 1;
    unsigned long nxy = nx*ny;

    // initialize the components
    component_t current_component = 0;
    memset(components, 0, nxy*nz*sizeof(component_t));

    // visit each element to see if it is a seed
    for (unsigned long k = 0; k < nz; ++k)
    {
        unsigned long kk = k*nxy;
        for (unsigned long j = 0; j < ny; ++j)
        {
            unsigned long jj = j*nx;
            for (unsigned long i = 0; i < nx; ++i)
            {
                unsigned long q = kk + jj + i;

                // found seed, label it
                if (segments[q] && !components[q])
                {
                    components[q] = ++current_component;
                    periodic_labeler(i,j,k, current_component,
                        nx,ny,nz,nxy, periodic_in_x, periodic_in_y,
                        periodic_in_z, segments, components);
                }
            }
        }
    }

    max_component = current_component;
}

};


// --------------------------------------------------------------------------
teca_connected_components::teca_connected_components() :
    component_variable(""), segmentation_variable("")
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_connected_components::~teca_connected_components()
{}

// --------------------------------------------------------------------------
std::string teca_connected_components::get_component_variable(
    const teca_metadata &request)
{
    std::string component_var = this->component_variable;
    if (component_var.empty())
    {
        if (request.has("component_variable"))
            request.get("component_variable", component_var);
        else if (this->segmentation_variable.empty())
            component_var = "components";
        else
            component_var = this->segmentation_variable + "_components";
    }
    return component_var;
}


// --------------------------------------------------------------------------
std::string teca_connected_components::get_segmentation_variable(
    const teca_metadata &request)
{
    std::string segmentation_var = this->segmentation_variable;

    if (segmentation_var.empty() &&
        request.has("segmentation_variable"))
            request.get("segmentation_variable",
                segmentation_var);

    return segmentation_var;
}

// --------------------------------------------------------------------------
teca_metadata teca_connected_components::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_connected_components::get_output_metadata" << endl;
#endif
    (void) port;

    std::string component_var = this->component_variable;
    if (component_var.empty())
    {
        if (this->segmentation_variable.empty())
            component_var = "components";
        else
            component_var = this->segmentation_variable + "_components";
    }

    // tell the downstream about the variable we produce
    teca_metadata md = input_md[0];
    md.append("variables", component_var);

    // add metadata for CF I/O
    teca_metadata atts;
    md.get("attributes", atts);

    std::ostringstream oss;
    oss << "the connected components of " << this->segmentation_variable;

    teca_array_attributes cc_atts(
        teca_variant_array_code<short>::get(),
        teca_array_attributes::point_centering,
        0, "unitless", component_var,
        oss.str().c_str());

    atts.set(component_var, (teca_metadata)cc_atts);

    md.set("attributes", atts);

    return md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_connected_components::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_connected_components::get_upstream_request" << endl;
#endif
    (void) port;
    (void) input_md;

    std::vector<teca_metadata> up_reqs;

    // get the name of the array to request
    std::string segmentation_var = this->get_segmentation_variable(request);
    if (segmentation_var.empty())
    {
        TECA_ERROR("A segmentation variable was not specified")
        return up_reqs;
    }

    // pass the incoming request upstream, and
    // add in what we need
    teca_metadata req(request);
    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);
    arrays.insert(segmentation_var);

    // remove fromt the request what we generate
    std::string component_var = this->get_component_variable(request);
    arrays.erase(component_var);

    req.set("arrays", arrays);

    // send up
    up_reqs.push_back(req);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_connected_components::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_connected_components::execute" << endl;
#endif
    (void)port;

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
    std::string segmentation_var = this->get_segmentation_variable(request);
    if (segmentation_var.empty())
    {
        TECA_ERROR("A segmentation variable was not specified")
        return nullptr;
    }

    const_p_teca_variant_array input_array
        = out_mesh->get_point_arrays()->get(segmentation_var);
    if (!input_array)
    {
        TECA_ERROR("The segmentation variable \"" << segmentation_var
            << "\" is not in the input")
        return nullptr;
    }

    // get mesh dimension
    unsigned long extent[6];
    out_mesh->get_extent(extent);

    unsigned long whole_extent[6];
    out_mesh->get_whole_extent(whole_extent);

    // check for periodic bc.
    int periodic_in_x = 0;
    out_mesh->get_periodic_in_x(periodic_in_x);
    if (periodic_in_x &&
        (extent[0] == whole_extent[0]) && (extent[1] == whole_extent[1]))
        periodic_in_x = 1;

    int periodic_in_y = 0;
    out_mesh->get_periodic_in_y(periodic_in_y);
    if (periodic_in_y &&
        (extent[2] == whole_extent[2]) && (extent[3] == whole_extent[3]))
        periodic_in_y = 1;

    int periodic_in_z = 0;
    out_mesh->get_periodic_in_z(periodic_in_z);
    if (periodic_in_z &&
        (extent[4] == whole_extent[4]) && (extent[5] == whole_extent[5]))
        periodic_in_z = 1;

    // do segmentation and component
    size_t n_elem = input_array->size();
    p_teca_short_array components = teca_short_array::New(n_elem);
    short *p_components = components->get();
    short max_component = 0;

    TEMPLATE_DISPATCH(const teca_variant_array_impl,
        input_array.get(),
        const NT *p_in = static_cast<TT*>(input_array.get())->get();
        ::label(extent, periodic_in_x, periodic_in_y,
            periodic_in_z, p_in, p_components, max_component);
        )

    // put components in output
    std::string component_var = this->get_component_variable(request);
    out_mesh->get_point_arrays()->set(component_var, components);

    // put the component ids in the metadata
    short num_components = max_component + 1;
    p_teca_short_array component_id = teca_short_array::New(num_components);
    for (short i = 0; i < num_components; ++i)
        component_id->set(i, i);

    teca_metadata &omd = out_mesh->get_metadata();
    omd.set("component_ids", component_id);
    omd.set("number_of_components", num_components);
    omd.set("background_id", short(0));

    return out_mesh;
}
