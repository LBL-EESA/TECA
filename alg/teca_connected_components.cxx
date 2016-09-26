#include "teca_connected_components.h"

#include "teca_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_cartesian_mesh.h"

#include <algorithm>
#include <iostream>
#include <deque>
#include <set>

using std::deque;
using std::vector;
using std::set;
using std::cerr;
using std::endl;

//#define TECA_DEBUG
namespace {

// set locations in the output where the input array
// has values within the low high range.
template <typename in_t, typename out_t>
void threshold(
    out_t *output, const in_t *input,
    size_t n_vals, in_t low, in_t high)
{
    for (size_t i = 0; i < n_vals; ++i)
        output[i] = ((input[i] >= low) && (input[i] <= high)) ? 1 : 0;
}

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
current label(current_label), a binary segmentation(segments),
and a set of labels(labels) of dimensions nx,ny,nz,nxy,
walk the segmentation from the seed labeling it as we go.
when this function returns this component is completely
labeled. this is the 1 pass algorithm.
*/
template <typename num_t>
void label(unsigned long i0, unsigned long j0, unsigned long k0,
    num_t current_label, unsigned long nx, unsigned long ny,
    unsigned long nz, unsigned long nxy, const num_t *segments,
    num_t *labels)
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

                    if (segments[w] && !labels[w])
                    {
                        labels[w] = current_label;
                        work_queue.push_back(id3(qq,rr,ss));
                    }
                }
            }
        }
    }
}

/// 2D/3D connected component label driver
/**
given a binary segmentation(segments) and buffer(labels), both
with dimensions described by the given exent(ext), compute
the labeling.
*/
template <typename num_t>
void label(unsigned long *ext, const num_t *segments, num_t *labels)
{
    unsigned long nx = ext[1] - ext[0] + 1;
    unsigned long ny = ext[3] - ext[2] + 1;
    unsigned long nz = ext[5] - ext[4] + 1;
    unsigned long nxy = nx*ny;

    // initialize the labels
    num_t current_label = 0;
    memset(labels, 0, nxy*nz*sizeof(num_t));

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
                if (segments[q] && !labels[q])
                {
                    labels[q] = ++current_label;
                    label(i,j,k, current_label,
                        nx,ny,nz,nxy, segments, labels);
                }
            }
        }
    }
}
};



// --------------------------------------------------------------------------
teca_connected_components::teca_connected_components() :
    label_variable(""), threshold_variable(""),
    low_threshold_value(std::numeric_limits<double>::lowest()),
    high_threshold_value(std::numeric_limits<double>::max())
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_connected_components::~teca_connected_components()
{}

// --------------------------------------------------------------------------
std::string teca_connected_components::get_label_variable(
    const teca_metadata &request)
{
    std::string label_var = this->label_variable;
    if (label_var.empty())
    {
        if (request.has("teca_connected_components::label_variable"))
            request.get("teca_connected_components::label_variable", label_var);
        else if (this->threshold_variable.empty())
            label_var = "labels";
        else
            label_var = this->threshold_variable + "labels";
    }
    return label_var;
}


// --------------------------------------------------------------------------
std::string teca_connected_components::get_threshold_variable(
    const teca_metadata &request)
{
    std::string threshold_var = this->threshold_variable;

    if (threshold_var.empty() &&
        request.has("teca_connected_components::threshold_variable"))
            request.get("teca_connected_components::threshold_variable",
                threshold_var);

    return threshold_var;
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


    std::string label_var = this->label_variable;
    if (label_var.empty())
    {
        if (this->threshold_variable.empty())
            label_var = "labels";
        else
            label_var = this->threshold_variable + "labels";
    }

    teca_metadata md = input_md[0];
    md.append("variables", label_var);
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

    vector<teca_metadata> up_reqs;

    // get the name of the array to request
    std::string threshold_var = this->get_threshold_variable(request);
    if (threshold_var.empty())
    {
        TECA_ERROR("A threshold variable was not specified")
        return up_reqs;
    }

    // pass the incoming request upstream, and
    // add in what we need
    teca_metadata req(request);
    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);
    arrays.insert(threshold_var);

    // remove fromt the request what we generate
    std::string label_var = this->get_label_variable(request);
    arrays.erase(label_var);

    req.insert("arrays", arrays);

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
    std::string threshold_var = this->get_threshold_variable(request);
    if (threshold_var.empty())
    {
        TECA_ERROR("A threshold variable was not specified")
        return nullptr;
    }

    const_p_teca_variant_array input_array
        = out_mesh->get_point_arrays()->get(threshold_var);
    if (!input_array)
    {
        TECA_ERROR("threshold variable \"" << threshold_var
            << "\" is not in the input")
        return nullptr;
    }

    // get threshold values
    double low = this->low_threshold_value;
    if (low == std::numeric_limits<double>::lowest()
        && request.has("teca_connected_components::low_threshold_value"))
        request.get("teca_connected_components::low_threshold_value", low);

    double high = this->high_threshold_value;
    if (high == std::numeric_limits<double>::max()
        && request.has("teca_connected_components::high_threshold_value"))
        request.get("teca_connected_components::high_threshold_value", high);

    // get mesh dimension
    unsigned long extent[6];
    out_mesh->get_extent(extent);

    // do segmentation and label
    size_t n_elem = input_array->size();
    p_teca_short_array labels = teca_short_array::New(n_elem);

    TEMPLATE_DISPATCH(const teca_variant_array_impl,
        input_array.get(),
        const NT *p_in = static_cast<TT*>(input_array.get())->get();
        short *p_seg = static_cast<short*>(malloc(sizeof(short)*n_elem));
        short *p_labels = labels->get();

        ::threshold(p_seg, p_in, n_elem,
            static_cast<NT>(low), static_cast<NT>(high));

        ::label(extent, p_seg, p_labels);

        free(p_seg);
        )

    // put labels in output
    std::string label_var = this->get_label_variable(request);
    out_mesh->get_point_arrays()->set(label_var, labels);

    return out_mesh;
}
