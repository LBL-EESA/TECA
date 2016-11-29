#include "teca_apply_binary_mask.h"

#include "teca_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_mesh.h"

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

// --------------------------------------------------------------------------
teca_apply_binary_mask::teca_apply_binary_mask() : mask_variable("")
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_apply_binary_mask::~teca_apply_binary_mask()
{}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_apply_binary_mask::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_apply_binary_mask::get_upstream_request" << endl;
#endif
    (void) port;
    (void) input_md;

    vector<teca_metadata> up_reqs;

    // get the name of the array to request
    if (this->mask_variable.empty())
    {
        TECA_ERROR("A mask variable was not specified")
        return up_reqs;
    }

    // pass the incoming request upstream, and
    // add in what we need
    teca_metadata req(request);
    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);
    arrays.insert(this->mask_variable);
    if (!this->mask_arrays.empty())
        arrays.insert(this->mask_arrays.begin(),
            this->mask_arrays.end());
    req.insert("arrays", arrays);

    // send up
    up_reqs.push_back(req);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_apply_binary_mask::execute(
    unsigned int port, const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_apply_binary_mask::execute" << endl;
#endif
    (void)port;
    (void)request;

    // get the input
    const_p_teca_mesh in_mesh =
        std::dynamic_pointer_cast<const teca_mesh>(input_data[0]);
    if (!in_mesh)
    {
        TECA_ERROR("empty input, or not a mesh")
        return nullptr;
    }

    // create output and copy metadata, coordinates, etc
    p_teca_mesh out_mesh =
        std::dynamic_pointer_cast<teca_mesh>(in_mesh->new_instance());
    out_mesh->copy(in_mesh);

    // get the mask array
    if (this->mask_variable.empty())
    {
        TECA_ERROR("A mask variable was not specified")
        return nullptr;
    }

    p_teca_array_collection arrays = out_mesh->get_point_arrays();

    p_teca_variant_array mask_array = arrays->get(this->mask_variable);
    if (!mask_array)
    {
        TECA_ERROR("mask variable \"" << this->mask_variable
            << "\" is not in the input")
        return nullptr;
    }

    // apply the mask
    unsigned long nelem = mask_array->size();

    NESTED_TEMPLATE_DISPATCH(teca_variant_array_impl,
        mask_array.get(), _1,

        NT_1 *pmask = static_cast<TT_1*>(mask_array.get())->get();

        unsigned int narrays = arrays->size();
        for (unsigned int i = 0; i < narrays; ++i)
        {
            // if the user provided a list, restrict masking to that
            // list. and if not, mask everything
            if (!this->mask_arrays.empty() &&
                !std::count(this->mask_arrays.begin(),
                this->mask_arrays.end(), arrays->get_name(i)))
                continue;

            p_teca_variant_array array = arrays->get(i);

            NESTED_TEMPLATE_DISPATCH(teca_variant_array_impl,
                array.get(), _2,

                NT_2 *parray = static_cast<TT_2*>(array.get())->get();

                for (unsigned long q = 0; q < nelem; ++q)
                {
                    parray[q] *= static_cast<NT_2>(pmask[q]);
                }
                )
        }
        )

    return out_mesh;
}
