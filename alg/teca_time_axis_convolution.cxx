#include "teca_time_axis_convolution.h"

#include "teca_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_array_attributes.h"
#include "teca_string_util.h"
#include "teca_metadata_util.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <vector>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using namespace teca_variant_array_util;

//#define TECA_DEBUG


namespace internals
{
//*****************************************************************************
template <typename NT>
void linspace(const NT &lo, const NT &hi, size_t n, NT *data)
{
    // generate n equally spaced points on the segment [lo hi] on real line
    // R^1.

    if (n == 1)
    {
        data[0] = (hi + lo) / 2.0;
        return;
    }

    NT delta = (hi - lo) / (n - 1);

    for (size_t i = 0; i < n; ++i)
    {
        data[i] = lo + i*delta;
    }
}

//*****************************************************************************
template <typename NT>
NT gaussian(NT X, NT a, NT b, NT c)
{
    // x - evaluate at this location
    // a - peak height
    // b - center
    // c - width

    NT x = X - b;
    NT r2 = x*x;
    return a*exp(-r2/(2.0*c*c));
}

//*****************************************************************************
template <typename NT>
void swap_weights_highlow(NT *weights, size_t n_weights, bool high_to_low)
{
    // Converts lowpass weights to high_pass weights or vice versa
    // weights - filter weights
    // N - the number of weights
    // high_to_low - flags whether to convert from high to low pass

    // calculate the midpoint of the weights
    size_t nmid = (n_weights - 1ul) / 2ul;

    // subtract 1 from the center weight to convert to lowpass
    if (high_to_low)
        weights[nmid] += 1;

    // Negate the weights
    for (size_t i = 0; i < n_weights; ++i)
        weights[i] = -weights[i];

    // add 1 to the center weight to convert to high_pass
    if (! high_to_low)
        weights[nmid] += 1;
}
}



// --------------------------------------------------------------------------
teca_time_axis_convolution::teca_time_axis_convolution() :
    stencil_type(centered), kernel_name("user defined"),
    variable_postfix("_time_convolved"), use_high_pass(0)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_time_axis_convolution::~teca_time_axis_convolution()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_time_axis_convolution::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_time_axis_convolution":prefix));

    opts.add_options()
        TECA_POPTS_GET(int, prefix, stencil_type,
            "use a backward(0), forward(1) or centered(2) stencil")
        TECA_POPTS_MULTI_GET(std::vector<double>, prefix, kernel_weights,
            "the kernel weights to apply along the time dimension.")
        TECA_POPTS_GET(std::string, prefix, kernel_name,
            "the name of the kernel to generate, or a name for"
            " the user provided kernel. The internally generated"
            " kernels are \"gaussian\" and \"constant\".")
        TECA_POPTS_GET(unsigned int, prefix, kernel_width,
            "sets the number of samples in the kernel.")
        TECA_POPTS_GET(int, prefix, use_high_pass,
            "transform kernel weights during generation to construct"
            " a high_pass filter")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_time_axis_convolution::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, int, prefix, stencil_type)
    TECA_POPTS_SET(opts, std::vector<double>, prefix, kernel_weights)
    TECA_POPTS_SET(opts, std::string, prefix, kernel_name)
    TECA_POPTS_SET(opts, unsigned int, prefix, kernel_width)
    TECA_POPTS_SET(opts, int, prefix, use_high_pass)
}
#endif

// --------------------------------------------------------------------------
int teca_time_axis_convolution::set_constant_kernel_weights(unsigned int width)
{
    if (width < 2)
    {
        TECA_ERROR("Invalid kernel width " << width)
        return -1;
    }

    // compute the normalized weights
    this->kernel_weights.resize(width);
    for (unsigned int i = 0; i < width; ++i)
    {
        this->kernel_weights[i] = 1.0 / double(width);
    }

    this->set_kernel_name("constant");

    return 0;
}

// --------------------------------------------------------------------------
int teca_time_axis_convolution::set_gaussian_kernel_weights(
    unsigned int width, int high_pass, double a, double b, double c)
{
    if (width < 2)
    {
        TECA_ERROR("Invalid kernel width " << width)
        return -1;
    }

    // compute the coordinate axes [-1, to 1]
    std::vector<double> x(width);
    internals::linspace(-1.0, 1.0, width, x.data());

    // compute the weights and the normalization factor
    double norm = 0.0;
    this->kernel_weights.resize(width);
    for (unsigned int i = 0; i < width; ++i)
    {
        double gw = internals::gaussian(x[i], a, b, c);
        this->kernel_weights[i] = gw;
        norm += gw;
    }

    // normalize the weights
    for (unsigned int i = 0; i < width; ++i)
    {
        this->kernel_weights[i] /= norm;
    }

    // convert to high_pass if flagged
    if (high_pass)
    {
        internals::swap_weights_highlow(this->kernel_weights.data(),
            width, false);
    }

    // record the settings used for the NetCDF CF metadata
    this->set_use_high_pass(high_pass);

    std::string name = "gaussian";
    if (high_pass)
        name += "_high_pass";
    this->set_kernel_name(name);

    return 0;
}

// --------------------------------------------------------------------------
int teca_time_axis_convolution::set_stencil_type(const std::string &type)
{
    if (type == "backward")
    {
        this->set_stencil_type(backward);
    }
    else if (type == "forward")
    {
        this->set_stencil_type(forward);
    }
    else if (type == "centered")
    {
        this->set_stencil_type(centered);
    }
    else
    {
        TECA_ERROR("Invlaid stencil type \"" << type << "\"")
        return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
std::string teca_time_axis_convolution::get_stencil_type_name()
{
    std::string name;
    switch(this->stencil_type)
    {
        case backward:
            name = "backward";
            break;
        case centered:
            name = "centered";
            break;
        case forward:
            name = "forward";
            break;
        default:
            TECA_ERROR("Invalid \"stencil_type\" " << this->stencil_type)
    }
    return name;
}

// --------------------------------------------------------------------------
int teca_time_axis_convolution::set_kernel_weights(const std::string &name,
    unsigned int width, int high_pass)
{
    if (name == "constant")
    {
        if (this->set_constant_kernel_weights(width))
        {
            TECA_ERROR("Failed to generate constant kernel weights")
            return -1;
        }
    }
    else if (name == "gaussian")
    {
        if (this->set_gaussian_kernel_weights(width, high_pass))
        {
            TECA_ERROR("Failed to generate gaussian kernel weights")
            return -1;
        }
    }
    else
    {
        TECA_ERROR("Invalid kernel name \"" << name << "\"")
        return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
teca_metadata teca_time_axis_convolution::get_output_metadata(
    unsigned int port, const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_time_axis_convolution::get_output_metadata" << std::endl;
#endif
    (void)port;

    // copy the incoming metadata
    teca_metadata md_out(input_md[0]);

    // get the execution control keys
    std::string initializer_key;
    unsigned long n_indices_in = 0;

    if (md_out.get("index_initializer_key", initializer_key) ||
        md_out.get(initializer_key, n_indices_in))
    {
        TECA_FATAL_ERROR("Invalid execution control metadata."
            " Missing the index initializer")
        return teca_metadata();
    }

    // get the kernel width
    int width = this->kernel_weights.size();
    if (width == 0)
    {
        // check for user provided parameters
        if (this->kernel_name.empty() || (this->kernel_width == 0))
        {
            TECA_FATAL_ERROR("The kernel has not been specified correctly."
                " kernel_name=\"" << this->kernel_name << "\" kernel_width="
                << this->kernel_width << ".")
            return teca_metadata();
        }

        // generate the kernel based on user provided parameters
        if (this->set_kernel_weights(this->kernel_name, this->kernel_width,
            this->use_high_pass))
        {
            TECA_FATAL_ERROR("Failed to generate the kernel weights")
            return teca_metadata();
        }
    }

    // adjust the size of the time axis to account for incomplete windows of
    // data at the start and end of the time series
    long n_indices_out = n_indices_in - 2l * long(width / 2l);
    md_out.set(initializer_key, n_indices_out);

    // if postfix is not empty, report the new arrays that we could
    // generate, and copy attributes/metadata etc
    teca_metadata atts;
    md_out.get("attributes", atts);

    std::vector<std::string> vars_in;
    md_out.get("variables", vars_in);

    // copy the existing variables
    unsigned int n_vars = vars_in.size();
    std::vector<std::string> vars_out(vars_in);

    // when there is postfix there will be a new variable for each existing one
    if (!this->variable_postfix.empty())
        vars_out.reserve(2*n_vars);

    for (unsigned int i = 0; i < n_vars; ++i)
    {
        const std::string &var_in = vars_in[i];
        std::string var_out = var_in;

        // report the new variable
        if (!this->variable_postfix.empty())
        {
            var_out += this->variable_postfix;
            vars_out.push_back(var_out);
        }

        // copy and update the metadata
        teca_metadata var_in_atts;
        atts.get(var_in, var_in_atts);

        std::ostringstream oss;
        oss << this->get_stencil_type_name() << " " << this->kernel_name
            << " convolution of " << var_in << " over " << width << " time points";

        teca_array_attributes var_out_atts(var_in_atts);
        var_out_atts.description = oss.str();

        // update the attributes
        atts.set(var_out, (teca_metadata)var_out_atts);
    }

    // update the output metadata
    md_out.set("attributes", atts);
    md_out.set("variables", vars_out);

    return md_out;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_time_axis_convolution::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_time_axis_convolution::get_upstream_request" << std::endl;
#endif
    (void) port;

    std::vector<teca_metadata> up_reqs;

    const teca_metadata &md_in = input_md[0];

    // translate between the input and output time axis.  this is done because
    // we skip time steps at the start and/or of the series where full windows
    // of data are not available
    std::string initializer_key;
    long n_indices_in = 0;

    if (md_in.get("index_initializer_key", initializer_key) ||
        md_in.get(initializer_key, n_indices_in))
    {
        TECA_FATAL_ERROR("Invalid execution control metadata."
            " Missing the index initializer.")
        return up_reqs;
    }

    // get the kernel width
    int width = this->kernel_weights.size();
    if (width < 2)
    {
        TECA_FATAL_ERROR("The kernel weights have not been"
            " specified or are invalid. " << width << " weights")
        return up_reqs;
    }

    if ((this->stencil_type == centered) && (width % 2 == 0))
    {
        TECA_FATAL_ERROR("The kernel width should be odd"
            " for a centered stencil but the width is " << width)
        return up_reqs;
    }

    // get the time values required to compute the average centered on the
    // requested time
    std::string request_key;
    long active_index = 0;
    if (teca_metadata_util::get_requested_index(request,
        request_key, active_index))
    {
        TECA_FATAL_ERROR("Invalid execution control metadata."
            " Failed to get the requested index.")
        return up_reqs;
    }

    // intercept requests for arrays that we generate and request the
    // arrays that we need.
    std::set<std::string> arrays;
    if (request.has("arrays"))
        request.get("arrays", arrays);

    // Cleaning off the postfix for arrays passed in the pipeline.
    if (!this->variable_postfix.empty())
        teca_string_util::remove_postfix(arrays, this->variable_postfix);

    // make a request for each time that will be used in the average
    for (long i = 0; i < width; ++i)
    {
        unsigned long ii = active_index + i;
        teca_metadata up_req(request);
        up_req.set("arrays", arrays);
        up_req.set(request_key, {ii, ii});
        up_reqs.push_back(up_req);
    }

#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id() << " processing " << active_index
        << " request " << active_index << " - " << active_index + width - 1
        << std::endl;
#endif

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_time_axis_convolution::execute(
    unsigned int port, const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_time_axis_convolution::execute" << std::endl;
#endif
    (void)port;

    // check for nothing to do
    size_t n_meshes = input_data.size();
    if ((n_meshes < 1) || !input_data[0])
        return nullptr;

    // initialize the output array collections from the first input
    const_p_teca_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_FATAL_ERROR("Failed to average. dataset is not a teca_mesh")
        return nullptr;
    }

    // create output with the same type as the input
    p_teca_mesh out_mesh
        = std::dynamic_pointer_cast<teca_mesh>(input_data[0]->new_instance());

    p_teca_array_collection out_arrays = out_mesh->get_point_arrays();

    // apply the kernel convolution in the time dimension
    size_t n_arrays = in_mesh->get_point_arrays()->size();
    for (size_t j = 0; j < n_arrays; ++j)
    {
        size_t n_elem = 0;
        p_teca_variant_array out_array = nullptr;

        for (size_t i = 0; i < n_meshes; ++i)
        {
            in_mesh = std::dynamic_pointer_cast<const teca_mesh>(input_data[i]);

            const_p_teca_array_collection in_arrays = in_mesh->get_point_arrays();
            const_p_teca_variant_array in_array = in_arrays->get(j);

            VARIANT_ARRAY_DISPATCH(in_array.get(),

                // allocate and initialize the output
                if (i == 0)
                {
                    // allocate the output array
                    n_elem = in_array->size();
                    out_array = TT::New(n_elem, NT(0));

                    // store it in the output mesh
                    std::string array_name =
                        in_arrays->get_name(j) + this->variable_postfix;

                    out_mesh->get_point_arrays()->set(array_name, out_array);
                }

                // get the typed instances
                auto [sp_in, p_in] = get_host_accessible<CTT>(in_array);
                auto [p_out] = data<TT>(out_array);

                sync_host_access_any(in_array);

                // apply the kernel weight for this time step
                NT weight = NT(this->kernel_weights[i]);
                for (size_t q = 0; q < n_elem; ++q)
                {
                    p_out[q] += p_in[q] * weight;
                }
                )
        }
    }

    // get active time step
    std::string request_key;
    long out_index = 0;
    if (teca_metadata_util::get_requested_index(request,
        request_key, out_index))
    {
        TECA_FATAL_ERROR("Invalid execution control metadata."
            " Failed to get the requested index.")
        return nullptr;
    }

    // get the active mesh. the output should have temporal metadata from the
    // requested step. A translation needs to be applied to account for the
    // partial windows at the start and end of the series.
    int width = this->kernel_weights.size();
    long in_idx = 0;
    switch(this->stencil_type)
    {
        case backward:
            in_idx = width - 1;  // last
            break;
        case centered:
            in_idx = width / 2l; // middle
            break;
        case forward:
            in_idx = 0;          // first
            break;
    }

    auto active_mesh = std::dynamic_pointer_cast<teca_mesh>
        (std::const_pointer_cast<teca_dataset>(input_data[in_idx]));

    // copy the metadata and fix the time step
    out_mesh->copy_metadata(active_mesh);
    out_mesh->set_time_step(out_index);

    // copy information arrays
    out_mesh->get_information_arrays()->shallow_copy
        (active_mesh->get_information_arrays());

    // copy point arrays if there is a postfix
    if (!this->variable_postfix.empty())
        out_mesh->get_point_arrays()->shallow_append
            (active_mesh->get_point_arrays());

#ifdef TECA_DEBUG
    long in_index = 0;
    active_mesh->get_metadata().get("time_step", in_index);

    std::cerr << " === src : " << in_idx << " in : " << in_index
        << " out : " << out_index << std::endl;
#endif

    return out_mesh;
}
