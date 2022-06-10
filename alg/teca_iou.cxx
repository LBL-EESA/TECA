#include "teca_iou.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_table.h"
#include "teca_metadata.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using std::string;
using std::vector;
using std::cerr;
using std::endl;
using std::cos;

//#define TECA_DEBUG

namespace {

// compute IOU
template <typename num0_t, typename num1_t>
void iou(double *p_iou, const num0_t *infield0, const num1_t *infield1,
    double fill_val_0, double fill_val_1,
    unsigned long n_lon, unsigned long n_lat)
{
    double d_intersection = 0.0;
    double d_union = 0.0;
    double d_iou = 0.0;

    for (unsigned long j = 0; j < n_lat; ++j)
    {

        unsigned long jj = j*n_lon;
        num0_t *i0 = infield0 + jj;
        num1_t *i1 = infield1 + jj;

        for (unsigned long i = 0; i < n_lon; ++i)
        {
            // promote/cast the input arrays as double
            // so we can be sure that comparison with NaN
            // (the default fill value) will work.
            double i0val = (double) i0[i];
            double i1val = (double) i1[i];

            // only proceed if neither value is missing
            if (!(
                   ((i0val == fill_val_0) || std::isnan(i0val)) 
                || ((i1val == fill_val_1) || std::isnan(i1val))
                 ))
            {
                // intersection
                if ( (i0[i] > 0) && (i1[i] > 0) )
                    d_intersection++;

                // union
                if ( (i0[i] > 0) || (i1[i] > 0) )
                    d_union++;
            }
        }

        if (d_union == 0.0)
            // set the IOU field to nan if neither field had segmented values
            // in the valid (non-missing) range
            d_iou = std::nan("0");
        else
            d_iou = d_intersection/d_union;

        // set the return value
        *p_iou = d_iou;
    }

    return;
}
};

// --------------------------------------------------------------------------
teca_iou::teca_iou() :
    iou_field_0_variable(), iou_field_1_variable(),
    iou_variable("iou"), fill_val_0(std::nan("0")),
    fill_val_1(std::nan("0")
    )
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_iou::~teca_iou()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_iou::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_iou":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, iou_field_0_variable,
            "array containing one of the segemented fields to compare")
        TECA_POPTS_GET(std::string, prefix, iou_field_1_variable,
            "array containing another of the segemented fields to compare")
        TECA_POPTS_GET(std::string, prefix, iou_variable,
            "array to store the computed iou in")
        TECA_POPTS_GET(double, prefix, fill_val_0,
            "fill value for input field 0")
        TECA_POPTS_GET(double, prefix, fill_val_1,
            "fill value for input field 1")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_iou::set_properties(
    const string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::string, prefix, iou_field_0_variable)
    TECA_POPTS_SET(opts, std::string, prefix, iou_field_1_variable)
    TECA_POPTS_SET(opts, std::string, prefix, iou_variable)
    TECA_POPTS_SET(opts, double, prefix, fill_val_0)
    TECA_POPTS_SET(opts, double, prefix, fill_val_1)
}
#endif

// --------------------------------------------------------------------------
std::string teca_iou::get_iou_field_0_variable(
    const teca_metadata &request)
{
    std::string iou_field_0_var = this->iou_field_0_variable;

    if (iou_field_0_var.empty() &&
        request.has("teca_iou::iou_field_0_variable"))
            request.get("teca_iou::iou_field_0_variable", iou_field_0_var);

    return iou_field_0_var;
}

// --------------------------------------------------------------------------
std::string teca_iou::get_iou_field_1_variable(
    const teca_metadata &request)
{
    std::string iou_field_1_var = this->iou_field_1_variable;

    if (iou_field_1_var.empty() &&
        request.has("teca_iou::iou_field_1_variable"))
            request.get("teca_iou::iou_field_1_variable", iou_field_1_var);

    return iou_field_1_var;
}

// --------------------------------------------------------------------------
double teca_iou::get_fill_val_0(const teca_metadata &request)
{
    double fill_val_0_var = this->fill_val_0;

    // check if this hasn't been set explicitly but exists in the metadata
    if (std::isnan(fill_val_0_var) &&
        request.has("teca_iou::fill_val_0"))
            request.get("teca_iou::iou_fill_val_0", fill_val_0_var);

    // get request metadata
    std::vector<std::string> request_vars;
    request.get("variables", request_vars);

    teca_metadata request_atts;
    request.get("attributes", request_atts);

    // get the source array attributes
    teca_metadata request_array_atts;
    request.get(this->get_iou_field_0_variable(), request_array_atts);

    // check if this hasn't been set explicitly but is set in the
    // netCDF metadata
    int type_code = 0;
    if (std::isnan(fill_val_0_var) &&
        request_array_atts.has("_FillValue") &&
        !request_array_atts.get("type_code", type_code))
    {
        // make sure the _FillValue is specified with the matching precision
        // and don't add a _FillValue for integer arrays
        CODE_DISPATCH_FP(type_code,
            NT nc_fill_value = NT();

            // if it's present, promote/cast it as double
            if (request_array_atts.get("_FillValue", nc_fill_value))
                fill_val_0_var = (double) nc_fill_value;
        )
    }

    return fill_val_0_var;
}

// --------------------------------------------------------------------------
double teca_iou::get_fill_val_1(const teca_metadata &request)
{
    double fill_val_1_var = this->fill_val_1;

    // check if this hasn't been set explicitly but exists in the metadata
    if (std::isnan(fill_val_1_var) &&
        request.has("teca_iou::fill_val_1"))
            request.get("teca_iou::iou_fill_val_1", fill_val_1_var);

    // get request metadata
    std::vector<std::string> request_vars;
    request.get("variables", request_vars);

    teca_metadata request_atts;
    request.get("attributes", request_atts);

    // get the source array attributes
    teca_metadata request_array_atts;
    request.get(this->get_iou_field_1_variable(), request_array_atts);

    // check if this hasn't been set explicitly but is set in the
    // netCDF metadata
    int type_code = 0;
    if (std::isnan(fill_val_1_var) &&
        request_array_atts.has("_FillValue") &&
        !request_array_atts.get("type_code", type_code))
    {
        // make sure the _FillValue is specified with the matching precision
        // and don't add a _FillValue for integer arrays
        CODE_DISPATCH_FP(type_code,
            NT nc_fill_value = NT();

            // if it's present, promote/cast it as double
            if (request_array_atts.get("_FillValue", nc_fill_value))
                fill_val_1_var = (double) nc_fill_value;
        )
    }

    return fill_val_1_var;
}

// --------------------------------------------------------------------------
std::string teca_iou::get_iou_variable(
    const teca_metadata &request)
{
    std::string iou_var = this->iou_variable;

    if (iou_var.empty())
    {
        if (request.has("teca_iou::iou_variable"))
            request.get("teca_iou::iou_variable", iou_var);
        else
            iou_var = "iou";
    }

    return iou_var;
}

// --------------------------------------------------------------------------
teca_metadata teca_iou::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_iou::get_output_metadata" << endl;
#endif
    (void)port;

    // add in the array we will generate
    teca_metadata out_md(input_md[0]);
    out_md.append("variables", this->iou_variable);

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_iou::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;
    (void)input_md;

    vector<teca_metadata> up_reqs;

    // get the name of the arrays we need to request
    std::string iou_field_0_var = this->get_iou_field_0_variable(request);
    if (iou_field_0_var.empty())
    {
        TECA_FATAL_ERROR("iou_field_0 array was not specified")
        return up_reqs;
    }

    std::string iou_field_1_var = this->get_iou_field_1_variable(request);
    if (iou_field_1_var.empty())
    {
        TECA_FATAL_ERROR("iou_field_1 array was not specified")
        return up_reqs;
    }

    // copy the incoming request to preserve the downstream
    // requirements and add the arrays we need
    teca_metadata req(request);

    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);

    arrays.insert(this->iou_field_0_variable);
    arrays.insert(this->iou_field_1_variable);

    // capture the array we produce
    arrays.erase(this->get_iou_variable(request));

    // update the request
    req.set("arrays", arrays);

    // send it up
    up_reqs.push_back(req);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_iou::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_iou::execute" << endl;
#endif
    (void)port;

    // get the input mesh
    const_p_teca_cartesian_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_FATAL_ERROR("teca_cartesian_mesh is required")
        return nullptr;
    }

    // get component 0 array
    std::string iou_field_0_var = this->get_iou_field_0_variable(request);

    if (iou_field_0_var.empty())
    {
        TECA_FATAL_ERROR("iou_field_0_variable was not specified")
        return nullptr;
    }

    const_p_teca_variant_array iou_field_0
        = in_mesh->get_point_arrays()->get(iou_field_0_var);

    if (!iou_field_0)
    {
        TECA_FATAL_ERROR("requested array \"" << iou_field_0_var << "\" not present.")
        return nullptr;
    }

    // get component 1 array
    std::string iou_field_1_var = this->get_iou_field_1_variable(request);

    if (iou_field_1_var.empty())
    {
        TECA_FATAL_ERROR("iou_field_1_variable was not specified")
        return nullptr;
    }

    const_p_teca_variant_array iou_field_1
        = in_mesh->get_point_arrays()->get(iou_field_1_var);

    if (!iou_field_1)
    {
        TECA_FATAL_ERROR("requested array \"" << iou_field_1_var << "\" not present.")
        return nullptr;
    }

    // get the input coordinate arra3ddys
    const_p_teca_variant_array lon = in_mesh->get_x_coordinates();
    const_p_teca_variant_array lat = in_mesh->get_y_coordinates();

    if (!lon || !lat)
    {
        TECA_FATAL_ERROR("lat lon mesh cooridinates not present.")
        return nullptr;
    }

    // create a mutable instance of the input fields so we can
    // get the types
    p_teca_variant_array iou_field_0_type = iou_field_0->new_instance();
    p_teca_variant_array iou_field_1_type = iou_field_1->new_instance();

    // initialize IOU
    double iou_val = 0.0;

    // get the fill values for the respective arrays
    double fill_val_0 = this->get_fill_val_0();
    double fill_val_1 = this->get_fill_val_1();

    // compute iou
    NESTED_TEMPLATE_DISPATCH_FP(
        teca_variant_array_impl,
        iou_field_0_type.get(),0,

        NESTED_TEMPLATE_DISPATCH_FP(
            teca_variant_array_impl,
            iou_field_1_type.get(),1,

            auto sp_iou_field_0 = dynamic_cast<const TT0*>(iou_field_0.get())->get_cpu_accessible();
            const NT0 *p_iou_field_0 = sp_iou_field_0.get();

            auto sp_iou_field_1 = dynamic_cast<const TT1*>(iou_field_1.get())->get_cpu_accessible();
            const NT1 *p_iou_field_1 = sp_iou_field_1.get();

            ::iou(&iou_val,
                p_iou_field_0, p_iou_field_1, fill_val_0, fill_val_1,
                lon->size(), lat->size());
        )
    )

    // get time step
    unsigned long time_step;
    in_mesh->get_time_step(time_step);

    // get temporal offset of the current timestep
    double time_offset = 0.0;
    in_mesh->get_time(time_offset);

    // get offset units
    std::string time_units;
    in_mesh->get_time_units(time_units);

    // get offset calendar
    std::string calendar;
    in_mesh->get_calendar(calendar);

    // create the output table, pass everything through, and
    // add the iou table entry
    p_teca_table out_table = teca_table::New();
    out_table->set_calendar(calendar);
    out_table->set_time_units(time_units);

    // add a row to the table
    out_table->declare_columns("step", long(), "time", double());
    out_table << time_step << time_offset << iou_val;

    return out_table;
}
