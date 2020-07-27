#include "teca_vertical_integral.h"

#include "teca_cartesian_mesh.h"
#include "teca_table.h"
#include "teca_array_attributes.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"

#include <stddef.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif


// 1/gravity - s^2/m
template <typename num_t>
constexpr num_t neg_one_over_g() {return num_t(-1.0)/num_t(9.806); } 

/* 
 * Calculates the vertical integral of 'array'
 *
 * input:
 * ------
 *
 *   array      - a pointer to a 3D array of values
 *   
 *   nx, ny, nz - the sizes of the x, y, z dimensions
 *
 *   pressure_level - the a hybrid coordinate, or sigma if a sigma coordinate
 *                system
 *   
 *   array_int  - a pointer to an already-allocated 2D array
 *                that will hold the values of the integral of 
 *                `array`
 *
 * output:
 * -------
 *
 *   Values of `array_int` are filled in with the integral
 *   of `array`
 *
 * Assumptions:
 *
 *  - array has dimensions [z,y,x]
 *
 *  - z is ordered such that the bottom of the atmosphere is at z = 0
 *
 *  - all values are given on level centers
 *
 *  - the units of pressure_level are in [Pa]
 *
 *
 *  
/ 
*/
template <typename num_t>
void mass_integral_trapezoidal(const num_t * array,
                               unsigned long nx,
                               unsigned long ny,
                               unsigned long nz,
                               const num_t * pressure_level,
                               num_t * array_int) {

  // loop over both horizontal dimensions
  for (unsigned long i = 0; i < nx; i++){
    for (unsigned long j = 0; j < ny; j++){


      // Determine the current index the integrated array
      unsigned long n2d = j + ny * i;

      // set the current integral value to zero
      num_t tmp_int = 0.0;

      // loop over the vertical dimension
      for (unsigned long k = 0; k < (nz-1); k++){

        // Determine the current index in the 3D array
        unsigned long n3d = k + nz*(j + ny * i);

        // calculate the pressure differential for this layer
        num_t dp = pressure_level[k+1] - pressure_level[k];

        // calculate the value of the array between these two levels
        num_t array_bar = (array[n3d] + array[n3d + 1])/num_t(2);

        // calculate the density-weighted vertical differential
        num_t rho_times_dz = neg_one_over_g<num_t>() * dp;

        // add the integral contribution to the array
        tmp_int +=  array_bar * rho_times_dz;

      }
      
      // set the integral value in the array
      array_int[n2d] = tmp_int;

    }
  } 
}

int request_var(
    std::string mesh_var,
    std::string expected_var,
    std::set<std::string> * arrays)
{
    // check that both variables are specified
    if ( mesh_var.empty() ) {
      TECA_ERROR("" << expected_var << " not specified")
      return 1;
    }

    // insert the request into the list
    arrays->insert(mesh_var);

    return 0;
}

const_p_teca_variant_array get_mesh_variable(
    std::string mesh_var,
    std::string expected_var,
    const_p_teca_cartesian_mesh in_mesh)
{
                                             
      // check that both variables are specified
      if ( mesh_var.empty() ) {
        TECA_ERROR("" << expected_var << " not specified")
        return nullptr;
      }

      // get the variable
      const_p_teca_variant_array out_array
        = in_mesh->get_point_arrays()->get(mesh_var);
      if (!out_array) {
        TECA_ERROR("variable \"" << mesh_var << "\" is not in the input")
        return nullptr;
      }

      return out_array;
}

const_p_teca_variant_array get_info_variable(
    std::string info_var,
    std::string expected_var,
    const_p_teca_cartesian_mesh in_info)
{
                                             
      // check that both variables are specified
      if ( info_var.empty() ) {
        TECA_ERROR("" << expected_var << " not specified")
        return nullptr;
      }

      // get the variable
      const_p_teca_variant_array out_array
        = in_info->get_information_arrays()->get(info_var);
      if (!out_array) {
        TECA_ERROR("variable \"" << info_var << "\" is not in the input")
        return nullptr;
      }

      return out_array;
}


// --------------------------------------------------------------------------
teca_vertical_integral::teca_vertical_integral() :
    long_name("integrated_var"),
    units("unknown"),
    pressure_level_variable("plev"),
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);

}

// --------------------------------------------------------------------------
teca_vertical_integral::~teca_vertical_integral()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_vertical_integral::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    (void) prefix;

    options_description opts("Options for "
        + (prefix.empty()?"teca_vertical_integral":prefix));

    /*opts.add_options()
        TECA_POPTS_GET(std::vector<std::string>, prefix, dependent_variables,
            "list of arrays to compute statistics for")
        ;*/
    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, integration_variable,
            "name of the variable to integrate (\"\")")
        TECA_POPTS_GET(std::string, prefix, long_name,
            "long name of the output variable (\"\")")
        TECA_POPTS_GET(std::string, prefix, units,
            "units of the output variable(\"\")")
        TECA_POPTS_GET(std::string, prefix, pressure_level_variable,
            "name of a coordinate in the hybrid coordinate system(\"\")")
        TECA_POPTS_GET(std::string, prefix, output_variable_name,
            "name for the integrated, output variable (\"\")")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_vertical_integral::set_properties(
    const std::string &prefix, variables_map &opts)
{
    (void) prefix;
    (void) opts;

    TECA_POPTS_SET(opts, std::string, prefix, integration_variable)
    TECA_POPTS_SET(opts, std::string, prefix, long_name)
    TECA_POPTS_SET(opts, std::string, prefix, units)
    TECA_POPTS_SET(opts, std::string, prefix, pressure_level_variable)
    TECA_POPTS_SET(opts, std::string, prefix, output_variable_name)

}
#endif

teca_metadata teca_vertical_integral::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> & input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_vertical_integral::get_output_metadata" << std::endl;
#endif

    teca_metadata report_md(input_md[0]);

    double bounds[6] = {0.0};
    unsigned long whole_extent[6] = {0ul};
    unsigned long extent[6] = {0ul};

    // get the extents and bounds
    report_md.get("whole_extent", whole_extent, 6);
    report_md.get("extent", extent, 6);
    report_md.get("bounds", bounds, 6);

    // get the coordinates
    teca_metadata coords;
    report_md.get("coordinates", coords);

    // set a new z coordinate with no value (this will cause cf_writer to skip)
    //p_teca_variant_array new_z = teca_variant_array_impl<double>::New(1);
    p_teca_variant_array new_z = coords.get("z")->new_instance();
    new_z->resize(1);
    new_z->set(0, 0.0);
    coords.set("z", new_z);

    // determine dimension sizes
    unsigned long nx = whole_extent[1] - whole_extent[0] + 1;
    unsigned long ny = whole_extent[3] - whole_extent[2] + 1;
    unsigned long nz = whole_extent[5] - whole_extent[4] + 1;

    // check that the variable has a z dimension
    if (nz == 1)
    {
        TECA_ERROR("This calculation requires 3D data. The current dataset "
            "whole_extents are [" << whole_extent[0] << ", " << whole_extent[1] << ", "
            << whole_extent[2] << ", " << whole_extent[3] << ", " << whole_extent[4] << ", "
            << whole_extent[5] << "]")
        return report_md;
    }

    // force the output data to have no z dimension
    for (size_t n = 4; n < 6; n++){
      extent[n] = 0;
      whole_extent[n] = 0;
      bounds[n] = 0.0;
    }

    if (report_md.has("variables")){
        report_md.append("variables", this->output_variable_name);
    }
    else{
        report_md.set("variables", this->output_variable_name);
    }

    // add attributes to enable CF I/O
    teca_metadata atts;
    report_md.get("attributes", atts);
    teca_array_attributes output_atts(
        teca_variant_array_code<double>::get(),
        teca_array_attributes::point_centering,
        0, "unset", "unset",
        "unset");

    atts.set(this->output_variable_name, (teca_metadata)output_atts);

    report_md.set("attributes", atts);
    // write the updated bounds/extent/coordinates
    report_md.set("whole_extent", whole_extent);
    if (report_md.has("extent")) report_md.set("extent", extent);
    if (report_md.has("bounds")) report_md.set("bounds", bounds);
    report_md.set("coordinates", coords);

    return report_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata>
teca_vertical_integral::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_vertical_integral::get_upstream_request" << std::endl;
#endif
    (void)port;

    // create the output request
    std::vector<teca_metadata> up_reqs;

    // copy the incoming request
    teca_metadata req(request);

    // copy the input metadata
    teca_metadata md_in(input_md[0]);

    // create a list of requested arrays
    std::set<std::string> arrays;
    // pre-populate with existing requests, if available
    if (req.has("arrays"))
        req.get("arrays", arrays);

    // get the pressure coordinate
    if (request_var(this->pressure_level_variable,
                                   std::string("pressure_level_variable"),
                                   &arrays) != 0 ) return up_reqs;

    // get the input array
    if (request_var(this->integration_variable,
                        std::string("integration_variable"),
                        &arrays) != 0 ) return up_reqs;

    // intercept request for our output
    arrays.erase(this->output_variable_name);

    /*
    double bounds[6] = {0.0};
    unsigned long extent[6] = {0ul};

    if (md_in.has("bounds")){
      // pass through bounds if bounds are given
      md_in.get("bounds", bounds, 6);
      req.set("bounds", bounds);
    }
    else if (md_in.has("extent")){
      // pass through extent if extent is given
      md_in.get("extent", extent, 6);
      req.set("extent", extent);
    }
    else {
      // pass the whole extent as the extent if neither
      // bounds nor extent are given
      md_in.get("whole_extent", extent, 6);
      req.set("extent", extent);
    }
    */


    // TODO: this overrides the above code and removes bounds/extent;
    // this disables the ability for a user to specify bounds/extent
    // The above (commented) code needs to be debugged in order for 
    // the bounds/extent pass-through to work properly though.
    req.remove("bounds");
    req.remove("extent");
    req.remove("whole_extent");

    // overwrite the existing request with the augmented one
    req.set("arrays", arrays);
    // put the request into the outgoing metadata
    up_reqs.push_back(req);

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_vertical_integral::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id() << "teca_vertical_integral::execute" << std::endl;
#endif
    (void)port;
    (void)request;

    // get the input mesh
    const_p_teca_cartesian_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_ERROR("dataset is not a teca_cartesian_mesh")
        return nullptr;
    }

    // get mesh dimension
    unsigned long extent[6];
    in_mesh->get_extent(extent);

    // set the dimension sizes
    unsigned long nx = extent[1] - extent[0] + 1;
    unsigned long ny = extent[3] - extent[2] + 1;
    unsigned long nz = extent[5] - extent[4] + 1;

#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id() << 
      "teca_vertical_integral::execute working on array of size [" <<
      nx << ", " << ny << ", " << nz << "]" << std::endl;
#endif

    const_p_teca_variant_array pressure_level = nullptr;
    
    // get the pressure coordinate
    pressure_level = get_info_variable(this->pressure_level_variable,
                                   std::string("pressure_level_variable"),
                                   in_mesh);
    if (!pressure_level) return nullptr;

#ifdef TECA_DEBUG
      std::cerr << teca_parallel_id() << "teca_vertical_integral::execute:" 
        << "reading the input variable" << std::endl;
#endif
    // get the input array
    const_p_teca_variant_array input_array = 
      get_mesh_variable(this->integration_variable,
                        std::string("integration_variable"),
                        in_mesh);
    if (!input_array) return nullptr;

#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id() << "teca_vertical_integral::execute:" 
      << "allocating the integration array" << std::endl;
#endif

    // allocate the output array
    p_teca_variant_array integrated_array = input_array->new_instance();
    integrated_array->resize(nx*ny);


    NESTED_TEMPLATE_DISPATCH_FP(
        teca_variant_array_impl,
        integrated_array.get(),
        _INARR,

        const NT_INARR * p_input_array 
          = dynamic_cast<const TT_INARR*>(input_array.get())->get();

        const NT_INARR * p_pressure_level 
          = dynamic_cast<const TT_INARR*>(pressure_level.get())->get();

        NT_INARR * p_integrated_array 
          = dynamic_cast<TT_INARR*>(integrated_array.get())->get();

        // call the vertical integration routine
        mass_integral_trapezoidal(p_input_array, nx, ny, nz, p_pressure_level, 
          p_integrated_array);
        )

#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id() << "teca_vertical_integral::execute:" 
      << "creating the output mesh" << std::endl;
#endif


    // create the output mesh, pass everything but the integration variable, 
    // and add the integrated array
    p_teca_cartesian_mesh out_mesh = teca_cartesian_mesh::New();

    out_mesh->shallow_copy(
        std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));

    // set mesh dimensions; use a scalar Z dimension
    unsigned long out_extent[6];
    unsigned long out_whole_extent[6];
    double out_bounds[6];
    out_mesh->get_extent(out_extent);
    out_mesh->get_whole_extent(out_whole_extent);
    out_mesh->get_bounds(out_bounds);

    for (size_t n = 4; n < 6; n++){
      out_extent[n] = 0;
      out_whole_extent[n] = 0;
      out_bounds[n] = 0;
    }

    out_mesh->set_extent(out_extent);
    out_mesh->set_whole_extent(out_whole_extent);
    out_mesh->set_bounds(out_bounds);

    // set the z coordinate
    p_teca_variant_array zo = in_mesh->get_z_coordinates()->new_instance();
    zo->resize(1);
    out_mesh->set_z_coordinates("z", zo);

    //  add the output variable to the mesh
    std::cerr << "Output variable name: " << this->output_variable_name << std::endl;
    out_mesh->get_point_arrays()->append(
        this->output_variable_name, integrated_array);

    return out_mesh;
}
