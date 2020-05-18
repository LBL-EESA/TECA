#include "teca_vertical_integral.h"

#include "teca_cartesian_mesh.h"
#include "teca_table.h"
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

using std::cerr;
using std::endl;

// 1/gravity - s^2/m
template <typename num_t>
constexpr num_t neg_one_over_g() {return num_t(-1.0)/num_t(9.81); } 

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
 *   csystem    - an integer indicating the vertical coordinate system
 *                type of `array`:
 *                0 = sigma coordinate system
 *                1 = hybrid coordinate system
 *
 *   a_or_sigma - the a hybrid coordinate, or sigma if a sigma coordinate
 *                system
 *   
 *   b          - the hybrid coordinates (see below), or a null pointer if the
 *                coordinate system is sigma
 *
 *   ps         - a pointer to a 2D array of pressure values
 *
 *   p_top         - a scalar value giving the model top pressure
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
 *  - ps has dimensions [y,x]
 *
 *  - z is either a hybrid coordinate system where pressure
 *    is defined as p = a * p_top + b * ps, or it is a sigma
 *    coordinate system where p is defined as p = (ps - p_top)*sigma + p_top
 *
 *  - z is ordered such that the bottom of the atmosphere is at z = 0
 *
 *  - the values of a_or_sigma, and b are given on level interfaces and
 *    the values of `array` are given on level centers, such that
 *    dp (the pressure differential) will also be on level centers
 *
 *  - a_or_sigma and b have shape [nz + 1]
 *
 *  - the units of ps and p_top are in Pa
 *
 *
 *  
/ 
*/
template <typename size_t, typename num_t>
void vertical_integral(num_t * array,
                       size_t nx,
                       size_t ny,
                       size_t nz,
                       size_t csystem,
                       num_t * a_or_sigma,
                       num_t * b,
                       num_t * ps,
                       num_t p_top,
                       num_t * array_int) {


  // loop over both horizontal dimensions
  for (size_t i = 0; i < nx; i++){
    for (size_t j = 0; j < ny; j++){

      // Determine the current index in ps
      size_t n2d = j + ny * i;

      // set the current integral value to zero
      num_t tmp_int = 0.0;

      // loop over the vertical dimension
      for (size_t k = 0; k < nz; k++){

        // Determine the current index in the 3D array
        size_t n3d = k + nz*(j + ny * i);

        num_t dp = num_t(0.0);

        // calculate the pressure differential for the sigma
        // coordinate system
        if (csystem == size_t(1)){
          // calculate da and db
          num_t da = a_or_sigma[k+1] - a_or_sigma[k];
          num_t db = b[k+1] - b[k];

          // calculate dp
          dp = p_top * da + ps[n2d] * db;
        }
        // calculate the pressure differential for the sigma
        // coordinate system
        else{
          num_t dsigma = a_or_sigma[k+1] - a_or_sigma[k];
          dp = (ps[n2d] - p_top)*dsigma;
        }
    
        // accumulate the integral
        tmp_int += neg_one_over_g<num_t>() * array[n3d] * dp;

      }
      
      // set the integral value in the array
      array_int[n2d] = tmp_int;

    }
  } 
}


const_p_teca_variant_array get_mesh_variable(
    std::string mesh_var,
    std::string expected_var,
    const_p_teca_cartesian_mesh in_mesh)
{
                                             
      // check that both variables are specified
      if ( mesh_var.empty() )
      {
        TECA_ERROR( expected_var << " not specified")
        return nullptr;
      }

      // get the variable
      const_p_teca_variant_array out_array
        = in_mesh->get_point_arrays()->get(mesh_var);
      if (!out_array)
      {
        TECA_ERROR("variable \"" << mesh_var << "\" is not in the input")
        return nullptr;
      }

      return out_array;
}


// --------------------------------------------------------------------------
teca_vertical_integral::teca_vertical_integral()
    // TODO: add variables for long_name, units, other metadata?
    // these needs to be passed in
    hybrid_a_variable("a_bnds"),
    hybrid_b_variable("b_bnds"),
    sigma_variable("sigma_bnds"),
    surface_p_variable("ps"),
    p_top_variable("ptop"),
    using_hybrid(1),
    p_top_override_value(float(-1.0))
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
        TECA_POPTS_GET(std::string, prefix, hybrid_a_variable,
            "name of a coordinate in the hybrid coordinate system(\"\")")
        TECA_POPTS_GET(std::string, prefix, hybrid_b_variable,
            "name of b coordinate in the hybrid coordinate system(\"\")")
        TECA_POPTS_GET(std::string, prefix, sigma_variable,
            "name of sigma coordinate (\"\")")
        TECA_POPTS_GET(std::string, prefix, surface_p_variable,
            "name of the surface pressure variable (\"\")")
        TECA_POPTS_GET(std::string, prefix, p_top_variable,
            "name of the model top variable (\"\")")
        TECA_POPTS_GET(std::string, prefix, output_variable_name,
            "name for the integrated, output variable (\"\")")
        TECA_POPTS_GET(int, prefix, using_hybrid,
            "flags whether the vertical coordinate is hybrid (1) or sigma (0) 
            (\"\")")
        TECA_POPTS_GET(float, prefix, p_top_override_value,
            "name of the model top variable(\"\")")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_vertical_integral::set_properties(
    const std::string &prefix, variables_map &opts)
{
    (void) prefix;
    (void) opts;

    TECA_POPTS_SET(opts, std::string, prefix, hybrid_a_variable)
    TECA_POPTS_SET(opts, std::string, prefix, hybrid_b_variable)
    TECA_POPTS_SET(opts, std::string, prefix, sigma_variable)
    TECA_POPTS_SET(opts, std::string, prefix, surface_p_variable)
    TECA_POPTS_SET(opts, std::string, prefix, p_top_variable)
    TECA_POPTS_SET(opts, int, prefix, using_hybrid)
    TECA_POPTS_SET(opts, float, prefix, p_top_override_value)
    TECA_POPTS_SET(opts, std::string, prefix, output_variable_name)

}
#endif

// --------------------------------------------------------------------------
std::vector<teca_metadata>
teca_vertical_integral::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_vertical_integral::get_upstream_request" << endl;
#endif
    (void)port;
    (void)input_md;

    std::vector<teca_metadata> up_reqs(1, request);
    return up_reqs;
}


// --------------------------------------------------------------------------
const_p_teca_dataset teca_vertical_integral::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_vertical_integral::execute" << endl;
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
    out_mesh->get_extent(extent);

    // set the dimension sizes
    unsigned long nx = extent[1] - extent[0] + 1;
    unsigned long ny = extent[3] - extent[2] + 1;
    unsigned long nz = extent[5] - extent[4] + 1;
    if (nz == 1)
    {
        TECA_ERROR("This calculation requires 3D data. The current dataset "
            "extents are [" << extent[0] << ", " << extent[1] << ", "
            << extent[2] << ", " << extent[3] << ", " << extent[4] << ", "
            << extent[5] << "]")
        return nullptr;
    }

    int using_hybrid = this->using_hybrid;

    const_p_teca_variant_array a_or_sigma = nullptr;
    const_p_teca_variant_array b_i = nullptr;
    
    if (using_hybrid)
    {
      // get the a coordinate
      a_or_sigma = get_mesh_variable(this->hybrid_a_variable,
                                     "hybrid_a_variable",
                                     in_mesh);
      if (!a_or_sigma) return nullptr;

      // get the b coordinate
      b_i = get_mesh_variable(this->hybrid_b_variable,
                              "hybrid_b_variable",
                              in_mesh);
      if (!b_i) return nullptr;
      
    else
    {
      // get the sigma coordinate
      a_or_sigma = get_mesh_variable(this->sigma_variable,
                                     "sigma_variable",
                                     in_mesh);
      if (!a_or_sigma) return nullptr;
    }

    // get the surface pressure
    const_p_teca_variant_array p_s = 
      get_mesh_variable(this->surface_p_variable,
                        "surface_p_variable",
                        in_mesh);
    if (!p_s) return nullptr;

    // get the model top pressure
    // TODO: p_top is a scalar: is const_p_teca_variant array correct?
    // probably need to put this in information array
    const_p_teca_variant_array p_top_array = 
      get_mesh_variable(this->p_top_variable,
                        "p_top_variable",
                        in_mesh);
    if (!p_top_array) return nullptr;

    // get the input array
    const_p_teca_variant_array input_array = 
      get_mesh_variable(this->integration_variable,
                        "integration_variable",
                        in_mesh);
    if (!input_array) return nullptr;


    // allocate the output array
    p_teca_variant_array integrated_array = input_array->new_instance();
    // TODO: figure out how to properly resize this to nz = 1
    integrated_array->resize(nx*ny);


    NESTED_TEMPLATE_DISPATCH_FP(
        const teca_variant_array_impl,
        input_array.get(),
        _INARR,

        const NT_INARR * p_input_array 
          = dynamic_cast<TT_INARR*>(input_array->get())->get();

        // call the vertical integration routine
        // TODO: do some templating magic on the rest of the variables
        // appending p_ to the front of each variable name
        vertical_integral(p_input_array, nx, ny, nz,
                          (size_t) using_hybrid,
                          p_a_or_sigma, p_b_i, p_p_s, p_p_top,
                          p_integrated_array);
        )


    // create the output mesh, pass everything but the integration variable, 
    // and add the integrated array
    p_teca_cartesian_mesh out_mesh = teca_cartesian_mesh::New();

    // TODO: figure out how to drop this->integration_variable
    // TODO: figure out how to drop the dimensionality of the data if needed
    // this should work -- just need to modify the dimensions 
    //  change extent, whole_extent, bounds, coordinates (set z to a 1-valued
    //  teca_variant_array [maybe pressure, maybe 0])
    //  see data/teca_cartesian_mesh.h
    //    e.g., set_z_coordinates()
    //    set new whole_extent, extent, bounds arrays in new teca_metadata
    //    objects.
    out_mesh->shallow_copy(
        std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));

    return out_mesh;
}
