#include "teca_config.h"
#include "teca_table.h"
#include "teca_metadata.h"
#include "teca_cf_reader.h"
#include "teca_multi_cf_reader.h"
#include "teca_table_writer.h"
#include "teca_index_executive.h"
#include "teca_normalize_coordinates.h"
#include "teca_integrated_vapor_transport.h"
#include "teca_valid_value_mask.h"
#include "teca_unpack_data.h"
#include "teca_cartesian_mesh_source.h"
#include "teca_cartesian_mesh_regrid.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_regional_moisture_flux.h"
#include "teca_surface_integral.h"
#include "teca_indexed_dataset_cache.h"
#include "teca_mpi_manager.h"
#include "teca_coordinate_util.h"
#include "teca_dataset_source.h"
#include "teca_table_reduce.h"
#include "teca_table_sort.h"
#include "teca_table_calendar.h"
#include "teca_table_join.h"
#include "teca_app_util.h"
#include "teca_calcalcs.h"
#include "teca_shape_file_mask.h"

#include <vector>
#include <string>
#include <iostream>

#include <boost/program_options.hpp>


using namespace std;

using boost::program_options::value;

// --------------------------------------------------------------------------
int main(int argc, char **argv)
{
    // initialize mpi
    teca_mpi_manager mpi_man(argc, argv);

    // initialize command line options description
    // set up some common options to simplify use for most
    // common scenarios
    int help_width = 100;
    options_description basic_opt_defs(
        "Basic usage:\n\n"
        "The following options are the most commonly used. Information\n"
        "on advanced options can be displayed using --advanced_help\n\n"
        "Basic command line options", help_width, help_width - 4
        );
    basic_opt_defs.add_options()
        ("input_file", value<std::string>(), "\na teca_multi_cf_reader configuration file"
            " identifying the set of NetCDF CF2 files to process. When present data is"
            " read using the teca_multi_cf_reader. Use one of either --input_file or"
            " --input_regex.\n")

        ("input_regex", value<std::string>(), "\na teca_cf_reader regex identifying the"
            " set of NetCDF CF2 files to process. When present data is read using the"
            " teca_cf_reader. Use one of either --input_file or --input_regex.\n")

        ("region_name", value<std::string>()->default_value("region"),
            "\nname of the region used in output variable names.\n")

        ("specific_humidity", value<std::string>()->default_value("hus"),
            "\nname of variable with the 3D specific humidity field.\n")

        ("wind_u", value<std::string>()->default_value("ua"),
            "\nname of variable with the 3D longitudinal component of the wind vector.\n")
        ("wind_v", value<std::string>()->default_value("va"),
            "\nname of variable with the 3D latitudinal component of the wind vector.\n")

        ("ivt_u", value<std::string>()->default_value("ivt_u"),
            "\nname to use for the longitudinal component of the integrated vapor transport vector.\n")
        ("ivt_v", value<std::string>()->default_value("ivt_v"),
            "\nname to use for the latitudinal component of the integrated vapor transport vector.\n")

        ("ivt", value<std::string>()->default_value("ivt"),
            "\nname of variable with the magnitude of integrated vapor transport (IVT)\n")

        ("compute_ivt", "\nwhen this is set the IVT vector is calculated from wind and"
            " specific humididty inputs.  --wind_u, --wind_v, and --specific_humidty are"
            " used to specify the inputs. --ivt_u  and --ivt_v are used to specify the outputs.\n")

        ("heat_flux", value<std::string>()->default_value("hfls"),
            "\nname of variable containing surface upward latent heat flux.\n")

        ("precip_flux", value<std::string>()->default_value("pr"),
            "\nname of variable containing the surface precipittion flux.\n")

        ("compute_iwv", "\nwhen this is set the column integrated water vapor is calculated from"
            " specific humididty. --specific_humidty is used to specify the input. --iwv is used"
            " to specify the output.\n")

        ("iwv", value<std::string>()->default_value("iwv"),
            "\nname of variable containing the integrated water vapor.\n")

        ("shape_file", value<std::string>(), "\nAn ESRI shape file identifying the region"
            " to compute moisture flux over\n")

        ("bounds", value<std::vector<double>>()->multitoken(),
            "\nA four-tuple of low and high values specifying lon lat bounding box to subset"
            " the input dataset with. The accepted format for bounds is: x0 x1 y0 y1 0 0\n")

        ("output_file", value<std::string>()->default_value("regional_moisture_flux.nc"),
            "\nA path and file name pattern for the output files.\n")

        ("x_axis_variable", value<std::string>()->default_value("lon"),
            "\nname of x coordinate variable\n")
        ("y_axis_variable", value<std::string>()->default_value("lat"),
            "\nname of y coordinate variable\n")
        ("z_axis_variable", value<std::string>()->default_value("plev"),
            "\nname of z coordinate variable\n")

        ("mesh_height", value<std::string>()->default_value("Zg"),
            "\nSets the name of the variable containing the point wise vertical height"
            " in meters above mean sea level\n")

        ("first_step", value<long>()->default_value(0), "\nfirst time step to process\n")
        ("last_step", value<long>()->default_value(-1), "\nlast time step to process\n")

        ("start_date", value<std::string>(), "\nThe first time to process in 'Y-M-D h:m:s'"
            " format. Note: There must be a space between the date and time specification\n")
        ("end_date", value<std::string>(), "\nThe last time to process in 'Y-M-D h:m:s' format\n")

        ("n_threads", value<int>()->default_value(-1), "\nSets the thread pool size on each MPI"
            "  rank. When the default value of -1 is used TECA will coordinate the thread pools"
            " across ranks such each thread is bound to a unique physical core.\n")

        ("verbose", "\nenable extra terminal output\n")

        ("help", "\ndisplays documentation for application specific command line options\n")
        ("advanced_help", "\ndisplays documentation for algorithm specific command line options\n")
        ("full_help", "\ndisplays both basic and advanced documentation together\n")
        ;

    // add all options from each pipeline stage for more advanced use
    options_description advanced_opt_defs(
        "Advanced usage:\n\n"
        "The following list contains the full set options giving one full\n"
        "control over all runtime modifiable parameters. The basic options\n"
        "(see" "--help) map to these, and will override them if both are\n"
        "specified.\n\n"
        "regional moisture flux pipeline:\n\n"
        "    (cf / mcf_reader)      (cache)--(shape file mask)--(mesh source)\n"
        "           \\                  \\\n"
        "        (vv_mask)--(unpack)--(regrid) \n"
        "                                 \\\n"
        "                 (moisture flux)--+--(surface flux)\n"
        "                        \\                  \\\n"
        "                         +---(table join)---+\n"
        "                                    \\\n"
        "                               (table reduce)--(table sort)--(calendar)--(writer)\n\n"
        "Advanced command line options", help_width, help_width - 4
        );

    // create the pipeline stages here, they contain the
    // documentation and parse command line.
    // objects report all of their properties directly
    // set default options here so that command line options override
    // them. while we are at it connect the pipeline
    p_teca_cf_reader cf_reader = teca_cf_reader::New();
    cf_reader->get_properties_description("cf_reader", advanced_opt_defs);

    p_teca_multi_cf_reader mcf_reader = teca_multi_cf_reader::New();
    mcf_reader->get_properties_description("mcf_reader", advanced_opt_defs);

    p_teca_normalize_coordinates norm_coords = teca_normalize_coordinates::New();
    norm_coords->get_properties_description("norm_coords", advanced_opt_defs);

    p_teca_valid_value_mask vv_mask = teca_valid_value_mask::New();
    vv_mask->get_properties_description("vv_mask", advanced_opt_defs);

    p_teca_unpack_data unpack = teca_unpack_data::New();
    unpack->get_properties_description("unpack", advanced_opt_defs);

    /*p_teca_cf_reader mask_reader = teca_cf_reader::New();
    mask_reader->get_properties_description("mask_reader", advanced_opt_defs);
    mask_reader->set_t_axis_variable("");

    p_teca_normalize_coordinates mask_coords = teca_normalize_coordinates::New();
    mask_coords->get_properties_description("mask_coords", advanced_opt_defs);
    mask_coords->set_enable_periodic_shift_x(1);

    p_teca_valid_value_mask mask_vvm = teca_valid_value_mask::New();
    mask_vvm->get_properties_description("mask_vvm", advanced_opt_defs);*/

    p_teca_cartesian_mesh_source mask_mesh = teca_cartesian_mesh_source::New();
    mask_mesh->get_properties_description("mask_mesh", advanced_opt_defs);

    p_teca_cartesian_mesh_regrid mask_regrid = teca_cartesian_mesh_regrid::New();
    mask_regrid->get_properties_description("mask_regrid", advanced_opt_defs);

    p_teca_indexed_dataset_cache mask_cache = teca_indexed_dataset_cache::New();
    mask_cache->get_properties_description("mask_cache", advanced_opt_defs);
    mask_cache->set_override_request_index(1);
    mask_cache->set_max_cache_size(1);

    p_teca_integrated_vapor_transport ivt_int = teca_integrated_vapor_transport::New();
    ivt_int->get_properties_description("ivt_integral", advanced_opt_defs);
    ivt_int->set_specific_humidity_variable("hus");
    ivt_int->set_wind_u_variable("ua");
    ivt_int->set_wind_v_variable("va");
    ivt_int->set_ivt_u_variable("ivt_u");
    ivt_int->set_ivt_v_variable("ivt_v");

    p_teca_shape_file_mask mask = teca_shape_file_mask::New();
    mask->get_properties_description("region_mask", advanced_opt_defs);
    mask->set_mask_variable("region_mask");

    p_teca_regional_moisture_flux moisture_flux = teca_regional_moisture_flux::New();
    moisture_flux->get_properties_description("moisture_flux", advanced_opt_defs);

    p_teca_surface_integral surface_flux = teca_surface_integral::New();
    surface_flux->get_properties_description("surface_flux", advanced_opt_defs);

    p_teca_table_join join = teca_table_join::New();
    join->get_properties_description("join", advanced_opt_defs);
    join->set_number_of_input_connections(2);

    p_teca_table_reduce map_reduce = teca_table_reduce::New();
    map_reduce->get_properties_description("map_reduce", advanced_opt_defs);

    p_teca_table_sort sort = teca_table_sort::New();
    sort->set_index_column("time_step");
    sort->get_properties_description("sort", advanced_opt_defs);

    p_teca_table_calendar calendar = teca_table_calendar::New();
    calendar->get_properties_description("calendar", advanced_opt_defs);

    p_teca_table_writer writer = teca_table_writer::New();
    writer->set_output_format_auto();
    writer->get_properties_description("writer", advanced_opt_defs);


    // package basic and advanced options for display
    options_description all_opt_defs(help_width, help_width - 4);
    all_opt_defs.add(basic_opt_defs).add(advanced_opt_defs);

    // parse the command line
    int ierr = 0;
    variables_map opt_vals;
    if ((ierr = teca_app_util::process_command_line_help(
        mpi_man.get_comm_rank(), argc, argv, basic_opt_defs,
        advanced_opt_defs, all_opt_defs, opt_vals)))
    {
        if (ierr == 1)
            return 0;
        return -1;
    }

    // pass command line arguments into the pipeline objects
    // advanced options are processed first, so that the basic
    // options will override them
    cf_reader->set_properties("cf_reader", opt_vals);
    mcf_reader->set_properties("mcf_reader", opt_vals);
    norm_coords->set_properties("norm_coords", opt_vals);
    vv_mask->set_properties("vv_mask", opt_vals);
    unpack->set_properties("unpack", opt_vals);
    ivt_int->set_properties("ivt_integral", opt_vals);
    mask->set_properties("region_mask", opt_vals);

    /*mask_reader->set_properties("mask_reader", opt_vals);
    mask_coords->set_properties("mask_coords", opt_vals);
    mask_vvm->set_properties("mask_vvm", opt_vals);*/
    mask_mesh->set_properties("mask_mesh", opt_vals);
    mask_regrid->set_properties("mask_regrid", opt_vals);
    mask_cache->set_properties("mask_cache", opt_vals);
    moisture_flux->set_properties("moisture_flux", opt_vals);
    surface_flux->set_properties("surface_flux", opt_vals);
    map_reduce->set_properties("map_reduce", opt_vals);
    sort->set_properties("sort", opt_vals);
    calendar->set_properties("calendar", opt_vals);
    writer->set_properties("writer", opt_vals);

    // now pass in the basic options, these are processed
    // last so that they will take precedence
    // configure the pipeline from the command line options.
    p_teca_algorithm reader;

    // some minimal check for missing options
    bool have_file = opt_vals.count("input_file");
    bool have_regex = opt_vals.count("input_regex");
    if ((have_file && have_regex) || !(have_file || have_regex))
    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_FATAL_ERROR("Extacly one of --input_file or --input_regex can be specified. "
                "Use --input_file to activate the multi_cf_reader (HighResMIP datasets) "
                "and --input_regex to activate the cf_reader (CAM like datasets)")
        }
        return -1;
    }

    if (!opt_vals.count("shape_file"))
    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_FATAL_ERROR("The region mask file was not specified. Use"
                " --shape_file to specify the path the ESRI shape file defining"
                " the region to compute moisture and surface fluxes over")
        }
        return -1;
    }

    if (!opt_vals.count("output_file"))
    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_FATAL_ERROR("The --output_file was not specified.")
        }
        return -1;
    }

    // set the verbosity level
    if (opt_vals.count("verbose"))
    {
        cf_reader->set_verbose(1);
        mcf_reader->set_verbose(1);
        norm_coords->set_verbose(1);
        unpack->set_verbose(1);
        vv_mask->set_verbose(1);
        ivt_int->set_verbose(1);
        mask_mesh->set_verbose(1);
        mask->set_verbose(1);
        mask_cache->set_verbose(1);
        mask_regrid->set_verbose(1);
        moisture_flux->set_verbose(1);
        surface_flux->set_verbose(1);
        map_reduce->set_verbose(1);
        sort->set_verbose(1);
        calendar->set_verbose(1);
        writer->set_verbose(1);
    }


    // configure the reader
    if (opt_vals.count("input_file"))
    {
        mcf_reader->set_input_file(opt_vals["input_file"].as<string>());
        reader = mcf_reader;
    }
    else if (opt_vals.count("input_regex"))
    {
        cf_reader->set_files_regex(opt_vals["input_regex"].as<string>());
        reader = cf_reader;
    }
    p_teca_algorithm head = reader;

    if (!opt_vals["x_axis_variable"].defaulted())
    {
        cf_reader->set_x_axis_variable(opt_vals["x_axis_variable"].as<string>());
        mcf_reader->set_x_axis_variable(opt_vals["x_axis_variable"].as<string>());
    }

    if (!opt_vals["y_axis_variable"].defaulted())
    {
        cf_reader->set_y_axis_variable(opt_vals["y_axis_variable"].as<string>());
        mcf_reader->set_y_axis_variable(opt_vals["y_axis_variable"].as<string>());
    }

    // add the valid value mask stage
    norm_coords->set_input_connection(head->get_output_port());
    vv_mask->set_input_connection(norm_coords->get_output_port());
    unpack->set_input_connection(vv_mask->get_output_port());
    head = unpack;

    if (opt_vals.count("compute_ivt"))
    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_STATUS("Computing IVT")
        }

        std::string z_var = "plev";
        if (!opt_vals["z_axis_variable"].defaulted())
            z_var = opt_vals["z_axis_variable"].as<string>();

        cf_reader->set_z_axis_variable(z_var);
        mcf_reader->set_z_axis_variable(z_var);

        ivt_int->set_input_connection(head->get_output_port());
        ivt_int->set_wind_u_variable(opt_vals["wind_u"].as<string>());
        ivt_int->set_wind_v_variable(opt_vals["wind_v"].as<string>());
        ivt_int->set_specific_humidity_variable(opt_vals["specific_humidity"].as<string>());
        ivt_int->set_ivt_u_variable(opt_vals["ivt_u"].as<string>());
        ivt_int->set_ivt_v_variable(opt_vals["ivt_v"].as<string>());

        head = ivt_int;
    }
/*

    // configure the reader
    if (!opt_vals.count("mask_regex"))
    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_FATAL_ERROR("The region mask file was not specified. See --mask_regex")
        }
        return -1;
    }

    if (!opt_vals.count("mask"))
    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_FATAL_ERROR("The region mask variable was not specified. See --mask")
        }
        return -1;
    }



    mask_reader->set_files_regex(opt_vals["mask_regex"].as<string>());

    mask_coords->set_input_connection(mask_reader->get_output_port());

    mask_vvm->set_input_connection(mask_coords->get_output_port());
*/
    // region mask goes in a separate branch of the pipeline so that it can be cached
    teca_metadata md = head->update_metadata();

    mask_mesh->set_spatial_bounds(md, false);
    mask_mesh->set_spatial_extents(md, false);
    mask_mesh->set_x_axis_variable(md);
    mask_mesh->set_y_axis_variable(md);
    //mask_mesh->set_z_axis_variable(md);
    mask_mesh->set_t_axis_variable(md);
    mask_mesh->set_t_axis(md);

    // shape file mask
    std::string region_name = opt_vals["region_name"].as<string>();

    mask->set_input_connection(mask_mesh->get_output_port());
    mask->set_shape_file(opt_vals["shape_file"].as<string>());
    mask->set_mask_variable(region_name + "_mask");

    mask_cache->set_input_connection(mask->get_output_port());

    mask_regrid->set_input_connection(0, head->get_output_port());
    mask_regrid->set_input_connection(1, mask_cache->get_output_port());

    /*p_teca_cartesian_mesh_writer rdw = teca_cartesian_mesh_writer::New();
    rdw->set_input_connection(mask_regrid->get_output_port());
    rdw->set_file_name("regrid_region_mask_%t%.vtk");*/


    std::vector<std::string> out_arrays;

    // moisture flux calc
    moisture_flux->set_input_connection(mask_regrid->get_output_port());
    moisture_flux->set_ivt_u_variable(opt_vals["ivt_u"].as<string>());
    moisture_flux->set_ivt_v_variable(opt_vals["ivt_v"].as<string>());
    moisture_flux->set_region_mask_variable(region_name + "_mask");

    std::string moisture_flux_var = region_name + "_moisture_flux";

    moisture_flux->set_moisture_flux_variable(moisture_flux_var);

    out_arrays.push_back(std::move(moisture_flux_var));

    head = moisture_flux;

    bool heat_flux = !opt_vals["heat_flux"].defaulted();
    bool precip_flux = !opt_vals["precip_flux"].defaulted();
    bool iwv = !opt_vals["iwv"].defaulted();
    if (heat_flux || precip_flux || iwv)
    {
        surface_flux->set_input_connection(mask_regrid->get_output_port());
        surface_flux->set_region_mask_variable(region_name + "_mask");

        if (heat_flux)
        {
            std::string in_var = opt_vals["heat_flux"].as<string>();
            std::string out_var = region_name + "_" + in_var;

            surface_flux->append_input_variable(in_var);
            surface_flux->append_output_variable(out_var);
            out_arrays.push_back(std::move(out_var));
        }

        if (precip_flux)
        {
            std::string in_var = opt_vals["precip_flux"].as<string>();
            std::string out_var = region_name + "_" + in_var;

            surface_flux->append_input_variable(in_var);
            surface_flux->append_output_variable(out_var);
            out_arrays.push_back(std::move(out_var));
        }

        if (iwv)
        {
            std::string in_var = opt_vals["iwv"].as<string>();
            std::string out_var = region_name + "_" + in_var;

            surface_flux->append_input_variable(in_var);
            surface_flux->append_output_variable(out_var);
            out_arrays.push_back(std::move(out_var));
        }

        join->set_input_connection(0, moisture_flux->get_output_port());
        join->set_input_connection(1, surface_flux->get_output_port());

        head = join;
    }

    // reduce
    map_reduce->set_input_connection(head->get_output_port());
    map_reduce->set_thread_pool_size(opt_vals["n_threads"].as<int>());

    map_reduce->set_arrays(out_arrays);

    if (opt_vals.count("bounds"))
    {
        auto bounds = opt_vals["bounds"].as<std::vector<double>>();
        map_reduce->set_bounds(bounds);
    }

    if (!opt_vals["first_step"].defaulted())
        map_reduce->set_start_index(opt_vals["first_step"].as<long>());

    if (!opt_vals["last_step"].defaulted())
        map_reduce->set_end_index(opt_vals["last_step"].as<long>());

    // sort
    sort->set_input_connection(map_reduce->get_output_port());

    // calendaring
    calendar->set_input_connection(sort->get_output_port());

    // writer
    writer->set_input_connection(calendar->get_output_port());
    writer->set_file_name(opt_vals["output_file"].as<string>());

    // look for requested time step range, start
    bool parse_start_date = opt_vals.count("start_date");
    bool parse_end_date = opt_vals.count("end_date");
    if (parse_start_date || parse_end_date)
    {
        // run the reporting phase of the pipeline
        teca_metadata md = reader->update_metadata();

        teca_metadata atrs;
        if (md.get("attributes", atrs))
        {
            TECA_FATAL_ERROR("metadata missing attributes")
            return -1;
        }

        teca_metadata time_atts;
        std::string calendar;
        std::string units;
        if (atrs.get("time", time_atts)
           || time_atts.get("calendar", calendar)
           || time_atts.get("units", units))
        {
            TECA_FATAL_ERROR("failed to determine the calendaring parameters")
            return -1;
        }

        teca_metadata coords;
        p_teca_variant_array time;
        if (md.get("coordinates", coords) || !(time = coords.get("t")))
        {
            TECA_FATAL_ERROR("failed to determine time coordinate")
            return -1;
        }

        // convert date string to step, start date
        if (parse_start_date)
        {
            unsigned long first_step = 0;
            std::string start_date = opt_vals["start_date"].as<string>();
            if (teca_coordinate_util::time_step_of(time, true, true, calendar,
                 units, start_date, first_step))
            {
                TECA_FATAL_ERROR("Failed to locate time step for start date \""
                    <<  start_date << "\"")
                return -1;
            }
            map_reduce->set_start_index(first_step);
        }

        // and end date
        if (parse_end_date)
        {
            unsigned long last_step = 0;
            std::string end_date = opt_vals["end_date"].as<string>();
            if (teca_coordinate_util::time_step_of(time, false, true, calendar,
                 units, end_date, last_step))
            {
                TECA_FATAL_ERROR("Failed to locate time step for end date \""
                    <<  end_date << "\"")
                return -1;
            }
            map_reduce->set_end_index(last_step);
        }
    }

    // run the pipeline
    writer->update();

    return 0;
}
