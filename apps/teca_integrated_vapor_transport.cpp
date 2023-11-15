#include "teca_config.h"
#include "teca_cf_reader.h"
#include "teca_cf_writer.h"
#include "teca_index_executive.h"
#include "teca_normalize_coordinates.h"
#include "teca_metadata.h"
#include "teca_integrated_vapor_transport.h"
#include "teca_binary_segmentation.h"
#include "teca_l2_norm.h"
#include "teca_multi_cf_reader.h"
#include "teca_integrated_vapor_transport.h"
#include "teca_valid_value_mask.h"
#include "teca_unpack_data.h"
#include "teca_cartesian_mesh_source.h"
#include "teca_cartesian_mesh_regrid.h"
#include "teca_elevation_mask.h"
#include "teca_indexed_dataset_cache.h"
#include "teca_mpi_manager.h"
#include "teca_coordinate_util.h"
#include "teca_table.h"
#include "teca_dataset_source.h"
#include "teca_app_util.h"
#include "teca_calcalcs.h"

#include <vector>
#include <string>
#include <iostream>
#include <boost/program_options.hpp>

#include "teca_cartesian_mesh_writer.h"

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

        ("specific_humidity", value<std::string>()->default_value("Q"),
            "\nname of variable with the 3D specific humidity field.\n")

        ("wind_u", value<std::string>()->default_value("U"),
            "\nname of variable with the 3D longitudinal component of the wind vector.\n")
        ("wind_v", value<std::string>()->default_value("V"),
            "\nname of variable with the 3D latitudinal component of the wind vector.\n")

        ("ivt_u", value<std::string>()->default_value("IVT_U"),
            "\nname to use for the longitudinal component of the integrated vapor transport vector.\n")
        ("ivt_v", value<std::string>()->default_value("IVT_V"),
            "\nname to use for the latitudinal component of the integrated vapor transport vector.\n")

        ("ivt", value<std::string>()->default_value("IVT"),
            "\nname of variable with the magnitude of integrated vapor transport (IVT)\n")

        ("write_ivt_magnitude", value<int>()->default_value(0),
            "\nwhen this is set to 1 magnitude of vector IVT is calculated. use --ivt_u and"
            " --ivt_v to set the name of the IVT vector components and --ivt to set the name"
            " of the result if needed.\n")

        ("write_ivt", value<int>()->default_value(1),
            "\nwhen this is set to 1 IVT vector is written to disk with the result. use"
            " --ivt_u and --ivt_v to set the name of the IVT vector components of the"
            " result if needed.\n")

        ("output_file", value<std::string>()->default_value("IVT_%t%.nc"),
            "\nA path and file name pattern for the output NetCDF files. %t% is replaced with a"
            " human readable date and time corresponding to the time of the first time step in"
            " the file. Use --cf_writer::date_format to change the formatting\n")
        ("file_layout", value<std::string>()->default_value("monthly"),
            "\nSelects the size and layout of the set of output files. May be one of"
            " number_of_steps, daily, monthly, seasonal, or yearly. Files are structured"
            " such that each file contains one of the selected interval. For the number_of_steps"
            " option use --steps_per_file.\n")
        ("steps_per_file", value<long>()->default_value(128),
            "\nThe number of time steps per output file when --file_layout number_of_steps is"
            " specified.\n")

        ("x_axis_variable", value<std::string>()->default_value("lon"),
            "\nname of x coordinate variable\n")
        ("y_axis_variable", value<std::string>()->default_value("lat"),
            "\nname of y coordinate variable\n")
        ("z_axis_variable", value<std::string>()->default_value("plev"),
            "\nname of z coordinate variable\n")

        ("dem", value<std::string>(), "\nA teca_cf_reader regex identifying the"
            " file containing surface elevation field or DEM.\n")

        ("dem_variable", value<std::string>()->default_value("Z"),
            "\nSets the name of the variable containing the surface elevation field\n")

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
        "integrated vapor transport pipeline:\n\n"
        "    (cf / mcf_reader)\n"
        "            \\\n"
        "        (ivt_integral)--(ivt_magnitude)\n"
        "                                 \\\n"
        "                              (cf_writer)\n\n"
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
    norm_coords->set_enable_unit_conversions(1);

    p_teca_valid_value_mask vv_mask = teca_valid_value_mask::New();
    vv_mask->get_properties_description("vv_mask", advanced_opt_defs);

    p_teca_unpack_data unpack = teca_unpack_data::New();
    unpack->get_properties_description("unpack", advanced_opt_defs);

    p_teca_cf_reader elev_reader = teca_cf_reader::New();
    elev_reader->get_properties_description("elev_reader", advanced_opt_defs);
    elev_reader->set_t_axis_variable("");

    p_teca_normalize_coordinates elev_coords = teca_normalize_coordinates::New();
    elev_coords->get_properties_description("elev_coords", advanced_opt_defs);
    elev_coords->set_enable_periodic_shift_x(1);

    p_teca_indexed_dataset_cache elev_cache = teca_indexed_dataset_cache::New();
    elev_cache->get_properties_description("elev_cache", advanced_opt_defs);
    elev_cache->set_max_cache_size(1);

    p_teca_cartesian_mesh_source elev_mesh = teca_cartesian_mesh_source::New();
    elev_mesh->get_properties_description("elev_mesh", advanced_opt_defs);

    p_teca_cartesian_mesh_regrid elev_regrid = teca_cartesian_mesh_regrid::New();
    elev_regrid->get_properties_description("elev_regrid", advanced_opt_defs);

    p_teca_elevation_mask elev_mask = teca_elevation_mask::New();
    elev_mask->get_properties_description("elev_mask", advanced_opt_defs);
    elev_mask->set_surface_elevation_variable("Z");
    elev_mask->set_mesh_height_variable("ZG");

    p_teca_integrated_vapor_transport ivt_int = teca_integrated_vapor_transport::New();
    ivt_int->get_properties_description("ivt_integral", advanced_opt_defs);
    ivt_int->set_specific_humidity_variable("Q");
    ivt_int->set_wind_u_variable("U");
    ivt_int->set_wind_v_variable("V");
    ivt_int->set_ivt_u_variable("IVT_U");
    ivt_int->set_ivt_v_variable("IVT_V");

    p_teca_l2_norm l2_norm = teca_l2_norm::New();
    l2_norm->get_properties_description("ivt_magnitude", advanced_opt_defs);
    l2_norm->set_component_0_variable("IVT_U");
    l2_norm->set_component_1_variable("IVT_V");
    l2_norm->set_l2_norm_variable("IVT");

    // Add an executive for the writer
    p_teca_index_executive exec = teca_index_executive::New();

    // Add the writer
    p_teca_cf_writer cf_writer = teca_cf_writer::New();
    cf_writer->get_properties_description("cf_writer", advanced_opt_defs);
    cf_writer->set_verbose(0);
    cf_writer->set_steps_per_file(128);
    cf_writer->set_layout(teca_cf_writer::monthly);

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
    elev_reader->set_properties("elev_reader", opt_vals);
    elev_coords->set_properties("elev_coords", opt_vals);
    elev_mesh->set_properties("elev_mesh", opt_vals);
    elev_cache->set_properties("elev_cache", opt_vals);
    elev_regrid->set_properties("elev_regrid", opt_vals);
    elev_mask->set_properties("elev_mask", opt_vals);
    ivt_int->set_properties("ivt_integral", opt_vals);
    l2_norm->set_properties("ivt_magnitude", opt_vals);
    cf_writer->set_properties("cf_writer", opt_vals);

    // now pass in the basic options, these are processed
    // last so that they will take precedence
    // configure the pipeline from the command line options.
    p_teca_algorithm reader;

    // configure the reader
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

    std::string z_var = "plev";
    if (!opt_vals["z_axis_variable"].defaulted())
        z_var = opt_vals["z_axis_variable"].as<string>();

    cf_reader->set_z_axis_variable(z_var);
    mcf_reader->set_z_axis_variable(z_var);

    // set the inputs to the integrator
    if (!opt_vals["wind_u"].defaulted())
    {
        ivt_int->set_wind_u_variable(opt_vals["wind_u"].as<string>());
    }

    if (!opt_vals["wind_v"].defaulted())
    {
        ivt_int->set_wind_v_variable(opt_vals["wind_v"].as<string>());
    }

    if (!opt_vals["specific_humidity"].defaulted())
    {
        ivt_int->set_specific_humidity_variable(
            opt_vals["specific_humidity"].as<string>());
    }

    // set all that use or produce ivt
    if (!opt_vals["ivt_u"].defaulted())
    {
        ivt_int->set_ivt_u_variable(opt_vals["ivt_u"].as<string>());
        l2_norm->set_component_0_variable(opt_vals["ivt_u"].as<string>());
    }

    if (!opt_vals["ivt_v"].defaulted())
    {
        ivt_int->set_ivt_v_variable(opt_vals["ivt_v"].as<string>());
        l2_norm->set_component_1_variable(opt_vals["ivt_v"].as<string>());
    }

    if (!opt_vals["ivt"].defaulted())
    {
        l2_norm->set_l2_norm_variable(opt_vals["ivt"].as<string>());
    }

    // add the valid value mask stage
    norm_coords->set_input_connection(head->get_output_port());
    vv_mask->set_input_connection(norm_coords->get_output_port());
    unpack->set_input_connection(vv_mask->get_output_port());
    head = unpack;

    // add the ivt caluation stages if needed
    bool do_ivt = opt_vals["write_ivt"].as<int>();
    bool do_ivt_magnitude = opt_vals["write_ivt_magnitude"].as<int>();
    if (!(do_ivt || do_ivt_magnitude))

    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_FATAL_ERROR("At least one of --write_ivt or --write_ivt_magnitude "
                " must be set.")
        }
        return -1;
    }

    // add the elevation mask stages
    teca_metadata md;
    if (opt_vals.count("dem"))
    {
        if (mpi_man.get_comm_rank() == 0)
            TECA_STATUS("Generating elevation mask")

        elev_reader->set_files_regex(opt_vals["dem"].as<string>());

        elev_coords->set_input_connection(elev_reader->get_output_port());

        md = head->update_metadata();

        elev_mesh->set_spatial_bounds(md, false);
        elev_mesh->set_spatial_extents(md, false);
        elev_mesh->set_x_axis_variable(md);
        elev_mesh->set_y_axis_variable(md);
        elev_mesh->set_z_axis_variable(md);
        elev_mesh->set_t_axis_variable(md);
        elev_mesh->set_t_axis(md);

        elev_regrid->set_input_connection(0, elev_mesh->get_output_port());
        elev_regrid->set_input_connection(1, elev_coords->get_output_port());

        elev_cache->set_input_connection(elev_regrid->get_output_port());

        /*p_teca_cartesian_mesh_writer rdw = teca_cartesian_mesh_writer::New();
        rdw->set_input_connection(elev_cache->get_output_port());
        rdw->set_file_name("regrid_dem_%t%.vtk");*/

        elev_mask->set_input_connection(0, head->get_output_port());
        elev_mask->set_input_connection(1, elev_cache->get_output_port());
        //elev_mask->set_input_connection(1, rdw->get_output_port());

        if (!opt_vals["dem_variable"].defaulted())
            elev_mask->set_surface_elevation_variable(
                opt_vals["dem_variable"].as<string>());

        if (!opt_vals["mesh_height"].defaulted())
            elev_mask->set_mesh_height_variable(
                opt_vals["mesh_height"].as<string>());

        elev_mask->set_mask_variables({
            ivt_int->get_specific_humidity_variable() + "_valid",
            ivt_int->get_wind_u_variable() + "_valid",
            ivt_int->get_wind_v_variable() + "_valid"});

        /*p_teca_cartesian_mesh_writer emw = teca_cartesian_mesh_writer::New();
        emw->set_input_connection(elev_mask->get_output_port());
        emw->set_file_name("elev_mask_%t%.vtk");
        emw->set_binary(1);
        head = emw;*/

        head = elev_mask;
    }

    ivt_int->set_input_connection(head->get_output_port());

    if (do_ivt_magnitude)
    {
        if (mpi_man.get_comm_rank() == 0)
            TECA_STATUS("Computing IVT magnitude")

        l2_norm->set_input_connection(ivt_int->get_output_port());
        head = l2_norm;
    }

    // tell the writer to write ivt if needed
    std::vector<std::string> point_arrays;
    if (do_ivt)
    {
        point_arrays.push_back(ivt_int->get_ivt_u_variable());
        point_arrays.push_back(ivt_int->get_ivt_v_variable());
    }
    if (do_ivt_magnitude)
    {
        point_arrays.push_back(l2_norm->get_l2_norm_variable());
    }
    cf_writer->set_point_arrays(point_arrays);

    cf_writer->set_file_name(opt_vals["output_file"].as<string>());

    if (!opt_vals["steps_per_file"].defaulted())
        cf_writer->set_steps_per_file(opt_vals["steps_per_file"].as<long>());

    if (!opt_vals["first_step"].defaulted())
        cf_writer->set_first_step(opt_vals["first_step"].as<long>());

    if (!opt_vals["last_step"].defaulted())
        cf_writer->set_last_step(opt_vals["last_step"].as<long>());

    if (!opt_vals["file_layout"].defaulted() &&
        cf_writer->set_layout(opt_vals["file_layout"].as<std::string>()))
    {
        TECA_FATAL_ERROR("An invalid file layout was provided \""
            << opt_vals["file_layout"].as<std::string>() << "\"")
        return -1;
    }

    if (opt_vals.count("verbose"))
    {
        cf_writer->set_verbose(1);
        exec->set_verbose(1);
    }

    cf_writer->set_thread_pool_size(opt_vals["n_threads"].as<int>());

    // some minimal check for missing options
    if (cf_writer->get_file_name().empty())
    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_FATAL_ERROR("missing file name pattern for netcdf writer. "
                "See --help for a list of command line options.")
        }
        return -1;
    }

    // connect the fixed stages of the pipeline
    cf_writer->set_input_connection(head->get_output_port());

    // look for requested time step range, start
    bool parse_start_date = opt_vals.count("start_date");
    bool parse_end_date = opt_vals.count("end_date");
    if (parse_start_date || parse_end_date)
    {
        // run the reporting phase of the pipeline
        if (md.empty())
            md = reader->update_metadata();

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
            cf_writer->set_first_step(first_step);
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
            cf_writer->set_last_step(last_step);
        }
    }

    // run the pipeline
    cf_writer->set_executive(exec);
    cf_writer->update();

    return 0;
}
