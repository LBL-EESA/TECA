#include "teca_config.h"
#include "teca_cf_reader.h"
#include "teca_cf_writer.h"
#include "teca_index_executive.h"
#include "teca_normalize_coordinates.h"
#include "teca_metadata.h"
#include "teca_bayesian_ar_detect.h"
#include "teca_bayesian_ar_detect_parameters.h"
#include "teca_binary_segmentation.h"
#include "teca_l2_norm.h"
#include "teca_apply_binary_mask.h"
#include "teca_multi_cf_reader.h"
#include "teca_integrated_vapor_transport.h"
#include "teca_valid_value_mask.h"
#include "teca_unpack_data.h"
#include "teca_mpi_manager.h"
#include "teca_coordinate_util.h"
#include "teca_table.h"
#include "teca_dataset_source.h"
#include "teca_app_util.h"
#include "calcalcs.h"

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
        "on all available options can be displayed using --advanced_help\n\n"
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

        ("ivt", value<std::string>()->default_value(std::string("IVT")),
            "\nname of variable with the magnitude of integrated vapor transport\n")

        ("compute_ivt_magnitude", "\nwhen this flag is present magnitude of vector"
            " IVT is calculated. use --ivt_u and --ivt_v to set the name of the IVT"
            " vector components if needed.\n")
        ("ivt_u", value<std::string>()->default_value(std::string("IVT_U")),
            "\nname of variable with longitudinal component of the integrated vapor"
            " transport vector.\n")
        ("ivt_v", value<std::string>()->default_value(std::string("IVT_V")),
            "\nname of variable with latitudinal component of the integrated vapor"
            " transport vector.\n")
        ("write_ivt_magnitude", "\nwhen this flag is present IVT magnitude is"
            " written to disk with the AR detector results\n")

        ("compute_ivt", "\nwhen this flag is present IVT vector is calculated from"
            " specific humidity, and wind vector components. use --specific_humidity"
            " --wind_u and --wind_v to set the name of the specific humidity and wind"
            " vector components, and --ivt_u and --ivt_v to control the names of"
            " the results, if needed.\n")
        ("specific_humidity", value<std::string>()->default_value(std::string("Q")),
            "\nname of variable with the 3D specific humidity field.(Q)\n")
        ("wind_u", value<std::string>()->default_value(std::string("U")),
            "\nname of variable with the 3D longitudinal component of the wind vector.\n")
        ("wind_v", value<std::string>()->default_value(std::string("V")),
            "\nname of variable with the 3D latitudinal component of the wind vector.\n")
        ("write_ivt", "\nwhen this flag is present IVT vector is written to disk with"
            " the result\n")

        ("ar_weighted_variables", value<std::vector<std::string>>()->multitoken(),
            "\nAn optional list of variables to weight with the computed AR probability."
            " Each such variable will be multiplied by the computed AR probability, and"
            " written to disk as \"NAME_ar_wgtd\".\n")

        ("x_axis_variable", value<std::string>()->default_value("lon"),
            "\nname of x coordinate variable\n")
        ("y_axis_variable", value<std::string>()->default_value("lat"),
            "\nname of y coordinate variable\n")
        ("z_axis_variable", value<std::string>()->default_value("plev"),
            "\nname of z coordinate variable\n")

        ("periodic_in_x", value<int>()->default_value(1),
            "\nFlags whether the x dimension (typically longitude) is periodic.\n")

        ("binary_ar_threshold", value<double>()->default_value(2.0/3.0,"0.667"),
            "\nprobability threshold for segmenting ar_probability to produce ar_binary_tag\n")

        ("output_file", value<std::string>()->default_value(std::string("TECA_BARD_%t%.nc")),
            "\nA path and file name pattern for the output NetCDF files. %t% is replaced with a"
            " human readable date and time corresponding to the time of the first time step in"
            " the file. Use --cf_writer::date_format to change the formatting\n")

        ("steps_per_file", value<long>()->default_value(128),
            "\nnumber of time steps per output file\n")

        ("first_step", value<long>()->default_value(0), "\nfirst time step to process\n")
        ("last_step", value<long>()->default_value(-1), "\nlast time step to process\n")

        ("start_date", value<std::string>(), "\nThe first time to process in 'Y-M-D h:m:s'"
            " format. Note: There must be a space between the date and time specification\n")
        ("end_date", value<std::string>(), "\nThe last time to process in 'Y-M-D h:m:s' format\n")

        ("n_threads", value<int>()->default_value(-1), "\nSets the thread pool size on each"
            " MPI rank. When the default value of -1 is used TECA will coordinate the thread"
            " pools across ranks such each thread is bound to a unique physical core.\n")

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
        "Advanced command line options:", help_width, help_width - 4
        );

    // create the pipeline stages here, they contain the
    // documentation and parse command line.
    // objects report all of their properties directly
    // set default options here so that command line options override
    // them. while we are at it connect the pipeline
    p_teca_cf_reader cf_reader = teca_cf_reader::New();
    cf_reader->get_properties_description("cf_reader", advanced_opt_defs);
    cf_reader->set_periodic_in_x(1);

    p_teca_multi_cf_reader mcf_reader = teca_multi_cf_reader::New();
    mcf_reader->get_properties_description("mcf_reader", advanced_opt_defs);

    p_teca_l2_norm l2_norm = teca_l2_norm::New();
    l2_norm->get_properties_description("ivt_magnitude", advanced_opt_defs);
    l2_norm->set_component_0_variable("IVT_U");
    l2_norm->set_component_1_variable("IVT_V");
    l2_norm->set_l2_norm_variable("IVT");

    p_teca_integrated_vapor_transport ivt_int = teca_integrated_vapor_transport::New();
    ivt_int->get_properties_description("ivt_integral", advanced_opt_defs);
    ivt_int->set_specific_humidity_variable("Q");
    ivt_int->set_wind_u_variable("U");
    ivt_int->set_wind_v_variable("V");
    ivt_int->set_ivt_u_variable("IVT_U");
    ivt_int->set_ivt_v_variable("IVT_V");

    p_teca_valid_value_mask vv_mask = teca_valid_value_mask::New();
    vv_mask->get_properties_description("vv_mask", advanced_opt_defs);

    p_teca_unpack_data unpack = teca_unpack_data::New();
    unpack->get_properties_description("unpack", advanced_opt_defs);

    p_teca_normalize_coordinates norm_coords = teca_normalize_coordinates::New();
    norm_coords->get_properties_description("norm_coords", advanced_opt_defs);

    // parameter source
    p_teca_bayesian_ar_detect_parameters params
        = teca_bayesian_ar_detect_parameters::New();

    params->get_properties_description("parameter_table", advanced_opt_defs);

    // Construct the AR detector and attach the input file and parameters
    p_teca_bayesian_ar_detect ar_detect = teca_bayesian_ar_detect::New();
    ar_detect->get_properties_description("ar_detect", advanced_opt_defs);
    ar_detect->set_ivt_variable("IVT");

    // segment the ar probability field
    p_teca_binary_segmentation ar_tag = teca_binary_segmentation::New();
    ar_tag->set_threshold_mode(ar_tag->BY_VALUE);
    ar_tag->set_threshold_variable("ar_probability");
    ar_tag->set_segmentation_variable("ar_binary_tag");

    // mask any requested variables by "ar_probability"
    p_teca_apply_binary_mask ar_mask = teca_apply_binary_mask::New();
    ar_mask->get_properties_description("ar_mask", advanced_opt_defs);
    ar_mask->set_mask_variable("ar_probability");
    ar_mask->set_output_variable_prefix("ar_wgtd_");

    // Add an executive for the writer
    p_teca_index_executive exec = teca_index_executive::New();

    // Add the writer
    p_teca_cf_writer cf_writer = teca_cf_writer::New();
    cf_writer->get_properties_description("cf_writer", advanced_opt_defs);
    cf_writer->set_verbose(0);
    cf_writer->set_thread_pool_size(1);
    cf_writer->set_steps_per_file(128);

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
    l2_norm->set_properties("ivt_magnitude", opt_vals);
    ivt_int->set_properties("ivt_integral", opt_vals);
    vv_mask->set_properties("vv_mask", opt_vals);
    unpack->set_properties("unpack", opt_vals);
    norm_coords->set_properties("norm_coords", opt_vals);
    params->set_properties("parameter_table", opt_vals);
    ar_detect->set_properties("ar_detect", opt_vals);
    ar_mask->set_properties("ar_mask", opt_vals);
    cf_writer->set_properties("cf_writer", opt_vals);

    // now pass in the basic options, these are processed
    // last so that they will take precedence
    // configure the pipeline from the command line options.

    // configure the reader
    bool have_file = opt_vals.count("input_file");
    bool have_regex = opt_vals.count("input_regex");

    if ((have_file && have_regex) || !(have_file || have_regex))
    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_ERROR("Extacly one of --input_file or --input_regex can be specified. "
                "Use --input_file to activate the multi_cf_reader (HighResMIP datasets) "
                "and --input_regex to activate the cf_reader (CAM like datasets)")
        }
        return -1;
    }


    p_teca_algorithm reader;
    if (have_file)
    {
        mcf_reader->set_input_file(opt_vals["input_file"].as<string>());
        reader = mcf_reader;
    }
    else if (have_regex)
    {
        cf_reader->set_files_regex(opt_vals["input_regex"].as<string>());
        reader = cf_reader;
    }

    // add transformation stages to the pipeline
    norm_coords->set_input_connection(reader->get_output_port());
    vv_mask->set_input_connection(norm_coords->get_output_port());
    unpack->set_input_connection(vv_mask->get_output_port());

    p_teca_algorithm head = unpack;

    if (!opt_vals["periodic_in_x"].defaulted())
    {
        cf_reader->set_periodic_in_x(opt_vals["periodic_in_x"].as<int>());
        mcf_reader->set_periodic_in_x(opt_vals["periodic_in_x"].as<int>());
    }

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
        ar_detect->set_ivt_variable(opt_vals["ivt"].as<string>());
    }

    // add the ivt calculation stages if needed
    bool do_ivt = opt_vals.count("compute_ivt");
    bool do_ivt_magnitude = opt_vals.count("compute_ivt_magnitude");

    if (do_ivt && do_ivt_magnitude)
    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_ERROR("Only one of --compute_ivt and compute_ivt_magnitude can "
                "be specified. --compute_ivt implies --compute_ivt_magnitude")
        }
        return -1;
    }

    if (do_ivt)
    {
        std::string z_var = "plev";
        if (!opt_vals["z_axis_variable"].defaulted())
            z_var = opt_vals["z_axis_variable"].as<string>();

        cf_reader->set_z_axis_variable(z_var);
        mcf_reader->set_z_axis_variable(z_var);

        ivt_int->set_input_connection(head->get_output_port());
        l2_norm->set_input_connection(ivt_int->get_output_port());

        head = l2_norm;
    }
    else if (do_ivt_magnitude)
    {
        l2_norm->set_input_connection(head->get_output_port());
        head = l2_norm;
    }

    // connect the detector and post detector operations
    ar_detect->set_input_connection(0, params->get_output_port());
    ar_detect->set_input_connection(1, head->get_output_port());

    ar_tag->set_input_connection(0, ar_detect->get_output_port());
    head = ar_tag;

    // set the variables to weight by AR probability
    if (opt_vals.count("ar_weighted_variables"))
    {
        ar_mask->set_masked_variables
            (opt_vals["ar_weighted_variables"].as<std::vector<std::string>>());
    }

    // if there are any variables to weight add the mask stage to the pipeline
    bool do_weighted = ar_mask->get_number_of_masked_variables();
    if (do_weighted)
    {
        ar_mask->set_input_connection(0, ar_tag->get_output_port());
        head = ar_mask;
    }

    // connect and configure the writer
    cf_writer->set_input_connection(head->get_output_port());

    // tell the writer to write ivt if needed
    std::vector<std::string> point_arrays({"ar_probability", "ar_binary_tag"});
    if ((do_ivt || do_ivt_magnitude) && opt_vals.count("write_ivt_magnitude"))
    {
        point_arrays.push_back(l2_norm->get_l2_norm_variable());
    }

    if (do_ivt && opt_vals.count("write_ivt"))
    {
        point_arrays.push_back(ivt_int->get_ivt_u_variable());
        point_arrays.push_back(ivt_int->get_ivt_v_variable());
    }

    // tell the writer to write ar weighted variables if needed
    if (do_weighted)
    {
        ar_mask->get_output_variable_names(point_arrays);
    }

    cf_writer->set_file_name(opt_vals["output_file"].as<string>());
    cf_writer->set_information_arrays({"ar_count", "parameter_table_row"});
    cf_writer->set_point_arrays(point_arrays);

    if (!opt_vals["steps_per_file"].defaulted())
        cf_writer->set_steps_per_file(opt_vals["steps_per_file"].as<long>());

    if (!opt_vals["first_step"].defaulted())
        cf_writer->set_first_step(opt_vals["first_step"].as<long>());

    if (!opt_vals["last_step"].defaulted())
        cf_writer->set_last_step(opt_vals["last_step"].as<long>());

    if (opt_vals.count("verbose"))
    {
        ar_detect->set_verbose(1);
        cf_writer->set_verbose(1);
        exec->set_verbose(1);
    }

    if (!opt_vals["n_threads"].defaulted())
        ar_detect->set_thread_pool_size(opt_vals["n_threads"].as<int>());
    else
        ar_detect->set_thread_pool_size(-1);

    if (cf_writer->get_file_name().empty())
    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_ERROR("missing file name pattern for netcdf writer. "
                "See --help for a list of command line options.")
        }
        return -1;
    }

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
            TECA_ERROR("metadata missing attributes")
            return -1;
        }

        teca_metadata time_atts;
        std::string calendar;
        std::string units;
        if (atrs.get("time", time_atts)
           || time_atts.get("calendar", calendar)
           || time_atts.get("units", units))
        {
            TECA_ERROR("failed to determine the calendaring parameters")
            return -1;
        }

        teca_metadata coords;
        p_teca_variant_array time;
        if (md.get("coordinates", coords) || !(time = coords.get("t")))
        {
            TECA_ERROR("failed to determine time coordinate")
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
                TECA_ERROR("Failed to locate time step for start date \""
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
                TECA_ERROR("Failed to locate time step for end date \""
                    <<  end_date << "\"")
                return -1;
            }
            cf_writer->set_last_step(last_step);
        }
    }

    // set the threshold for calculating ar_binary_tag
    double ar_tag_threshold = opt_vals["binary_ar_threshold"].as<double>();
    ar_tag->set_low_threshold_value(ar_tag_threshold);

    // add metadata for ar_binary_tag
    teca_metadata seg_atts;
    seg_atts.set("long_name", std::string("binary indicator of atmospheric river"));
    seg_atts.set("description", std::string("binary indicator of atmospheric river"));
    seg_atts.set("scheme", std::string("TECA_BARD"));
    seg_atts.set("version", std::string("1.0"));
    seg_atts.set("note",
        std::string("derived by thresholding ar_probability >= ") +
        std::to_string(ar_tag_threshold));
    ar_tag->set_segmentation_variable_attributes(seg_atts);

    // run the pipeline
    cf_writer->set_executive(exec);
    cf_writer->update();

    return 0;
}
