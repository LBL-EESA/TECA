#include "teca_config.h"
#include "teca_cf_reader.h"
#include "teca_cf_writer.h"
#include "teca_index_executive.h"
#include "teca_normalize_coordinates.h"
#include "teca_metadata.h"
#include "teca_bayesian_ar_detect.h"
#include "teca_bayesian_ar_detect_parameters.h"
#include "teca_binary_segmentation.h"
#include "teca_mpi_manager.h"
#include "teca_coordinate_util.h"
#include "teca_table.h"
#include "teca_dataset_source.h"
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
    options_description basic_opt_defs(
        "Basic usage:\n\n"
        "The following options are the most commonly used. Information\n"
        "on advanced options can be displayed using --advanced_help\n\n"
        "Basic command line options", 120, -1
        );
    basic_opt_defs.add_options()
        ("input_file", value<string>(), "file path to the simulation to search for atmospheric rivers")
        ("output_file", value<string>()->default_value(std::string("bayesian_ar_detect_%t%.nc")),
            "file pattern for output netcdf files (%t% is the time index)")
        ("input_regex", value<string>(), "regex matching simulation files to search for atmospheric rivers")
        ("ivt", value<string>()->default_value(std::string("IVT")),
            "name of variable with integrated vapor transport (IVT)")
        ("binary_ar_threshold", value<double>()->default_value(0.6666666667),
            "probability threshold for segmenting ar_probability to produce ar_binary_tag")
        ("first_step", value<long>(), "first time step to process")
        ("last_step", value<long>(), "last time step to process")
        ("steps_per_file", value<long>(), "number of time steps per output file")
        ("start_date", value<string>(), "first time to proces in YYYY-MM-DD hh:mm:ss format")
        ("end_date", value<string>(), "first time to proces in YYYY-MM-DD hh:mm:ss format")
        ("n_threads", value<int>(), "thread pool size. default is -1. -1 for all")
        ("periodic_in_x", value<int>()->default_value(1),
            "Flags whether the x dimension (typically longitude) is periodic.")
        ("verbose", "enable extra terminal output")
        ("help", "display the basic options help")
        ("advanced_help", "display the advanced options help")
        ("full_help", "display entire help message")
        ;

    // add all options from each pipeline stage for more advanced use
    options_description advanced_opt_defs(
        "Advanced usage:\n\n"
        "The following list contains the full set options giving one full\n"
        "control over all runtime modifiable parameters. The basic options\n"
        "(see" "--help) map to these, and will override them if both are\n"
        "specified.\n\n"
        "Advanced command line options", -1, 1
        );

    // create the pipeline stages here, they contain the
    // documentation and parse command line.
    // objects report all of their properties directly
    // set default options here so that command line options override
    // them. while we are at it connect the pipeline
    p_teca_cf_reader cf_reader = teca_cf_reader::New();
    cf_reader->get_properties_description("cf_reader", advanced_opt_defs);

    p_teca_normalize_coordinates norm_coords = teca_normalize_coordinates::New();
    norm_coords->get_properties_description("norm_coords", advanced_opt_defs);
    norm_coords->set_input_connection(cf_reader->get_output_port());

    // parameter source
    p_teca_bayesian_ar_detect_parameters params
        = teca_bayesian_ar_detect_parameters::New();

    params->get_properties_description("parameter_table", advanced_opt_defs);

    // Construct the AR detector and attach the input file and parameters
    p_teca_bayesian_ar_detect ar_detect = teca_bayesian_ar_detect::New();
    ar_detect->get_properties_description("ar_detect", advanced_opt_defs);
    ar_detect->set_input_connection(0, params->get_output_port());
    ar_detect->set_input_connection(1, norm_coords->get_output_port());

    // segment the ar probability field
    p_teca_binary_segmentation ar_tag = teca_binary_segmentation::New();
    ar_tag->set_input_connection(0, ar_detect->get_output_port());
    ar_tag->set_threshold_mode(ar_tag->BY_VALUE);
    ar_tag->set_threshold_variable("ar_probability");
    ar_tag->set_segmentation_variable("ar_binary_tag");

    // Add an executive for the writer
    p_teca_index_executive exec = teca_index_executive::New();

    // Add the writer
    p_teca_cf_writer cf_writer = teca_cf_writer::New();
    cf_writer->get_properties_description("cf_writer", advanced_opt_defs);
    cf_writer->set_input_connection(ar_tag->get_output_port());
    cf_writer->set_verbose(0);
    cf_writer->set_thread_pool_size(1);

    // package basic and advanced options for display
    options_description all_opt_defs(-1, -1);
    all_opt_defs.add(basic_opt_defs).add(advanced_opt_defs);

    // parse the command line
    variables_map opt_vals;
    try
    {
        boost::program_options::store(
            boost::program_options::command_line_parser(argc, argv).options(all_opt_defs).run(),
            opt_vals);

        if (mpi_man.get_comm_rank() == 0)
        {
            if (opt_vals.count("help"))
            {
                cerr << endl
                    << "usage: teca_bayesian_ar_detect [options]" << endl
                    << endl
                    << basic_opt_defs << endl
                    << endl;
                return -1;
            }
            if (opt_vals.count("advanced_help"))
            {
                cerr << endl
                    << "usage: teca_bayesian_ar_detect [options]" << endl
                    << endl
                    << advanced_opt_defs << endl
                    << endl;
                return -1;
            }

            if (opt_vals.count("full_help"))
            {
                cerr << endl
                    << "usage: teca_bayesian_ar_detect [options]" << endl
                    << endl
                    << all_opt_defs << endl
                    << endl;
                return -1;
            }
        }

        boost::program_options::notify(opt_vals);
    }
    catch (std::exception &e)
    {
        TECA_ERROR("Error parsing command line options. See --help "
            "for a list of supported options. " << e.what())
        return -1;
    }

    // pass command line arguments into the pipeline objects
    // advanced options are processed first, so that the basic
    // options will override them
    cf_reader->set_properties("cf_reader", opt_vals);
    norm_coords->set_properties("norm_coords", opt_vals);
    params->set_properties("parameter_table", opt_vals);
    ar_detect->set_properties("ar_detect", opt_vals);
    cf_writer->set_properties("cf_writer", opt_vals);

    // now pass in the basic options, these are processed
    // last so that they will take precedence
    if (opt_vals.count("input_file"))
        cf_reader->append_file_name(
            opt_vals["input_file"].as<string>());

    if (opt_vals.count("input_regex"))
        cf_reader->set_files_regex(
            opt_vals["input_regex"].as<string>());

    if (opt_vals.count("periodic_in_x"))
        cf_reader->set_periodic_in_x(
            opt_vals["periodic_in_x"].as<int>());

    if (opt_vals.count("output_file"))
        cf_writer->set_file_name(
            opt_vals["output_file"].as<string>());

    if (opt_vals.count("ivt"))
        ar_detect->set_water_vapor_variable(
            opt_vals["ivt"].as<string>());

    if (opt_vals.count("steps_per_file"))
        cf_writer->set_steps_per_file(
            opt_vals["steps_per_file"].as<long>());

    if (opt_vals.count("first_step"))
        cf_writer->set_first_step(opt_vals["first_step"].as<long>());

    if (opt_vals.count("last_step"))
        cf_writer->set_last_step(opt_vals["last_step"].as<long>());

    if (opt_vals.count("verbose"))
    {
        ar_detect->set_verbose(1);
        cf_writer->set_verbose(1);
        exec->set_verbose(1);
    }

    if (opt_vals.count("n_threads"))
        ar_detect->set_thread_pool_size(opt_vals["n_threads"].as<int>());
    else
        ar_detect->set_thread_pool_size(-1);


    // some minimal check for missing options
    if (cf_reader->get_number_of_file_names() == 0
        && cf_reader->get_files_regex().empty())
    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_ERROR(
                "missing file name or regex for simulation reader. "
                "See --help for a list of command line options.")
        }
        return -1;
    }

    if (cf_writer->get_file_name().empty())
    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_ERROR(
                "missing file name pattern for netcdf writer. "
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
        teca_metadata md = cf_reader->update_metadata();

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
        p_teca_double_array time;
        if (md.get("coordinates", coords) ||
            !(time = std::dynamic_pointer_cast<teca_double_array>(
                coords.get("t"))))
        {
            TECA_ERROR("failed to determine time coordinate")
            return -1;
        }

        // convert date string to step, start date
        if (parse_start_date)
        {
            unsigned long first_step = 0;
            std::string start_date = opt_vals["start_date"].as<string>();
            if (teca_coordinate_util::time_step_of(time, true, calendar,
                 units, start_date, first_step))
            {
                TECA_ERROR("Failed to locate time step for start date \""
                    <<  start_date << "\"")
                return -1;
            }
            exec->set_start_index(first_step);
        }

        // and end date
        if (parse_end_date)
        {
            unsigned long last_step = 0;
            std::string end_date = opt_vals["end_date"].as<string>();
            if (teca_coordinate_util::time_step_of(time, false, calendar,
                 units, end_date, last_step))
            {
                TECA_ERROR("Failed to locate time step for end date \""
                    <<  end_date << "\"")
                return -1;
            }
            exec->set_end_index(last_step);
        }
    }


    double ar_tag_threshold = opt_vals["binary_ar_threshold"].as<double>();
    // set the threshold for calculating ar_binary_tag
    ar_tag->set_low_threshold_value(ar_tag_threshold);
    // add metadata for ar_binary_tag
    teca_metadata seg_atts;
    seg_atts.set("long_name",std::string("binary indicator of atmospheric river"));
    seg_atts.set("description",std::string("binary indicator of atmospheric river"));
    seg_atts.set("scheme",std::string("cascade_bard"));
    seg_atts.set("version",std::string("1.0"));
    seg_atts.set("note",
        std::string("derived by thresholding ar_probability >= ") + 
        std::to_string(ar_tag_threshold));
    ar_tag->set_segmentation_variable_atts(seg_atts);

    // run the pipeline

    cf_writer->set_executive(exec);
    cf_writer->update();

    return 0;
}
