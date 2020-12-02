#include "teca_config.h"
#include "teca_metadata.h"
#include "teca_array_attributes.h"
#include "teca_variant_array.h"
#include "teca_cf_reader.h"
#include "teca_multi_cf_reader.h"
#include "teca_cf_writer.h"
#include "teca_dataset_diff.h"
#include "teca_index_executive.h"
#include "teca_system_interface.h"
#include "teca_coordinate_util.h"
#include "teca_file_util.h"
#include "teca_mpi_manager.h"
#include "teca_mpi.h"
#include "teca_app_util.h"
#include "calcalcs.h"

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <boost/program_options.hpp>

using boost::program_options::value;


int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    teca_system_interface::set_stack_trace_on_error();
    teca_system_interface::set_stack_trace_on_mpi_error();

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

        ("x_axis_variable", value<std::string>(), "\nname of x coordinate variable (lon)\n")
        ("y_axis_variable", value<std::string>(), "\nname of y coordinate variable (lat)\n")
        ("z_axis_variable", value<std::string>(), "\nname of z coordinate variable (plev)\n")

        ("point_arrays", value<std::vector<std::string>>()->multitoken(),
            "\nA list of point centered arrays to write\n")

        ("information_arrays", value<std::vector<std::string>>()->multitoken(),
            "\nA list of non-geometric arrays to write\n")

        ("output_file", value<std::string>()->default_value(std::string("IVT_%t%.nc")),
            "\nA path and file name pattern for the output NetCDF files. %t% is replaced with a"
            " human readable date and time corresponding to the time of the first time step in"
            " the file. Use --cf_writer::date_format to change the formatting\n")

        ("steps_per_file", value<long>(), "\nnumber of time steps per output file\n")

        ("bounds", value<std::vector<double>>()->multitoken(),
            "\nlat lon lev bounding box to subset with\n")

        ("first_step", value<long>(), "\nfirst time step to process\n")
        ("last_step", value<long>(), "\nlast time step to process\n")
        ("start_date", value<std::string>(), "\nfirst time to proces in YYYY-MM-DD hh:mm:ss format\n")
        ("end_date", value<std::string>(), "\nfirst time to proces in YYYY-MM-DD hh:mm:ss format\n")
        ("n_threads", value<int>(), "\nSets the thread pool size on each MPI rank. When the default"
            " value of -1 is used TECA will coordinate the thread pools across ranks such each"
            " thread is bound to a unique physical core.\n")
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

    p_teca_cf_writer cf_writer = teca_cf_writer::New();
    cf_writer->get_properties_description("cf_writer", advanced_opt_defs);

    // Add an executive for the writer
    p_teca_index_executive exec = teca_index_executive::New();

    // package basic and advanced options for display
    options_description all_opt_defs(help_width, help_width - 4);
    all_opt_defs.add(basic_opt_defs).add(advanced_opt_defs);

    // parse the command line
    variables_map opt_vals;
    if (teca_app_util::process_command_line_help(mpi_man.get_comm_rank(),
        argc, argv, basic_opt_defs, advanced_opt_defs, all_opt_defs, opt_vals))
    {
        return -1;
    }

    // pass command line arguments into the pipeline objects
    // advanced options are processed first, so that the basic
    // options will override them
    cf_reader->set_properties("cf_reader", opt_vals);
    mcf_reader->set_properties("mcf_reader", opt_vals);
    cf_writer->set_properties("cf_writer", opt_vals);

    // now pass in the basic options, these are processed
    // last so that they will take precedence
    p_teca_algorithm reader;
    bool have_file = opt_vals.count("input_file");
    bool have_regex = opt_vals.count("input_regex");
    if (have_file)
    {
        mcf_reader->set_input_file(opt_vals["input_file"].as<std::string>());
        reader = mcf_reader;
    }
    else if (have_regex)
    {
        cf_reader->set_files_regex(opt_vals["input_regex"].as<std::string>());
        reader = cf_reader;
    }

    if (opt_vals.count("x_axis_variable"))
    {
        cf_reader->set_x_axis_variable(opt_vals["x_axis_variable"].as<std::string>());
        mcf_reader->set_x_axis_variable(opt_vals["x_axis_variable"].as<std::string>());
    }

    if (opt_vals.count("y_axis_variable"))
    {
        cf_reader->set_y_axis_variable(opt_vals["y_axis_variable"].as<std::string>());
        mcf_reader->set_y_axis_variable(opt_vals["y_axis_variable"].as<std::string>());
    }

    if (opt_vals.count("z_axis_variable"))
    {
        cf_reader->set_z_axis_variable(opt_vals["z_axis_variable"].as<std::string>());
        mcf_reader->set_z_axis_variable(opt_vals["z_axis_variable"].as<std::string>());
    }

    if (opt_vals.count("output_file"))
        cf_writer->set_file_name(
            opt_vals["output_file"].as<std::string>());

    if (opt_vals.count("point_arrays"))
        cf_writer->set_point_arrays(
            opt_vals["point_arrays"].as<std::vector<std::string>>());

    if (opt_vals.count("information_arrays"))
        cf_writer->set_information_arrays(
            opt_vals["information_arrays"].as<std::vector<std::string>>());

    if (opt_vals.count("steps_per_file"))
        cf_writer->set_steps_per_file(
            opt_vals["steps_per_file"].as<long>());

    if (opt_vals.count("first_step"))
        cf_writer->set_first_step(opt_vals["first_step"].as<long>());

    if (opt_vals.count("last_step"))
        cf_writer->set_last_step(opt_vals["last_step"].as<long>());

    if (opt_vals.count("bounds"))
        exec->set_bounds(
            opt_vals["bounds"].as<std::vector<double>>());

    if (opt_vals.count("verbose"))
    {
        cf_writer->set_verbose(1);
        exec->set_verbose(1);
    }

    if (opt_vals.count("n_threads"))
        cf_writer->set_thread_pool_size(opt_vals["n_threads"].as<int>());
    else
        cf_writer->set_thread_pool_size(1);

    // some minimal check for missing options
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

    if (cf_writer->get_file_name().empty())
    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_ERROR("missing file name pattern for the NetCDF CF writer. "
                "See --help for a list of command line options.")
        }
        return -1;
    }

    // if no point arrays were specified on the command line by default
    // write all point arrays
    teca_metadata md;
    teca_metadata atts;
    // TODO -- this will need some more work in the reader as currently
    // all arrays are marked as being point centered, but here we need
    // to identify only the arrays on the mesh.
    /*if (cf_writer->get_number_of_point_arrays() == 0)
    {
        // run the reporting phase of the pipeline
        if (md.empty())
            md = cf_reader->update_metadata();

        // if array attributes are present, use them to locate the set of
        // point centered arrrays
        if (atts.empty() && md.get("attributes", atts))
        {
            TECA_ERROR("metadata missing attributes")
            return -1;
        }

        // for each array check if it's point centered, if so add it to
        // the list of arrays to write.
        unsigned int n_arrays = atts.size();
        for (unsigned int i = 0; i < n_arrays; ++i)
        {
            std::string array_name;
            atts.get_name(i, array_name);

            teca_metadata array_atts;
            atts.get(array_name, array_atts);

            unsigned int array_cen = 0;
            array_atts.get("centering", array_cen);

            if (array_cen == teca_array_attributes::point_centering)
            {
                cf_writer->append_point_array(array_name);
            }
        }
    }*/

    // look for requested time step range, start
    bool parse_start_date = opt_vals.count("start_date");
    bool parse_end_date = opt_vals.count("end_date");
    if (parse_start_date || parse_end_date)
    {
        // run the reporting phase of the pipeline
        if (md.empty())
            md = cf_reader->update_metadata();

        if (atts.empty() && md.get("attributes", atts))
        {
            TECA_ERROR("metadata missing attributes")
            return -1;
        }

        teca_metadata time_atts;
        std::string calendar;
        std::string units;
        if (atts.get("time", time_atts)
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
            std::string start_date = opt_vals["start_date"].as<std::string>();
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
            std::string end_date = opt_vals["end_date"].as<std::string>();
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

    // connect the pipeline
    cf_writer->set_input_connection(reader->get_output_port());

    // run the pipeline
    cf_writer->set_executive(exec);
    cf_writer->update();

    return 0;
}
