#include "teca_config.h"
#include "teca_metadata.h"
#include "teca_multi_cf_reader.h"
#include "teca_cf_writer.h"
#include "teca_integrated_vapor_transport.h"
#include "teca_index_executive.h"
#include "teca_system_interface.h"
#include "teca_coordinate_util.h"
#include "teca_mpi_manager.h"
#include "teca_mpi.h"

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

using boost::program_options::value;


int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    teca_system_interface::set_stack_trace_on_error();
    teca_system_interface::set_stack_trace_on_mpi_error();

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
        ("readers", value<std::vector<std::string>>()->multitoken()->composing(),
            "readers for multiple NetCDF CF2 files")
        ("output_file", value<std::string>(), "file path for output NetCDF CF2 dataset")
        ("wind_u_var", value<std::string>(),
            "name of the variable that contains the longitudinal component of the wind vector (ua)")
        ("wind_v_var", value<std::string>(),
            "name of the variable that contains the latitudinal component of the wind vector (va)")
        ("specific_humidity_var", value<std::string>(),
            "name of the variable that contains the specific humidity (hus)")
        ("time_reader", value<std::string>(), "name of reader that provides time axis")
        ("geometry_reader", value<std::string>(), "name of reader the provides mesh geometry")
        ("bounds", value<std::vector<double>>()->multitoken(), "lat lon lev bounding box to subset with")
        ("first_step", value<long>(), "first time step to process")
        ("last_step", value<long>(), "last time step to process")
        ("steps_per_file", value<long>(), "number of time steps per output file")
        ("start_date", value<std::string>(), "first time to proces in YYYY-MM-DD hh:mm:ss format")
        ("end_date", value<std::string>(), "first time to proces in YYYY-MM-DD hh:mm:ss format")
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
    p_teca_multi_cf_reader multi_cf_reader = teca_multi_cf_reader::New();
    multi_cf_reader->get_properties_description("multi_cf_reader", advanced_opt_defs);
    multi_cf_reader->set_z_axis_variable("plev");

    p_teca_integrated_vapor_transport ivt = teca_integrated_vapor_transport::New();
    ivt->get_properties_description("ivt", advanced_opt_defs);
    ivt->set_input_connection(multi_cf_reader->get_output_port());

    p_teca_cf_writer cf_writer = teca_cf_writer::New();
    cf_writer->get_properties_description("cf_writer", advanced_opt_defs);
    cf_writer->set_input_connection(ivt->get_output_port());
    cf_writer->set_point_arrays({"ivt_u", "ivt_v"});
    cf_writer->set_thread_pool_size(1);

    // Add an executive for the writer
    p_teca_index_executive exec = teca_index_executive::New();

    // package basic and advanced options for display
    options_description all_opt_defs(-1, -1);
    all_opt_defs.add(basic_opt_defs).add(advanced_opt_defs);

    // parse the command line
    variables_map opt_vals;
    try
    {
        boost::program_options::store(
            boost::program_options::command_line_parser(argc, argv)
                .style(boost::program_options::command_line_style::unix_style ^
                       boost::program_options::command_line_style::allow_short)
                .options(all_opt_defs)
                .run(),
            opt_vals);

        if (mpi_man.get_comm_rank() == 0)
        {
            if (opt_vals.count("help"))
            {
                std::cerr << std::endl
                    << "usage: teca_ivt_computation [options]" << std::endl
                    << std::endl
                    << basic_opt_defs << std::endl
                    << std::endl;
                return -1;
            }
            if (opt_vals.count("advanced_help"))
            {
                std::cerr << std::endl
                    << "usage: teca_ivt_computation [options]" << std::endl
                    << std::endl
                    << advanced_opt_defs << std::endl
                    << std::endl;
                return -1;
            }

            if (opt_vals.count("full_help"))
            {
                std::cerr << std::endl
                    << "usage: teca_ivt_computation [options]" << std::endl
                    << std::endl
                    << all_opt_defs << std::endl
                    << std::endl;
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
    multi_cf_reader->set_properties("multi_cf_reader", opt_vals);
    ivt->set_properties("ivt", opt_vals);
    cf_writer->set_properties("cf_writer", opt_vals);

    // now pass in the basic options, these are processed
    // last so that they will take precedence
    std::vector<std::string> readers;
    if (opt_vals.count("readers"))
        readers = opt_vals["readers"].as<std::vector<std::string>>();

    if (opt_vals.count("output_file"))
        cf_writer->set_file_name(
            opt_vals["output_file"].as<std::string>());

    if (opt_vals.count("wind_u_var"))
        ivt->set_wind_u_variable(
            opt_vals["wind_u_var"].as<std::string>());

    if (opt_vals.count("wind_v_var"))
        ivt->set_wind_v_variable(
            opt_vals["wind_v_var"].as<std::string>());

    if (opt_vals.count("specific_humidity_var"))
        ivt->set_specific_humidity_variable(
            opt_vals["specific_humidity_var"].as<std::string>());

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

    // some minimal check for missing options
    if (readers.size() == 0)
    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_ERROR("missing readers for NetCDF CF files. "
                "See --help for a list of command line options.")
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

    teca_metadata md;
    teca_metadata atts;

    // look for requested time step range, start
    bool parse_start_date = opt_vals.count("start_date");
    bool parse_end_date = opt_vals.count("end_date");
    if (parse_start_date || parse_end_date)
    {
        // run the reporting phase of the pipeline
        if (md.empty())
            md = multi_cf_reader->update_metadata();

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

    // Add readers
    for (unsigned int i = 0; i < readers.size(); ++i)
    {
        int provides_time_geo = i ? 0 : 1;

        std::vector<std::string> reader;
        boost::algorithm::split(reader, readers[i], [](char c){return c == ',';});

        multi_cf_reader->add_reader(
            reader[0], reader[1], provides_time_geo, provides_time_geo,
            std::vector<std::string>(reader.begin() + 2, reader.end())
            );
    }

    if (opt_vals.count("time_reader"))
        multi_cf_reader->set_time_reader(opt_vals["time_reader"].as<std::string>());

    if (opt_vals.count("geometry_reader"))
        multi_cf_reader->set_geometry_reader(opt_vals["geometry_reader"].as<std::string>());

    // run the pipeline
    cf_writer->set_executive(exec);
    cf_writer->update();

    return 0;
}
