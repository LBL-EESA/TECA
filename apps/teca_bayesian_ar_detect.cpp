#include "teca_config.h"
#include "teca_cf_reader.h"
#include "teca_cf_writer.h"
#include "teca_index_executive.h"
#include "teca_normalize_coordinates.h"
#include "teca_metadata.h"
#include "teca_bayesian_ar_detect.h"
#include "teca_bayesian_ar_detector_parameters.h"
#include "teca_mpi_manager.h"
#include "teca_coordinate_util.h"
#include "teca_table.h"
#include "teca_dataset_source.h"
#include "calcalcs.h"

#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <boost/program_options.hpp>

using namespace std;

using boost::program_options::value;

using seconds_t =
    std::chrono::duration<double, std::chrono::seconds::period>;

// --------------------------------------------------------------------------
int main(int argc, char **argv)
{
    // initialize mpi
    teca_mpi_manager mpi_man(argc, argv);

    std::chrono::high_resolution_clock::time_point t0, t1;
    if (mpi_man.get_comm_rank() == 0)
        t0 = std::chrono::high_resolution_clock::now();

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
        ("output_file", value<string>()->default_value(std::string("bayesian_ar_detect_%t%.%e%")), "file pattern for output netcdf files (%t% is the time index and %e% is the netCDF suffix (nc))")
        ("input_regex", value<string>(), "regex matching simulation files to search for atmospheric rivers")
        ("ivt", value<string>()->default_value(std::string("IVT")), "name of variable with integrated vapor transport (IVT)")
        ("first_step", value<long>(), "first time step to process")
        ("last_step", value<long>(), "last time step to process")
        ("start_date", value<string>(), "first time to proces in YYYY-MM-DD hh:mm:ss format")
        ("end_date", value<string>(), "first time to proces in YYYY-MM-DD hh:mm:ss format")
        ("n_threads", value<int>(), "thread pool size. default is 1. -1 for all")
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

    p_teca_normalize_coordinates sim_coords = teca_normalize_coordinates::New();
    sim_coords->set_input_connection(cf_reader->get_output_port());
    
    /* Load detector parameters into a data table */
    p_teca_variant_array_impl<parameter_t> min_water_vapor = teca_variant_array_impl<parameter_t>::New(num_ar_detector_parameters);
    p_teca_variant_array_impl<parameter_t> filter_lat_width = teca_variant_array_impl<parameter_t>::New(num_ar_detector_parameters);
    p_teca_variant_array_impl<parameter_t> min_area_kmsq = teca_variant_array_impl<parameter_t>::New(num_ar_detector_parameters);
    
    for (unsigned long n = 0; n < num_ar_detector_parameters; n++){
        min_water_vapor->get()[n] = quantile_array[n];
        filter_lat_width->get()[n] = filter_lat_width_array[n];
        min_area_kmsq->get()[n] = min_area_kmsq_array[n];
    }

    p_teca_table tab = teca_table::New();
    tab->append_column("hwhm_latitude", filter_lat_width);
    tab->append_column("min_water_vapor", min_water_vapor);
    tab->append_column("min_component_area", min_area_kmsq);

    teca_metadata table_md;
    table_md.set("number_of_tables", 1);
    table_md.set("index_initializer_key", std::string("number_of_tables"));
    table_md.set("index_request_key", std::string("table_id"));

    p_teca_dataset_source dss = teca_dataset_source::New();
    dss->set_metadata(table_md);
    dss->set_dataset(tab);
    
    // Construct the AR detector and attach the input file and parameters
    p_teca_bayesian_ar_detect ar_detect = teca_bayesian_ar_detect::New();
    ar_detect->get_properties_description("ar_detect", advanced_opt_defs);
    ar_detect->set_input_connection(0, dss->get_output_port());
    ar_detect->set_input_connection(1, sim_coords->get_output_port());

    // Add an executive for the writer
    p_teca_index_executive exec = teca_index_executive::New();
    
        
    // Add the writer
    p_teca_cf_writer cf_writer = teca_cf_writer::New();
    cf_writer->get_properties_description("cf_writer", advanced_opt_defs);
    cf_writer->set_input_connection(ar_detect->get_output_port());
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
    cf_writer->set_properties("cf_writer", opt_vals);

    // now pass in the basic options, these are processed
    // last so that they will take precedence
    if (opt_vals.count("input_file"))
        cf_reader->append_file_name(
            opt_vals["input_file"].as<string>());

    if (opt_vals.count("input_regex"))
        cf_reader->set_files_regex(
            opt_vals["input_regex"].as<string>());
    
    if (opt_vals.count("output_file"))
        cf_writer->set_file_name(
            opt_vals["output_file"].as<string>());

    if (opt_vals.count("ivt"))
        ar_detect->set_water_vapor_variable(
            opt_vals["ivt"].as<string>());

    if (opt_vals.count("first_step"))
        exec->set_start_index(opt_vals["first_step"].as<long>());

    if (opt_vals.count("last_step"))
        exec->set_end_index(opt_vals["last_step"].as<long>());

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
            TECA_ERROR("metadata mising attributes")
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
                TECA_ERROR("Failed to lcoate time step for start date \""
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
                TECA_ERROR("Failed to lcoate time step for end date \""
                    <<  end_date << "\"")
                return -1;
            }
            exec->set_end_index(last_step);
        }
    }
    
    // set up the array request
    vector<string> arrays;
    arrays.push_back(std::string("ar_probability"));
    exec->set_arrays(arrays);
    exec->set_verbose(1);
    
    // set up metadata
    teca_metadata cf_metadata;
    teca_metadata ar_probability_metadata;
    
    // set metadata for the 'ar_probability' variable
    ar_probability_metadata.set("long_name", std::string("posterior AR flag"));
    ar_probability_metadata.set("units", std::string("probability"));
    // load the metadata into the master array
    cf_metadata.set("ar_probability", ar_probability_metadata);
    
    // override the incoming metadata for the executive
    //cf_writer->get_metadata() = &cf_metadata;
        
        
    // run the pipeline
    cf_writer->set_executive(exec);
    cf_writer->update();

    if (mpi_man.get_comm_rank() == 0)
    {
        t1 = std::chrono::high_resolution_clock::now();
        seconds_t dt(t1 - t0);
        TECA_STATUS("teca_bayesian_ar_detect run_time=" << dt.count() << " sec")
    }

    return 0;
}
