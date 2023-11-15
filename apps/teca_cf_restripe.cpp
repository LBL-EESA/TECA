#include "teca_config.h"
#include "teca_metadata.h"
#include "teca_array_attributes.h"
#include "teca_variant_array.h"
#include "teca_cf_reader.h"
#include "teca_multi_cf_reader.h"
#include "teca_normalize_coordinates.h"
#include "teca_cartesian_mesh_regrid.h"
#include "teca_cartesian_mesh_source.h"
#include "teca_rename_variables.h"
#include "teca_cf_writer.h"
#include "teca_dataset_diff.h"
#include "teca_index_executive.h"
#include "teca_system_interface.h"
#include "teca_coordinate_util.h"
#include "teca_file_util.h"
#include "teca_mpi_manager.h"
#include "teca_mpi.h"
#include "teca_app_util.h"
#include "teca_calcalcs.h"

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

    int rank = mpi_man.get_comm_rank();
    int n_ranks = mpi_man.get_comm_size();

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
        ("output_file", value<std::string>(),
            "\nA path and file name pattern for the output NetCDF files. %t% is replaced with a"
            " human readable date and time corresponding to the time of the first time step in"
            " the file. Use --cf_writer::date_format to change the formatting\n")
        ("file_layout", value<std::string>()->default_value("monthly"),
            "\nSelects the size and layout of the set of output files. May be one of"
            " number_of_steps, daily, monthly, seasonal, or yearly. Files are structured"
            " such that each file contains one of the selected interval. For the number_of_steps"
            " option use --steps_per_file.\n")
        ("steps_per_file", value<long>(), "\nThe number of time steps per output file when "
            " --file_layout number_of_steps is specified.\n")

        ("normalize_coordinates", "\nEnable coordinate normalization pipeline stage\n")

        ("regrid", "\nEnable mesh regridding pipeline stage. When enabled requires --dims"
            " to be provided\n")
        ("dims", value<std::vector<unsigned long>>()->multitoken(),
            "\nA 3-tuple of values specifying the mesh size of the output dataset in the x, y,"
            " and z dimensions. The accepted format for dimensions is: nx ny nz\n")
        ("bounds", value<std::vector<double>>()->multitoken(),
            "\nA hex-tuple of low and high values specifying lon lat lev bounding box to subset"
            " the input dataset with. The accepted format for bounds is: x0 x1 y0 y1 z0 z1\n")

        ("rename", "\nEnable variable renaming stage\n")
        ("original_name", value<std::vector<std::string>>()->multitoken(),
            "\nA list of variables to rename. Use --new_name to set the new names\n")
        ("new_name", value<std::vector<std::string>>()->multitoken(),
            "\nThe new names to use when renaming variables. Use --original_name to set the"
            " list of variables to rename\n")

        ("enable_cb", "\nEnables collective buffering.\n")
        ("temporal_partitions", value<long>(), "\nThe number of temporal partitions to"
            " use in collective buffering mode. The default of 1 requires enough RAM to load the"
            " entire dataset. Increase to reduce memory pressure and stream the partitions"
            " sequentially\n")

        ("first_step", value<long>(), "\nfirst time step to process\n")
        ("last_step", value<long>(), "\nlast time step to process\n")
        ("start_date", value<std::string>(), "\nfirst time to proces in YYYY-MM-DD hh:mm:ss format\n")
        ("end_date", value<std::string>(), "\nfirst time to proces in YYYY-MM-DD hh:mm:ss format\n")
        ("n_threads", value<int>(), "\nSets the thread pool size on each MPI rank. A value of -1"
            " will coordinate across MPI ranks such that each thread is bound to a unique physical"
            " core.\n")
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
    // them.
    p_teca_cf_reader cf_reader = teca_cf_reader::New();
    cf_reader->get_properties_description("cf_reader", advanced_opt_defs);

    p_teca_multi_cf_reader mcf_reader = teca_multi_cf_reader::New();
    mcf_reader->get_properties_description("mcf_reader", advanced_opt_defs);

    p_teca_normalize_coordinates norm_coords = teca_normalize_coordinates::New();
    norm_coords->get_properties_description("norm_coords", advanced_opt_defs);

    p_teca_cartesian_mesh_regrid regrid = teca_cartesian_mesh_regrid::New();
    regrid->set_interpolation_mode_linear();
    regrid->get_properties_description("regrid", advanced_opt_defs);

    p_teca_cartesian_mesh_source regrid_src = teca_cartesian_mesh_source::New();
    regrid_src->get_properties_description("regrid_source", advanced_opt_defs);

    p_teca_rename_variables rename = teca_rename_variables::New();
    rename->get_properties_description("rename", advanced_opt_defs);

    p_teca_cf_writer cf_writer = teca_cf_writer::New();
    cf_writer->get_properties_description("cf_writer", advanced_opt_defs);
    cf_writer->set_layout(teca_cf_writer::monthly);
    cf_writer->set_threads_per_device(0); // CPU only

    // Add an executive for the writer
    p_teca_index_executive exec = teca_index_executive::New();

    // package basic and advanced options for display
    options_description all_opt_defs(help_width, help_width - 4);
    all_opt_defs.add(basic_opt_defs).add(advanced_opt_defs);

    // parse the command line
    int ierr = 0;
    variables_map opt_vals;
    if ((ierr = teca_app_util::process_command_line_help(
        rank, argc, argv, basic_opt_defs,
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
    regrid->set_properties("regrid", opt_vals);
    regrid_src->set_properties("regrid_source", opt_vals);
    rename->set_properties("rename", opt_vals);
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

    if (!opt_vals["file_layout"].defaulted() &&
        cf_writer->set_layout(opt_vals["file_layout"].as<std::string>()))
    {
        TECA_FATAL_ERROR("An invalid file layout was provided \""
            << opt_vals["file_layout"].as<std::string>() << "\"")
        return -1;
    }

    if (opt_vals.count("steps_per_file"))
        cf_writer->set_steps_per_file(
            opt_vals["steps_per_file"].as<long>());

    if (opt_vals.count("first_step"))
        cf_writer->set_first_step(opt_vals["first_step"].as<long>());

    if (opt_vals.count("last_step"))
        cf_writer->set_last_step(opt_vals["last_step"].as<long>());

    std::vector<double> bounds;
    bool have_bounds = opt_vals.count("bounds");
    if (have_bounds)
    {
        bounds = opt_vals["bounds"].as<std::vector<double>>();
        if (bounds.size() != 6)
        {
            TECA_FATAL_ERROR("An invlaid bounds specification was provided in"
                " --bounds, size != 6. Use: --bounds x0 x1 y0 y1 z0 z1")
            return -1;
        }
    }

    bool do_regrid = opt_vals.count("regrid");

    // when not regriding let the executive subset. when regriding
    // the regrid algorithm handles subsetting and the executive should
    // request the entire domain.
    if (have_bounds && !do_regrid)
        exec->set_bounds(bounds);

    // when regriding target mesh dimensions must be provided
    std::vector<unsigned long> dims;
    if (do_regrid)
    {
        if (opt_vals.count("dims"))
        {
            dims = opt_vals["dims"].as<std::vector<unsigned long>>();
            if (dims.size() != 3)
            {
                TECA_FATAL_ERROR("An invlaid dimension specification was provided in"
                    " --dims, size != 3. Use: --dims nx ny nz")
                return -1;
            }
        }
        else
        {
            TECA_FATAL_ERROR("The --regrid option requires that --dims"
                " also be specified")
            return -1;
        }
    }

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
        if (rank == 0)
        {
            TECA_FATAL_ERROR("Extacly one of --input_file or --input_regex can be specified. "
                "Use --input_file to activate the multi_cf_reader (HighResMIP datasets) "
                "and --input_regex to activate the cf_reader (CAM like datasets)")
        }
        return -1;
    }

    if (cf_writer->get_file_name().empty())
    {
        if (rank == 0)
        {
            TECA_FATAL_ERROR("missing file name pattern for the NetCDF CF writer. "
                "See --help for a list of command line options.")
        }
        return -1;
    }

    // add the normalize coordinates stage before accessing metadata
    p_teca_algorithm head = reader;
    if (opt_vals.count("normalize_coordinates"))
    {
        if (rank == 0)
            TECA_STATUS("Added cooridnate normalization stage");

        norm_coords->set_input_connection(reader->get_output_port());
        head = norm_coords;
    }

    // if no point arrays were specified on the command line by default
    // write all point arrays
    teca_metadata md;
    teca_metadata atts;
    teca_metadata time_atts;
    std::string calendar;
    std::string units;
    // TODO -- this will need some more work in the reader as currently
    // all arrays are marked as being point centered, but here we need
    // to identify only the arrays on the mesh.
    /*if (cf_writer->get_number_of_point_arrays() == 0)
    {
        // run the reporting phase of the pipeline
        if (md.empty())
            md = head->update_metadata();

        // if array attributes are present, use them to locate the set of
        // point centered arrrays
        if (atts.empty() && md.get("attributes", atts))
        {
            TECA_FATAL_ERROR("metadata missing attributes")
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
            md = head->update_metadata();

        if (atts.empty() && md.get("attributes", atts))
        {
            if (rank == 0)
                TECA_FATAL_ERROR("metadata missing attributes")
            return -1;
        }

        if (atts.get("time", time_atts)
           || time_atts.get("calendar", calendar)
           || time_atts.get("units", units))
        {
            if (rank == 0)
                TECA_FATAL_ERROR("failed to determine the calendaring parameters")
            return -1;
        }

        teca_metadata coords;
        p_teca_variant_array time;
        if (md.get("coordinates", coords) || !(time = coords.get("t")))
        {
            if (rank == 0)
                TECA_FATAL_ERROR("failed to determine time coordinate")
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
                if (rank == 0)
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
            std::string end_date = opt_vals["end_date"].as<std::string>();
            if (teca_coordinate_util::time_step_of(time, false, true, calendar,
                 units, end_date, last_step))
            {
                if (rank == 0)
                    TECA_FATAL_ERROR("Failed to locate time step for end date \""
                        <<  end_date << "\"")
                return -1;
            }
            cf_writer->set_last_step(last_step);
        }
    }

    // set up regriding
    if (do_regrid)
    {
        if (rank == 0)
            TECA_STATUS("Added regrid stage");

        // run the reporting phase of the pipeline, the resulting metadata
        // can be used to automatically determine the calendaring parameters
        // and spatial bounds
        if (md.empty())
            md = head->update_metadata();

        // use the calendar and time axis of the input dataset
        if (regrid_src->set_t_axis_variable(md) || regrid_src->set_t_axis(md))
        {
            if (rank == 0)
                TECA_WARNING("Failed to determine the time axis, assuming a"
                    " single time step")

            p_teca_double_array t = teca_double_array::New(1);
            regrid_src->set_t_axis_variable("");
            regrid_src->set_t_axis(t);
        }

        // to construct the target mesh we need bounds.  if no bounds are
        // specified on the command line use those of the input dataset and
        // error out if that fails
        if (have_bounds)
        {
            // extend to include time
            bounds.resize(8, 0.0);
            regrid_src->set_bounds(bounds);
        }
        else
        {
            // try to determine the bounds from the input mesh metadata
            if (regrid_src->set_spatial_bounds(md))
            {
                if (rank == 0)
                    TECA_FATAL_ERROR("Failed to determine target mesh bounds from the"
                        " input metadata. Use --bounds to specify them manually.")
                return -1;
            }
        }

        // set the target mesh dimensions
        regrid_src->set_whole_extents({0lu, dims[0] - 1lu,
            0lu, dims[1] - 1lu, 0lu, dims[2] - 1lu, 0lu, 0lu});

        // connect to the pipeline
        regrid->set_input_connection(0, regrid_src->get_output_port());
        regrid->set_input_connection(1, head->get_output_port());
        head = regrid;
    }

    // add rename stage
    if (opt_vals.count("rename"))
    {
        if (!opt_vals.count("original_name"))
        {
            if (rank == 0)
                TECA_FATAL_ERROR("--original_name is required when renaming variables")
            return -1;
        }

        std::vector<std::string> original_name =
            opt_vals["original_name"].as<std::vector<std::string>>();

        if (!opt_vals.count("new_name"))
        {
            if (rank == 0)
                TECA_FATAL_ERROR("--new_name is required when renaming variables")
            return -1;
        }

        std::vector<std::string> new_name =
            opt_vals["new_name"].as<std::vector<std::string>>();

        if (original_name.size() != new_name.size())
        {
            if (rank == 0)
                TECA_FATAL_ERROR("--original_name and --new_name must have the same"
                    " number of values")
            return -1;

        }

        if (rank == 0)
            TECA_STATUS("Added rename stage");

        rename->set_input_connection(head->get_output_port());
        rename->set_original_variable_names(original_name);
        rename->set_new_variable_names(new_name);

        head = rename;
    }

    // enable collective buffering
    if (opt_vals.count("enable_cb"))
    {
        if (rank == 0)
            TECA_STATUS("Enabled collective buffer mode")

        cf_reader->set_collective_buffer(1);
        mcf_reader->set_collective_buffer(1);

        cf_writer->set_partitioner(teca_cf_writer::spatial);
        cf_writer->set_number_of_spatial_partitions(n_ranks);
        cf_writer->set_collective_buffer(1);

        if (opt_vals.count("temporal_partitions"))
        {
            cf_writer->set_number_of_temporal_partitions
                (opt_vals["temporal_partitions"].as<long>());
        }
        else
        {
            cf_writer->set_number_of_temporal_partitions(1);
        }
    }

    // add the writer last
    cf_writer->set_input_connection(head->get_output_port());

    // run the pipeline
    cf_writer->set_executive(exec);
    cf_writer->update();

    return 0;
}
