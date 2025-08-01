#include "teca_cf_reader.h"
#include "teca_multi_cf_reader.h"
#include "teca_normalize_coordinates.h"
#include "teca_l2_norm.h"
#include "teca_derived_quantity.h"
#include "teca_derived_quantity_numerics.h"
#include "teca_mpi_manager.h"
#include "teca_app_util.h"
#include "teca_table_reduce.h"
#include "teca_table_sort.h"
#include "teca_table_calendar.h"
#include "teca_table_writer.h"
#include "teca_cartesian_mesh_regrid.h"
#include "teca_cartesian_mesh_source.h"
#include "teca_rename_variables.h"
#include "teca_mesh_join.h"
#include "teca_detect_nodes.h"
#include "teca_stitch_nodes.h"

#include <string>
#include <iostream>
#include <boost/program_options.hpp>

using namespace std;
using namespace teca_derived_quantity_numerics;

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

        ("candidate_file", value<string>()->default_value("candidates.csv"),
            "\nfile path to write the storm candidates to. The extension determines"
            " the file format. May be one of `.nc`, `.csv`, or `.bin`\n")

        ("track_file", value<string>()->default_value("tracks.csv"),
            "\nfile path to write the storm tracks to. The extension determines"
            " the file format. May be one of `.nc`, `.csv`, or `.bin`\n")

        ("x_axis_variable", value<std::string>()->default_value("lon"),
            "\nName of the variable to use for x-coordinates.\n")

        ("y_axis_variable", value<std::string>()->default_value("lat"),
            "\nName of the variable to use for y-coordinates.\n")

        ("z_axis_variable", value<std::string>()->default_value("level"),
            "\nName of the variable to use for z-coordinates.\n")

        ("in_connect", value<string>()->default_value(""),
            "\nConnectivity file that describes the unstructured grid\n")
        ("search_by_min", value<string>()->default_value(""),
            "\nThe input variable to use for initially selecting candidate points"
            " (defined as local minima). At least one (and at most one)"
            " of --search_by_min or --search_by_max must be specified.\n")
        ("search_by_max", value<string>()->default_value(""),
            "\nThe input variable to use for initially selecting candidate points"
            " (defined as local maxima). At least one (and at most one)"
            " of --search_by_min or --search_by_max must be specified.\n")
        ("closed_contour_cmd", value<string>()->default_value(""),
            "\nEliminate candidates if they do not have a closed contour."
            " The closed contour is determined by breadth first search:"
            " if any paths exist from the candidate point"
            " (or nearby minima/maxima if minmaxdist is specified)"
            " that reach the specified distance before achieving the specified delta"
            " then we say no closed contour is present."
            " Closed contour commands are separated by a semicolon (\"<cmd1>;<cmd2>;...\")."
            " Each closed contour command takes the form \"var,delta,dist,minmaxdist\"."
            " These arguments are as follows."
            " var is the name of the variable used for the contour search."
            " dist is the great-circle distance (in degrees) from the pivot"
            " within which the closed_contour criteria must be satisfied."
            " delta is the amount by which the field must change from the pivot value."
            " If positive (negative) the field must increase (decrease) by this value along the contour."
            " minmaxdist is the great-circle distance away from the candidate"
            " to search for the minima/maxima. If delta is positive (negative),"
            " the pivot is a local minimum (maximum).\n")
        ("no_closed_contour_cmd", value<string>()->default_value(""),
            "\nAs --closed_contour_cmd,"
            " except it eliminates candidates if a closed contour is present.\n")
        ("candidate_threshold_cmd", value<string>()->default_value(""),
            "\nThreshold commands for candidates."
            " Eliminate candidates that do not satisfy a threshold criteria"
            " (there must exist a point within a given distance of the candidate"
            " that satisfies a given equality or inequality)."
            " Search is performed by breadth-first search over the grid."
            " Threshold commands are separated by a semicolon (\"<cmd1>;<cmd2>;...\")."
            " Each threshold command takes the form \"var,op,value,dist\"."
            " These arguments are as follows."
            " var is the name of the variable used for the thresholding."
            " op is the operator that must be satisfied for threshold (options include >,>=,<,<=,=,!=)."
            " value is the value on the right-hand-side of the comparison."
            " dist is the great-circle distance away from the candidate"
            " to search for a point that satisfies the threshold.\n")
        ("output_cmd", value<string>()->default_value(""),
            "\nInclude additional columns in the candidates output file."
            " Each output command takes the form \"var,op,dist\"."
            " These arguments are as follows."
            " var is the name of the variable used for output."
            " op is the operator that is applied over all points"
            " within the specified distance of the candidate (options include max, min, avg, maxdist, mindist)."
            " dist is the great-circle distance away from the candidate wherein the operator is applied.\n")
        ("search_by_threshold", value<string>()->default_value(""),
            "\nThreshold for search operation in the form \"<op><value>\""
            " These arguments are as follows."
            " op is the operator that must be satisfied for threshold (options include >,>=,<,<=,=,!=)."
            " value is the value on the right-hand-side of the comparison.\n")
        ("min_lon", value<double>()->default_value(1.0),
            "\nMinimum longitude in degrees for detection\n")
        ("max_lon", value<double>()->default_value(0.0),
            "\nMaximum longitude in degrees for detection"
            " As longitude is a periodic dimension,"
            " when --regional is not specified --min_lon may be larger than --max_lon."
            " If --max_lon and --min_lon are equal then these arguments are ignored.\n")
        ("min_lat", value<double>()->default_value(1.0),
            "\nMinimum latitude in degrees for detection\n")
        ("max_lat", value<double>()->default_value(0.0),
            "\nMaximum latitude in degrees for detection"
            " If --max_lat and --min_lat are equal then these arguments are ignored.\n")
        ("merge_dist", value<double>()->default_value(6.0),
            "\nMinimum allowable distance between two candidates in degrees."
            " Candidate points with a distance (in degrees great-circle-distance)"
            " shorter than the specified value are merged."
            " Among two candidates within the merge distance,"
            " only the candidate with the lowest value of the --search_by_min field"
            " or highest value of the --search_by_max field are retained.\n")
        ("diag_connect", value<bool>()->default_value(false),
            "\nWhen the data is on a structured grid,"
            " consider grid cells to be connected in the diagonal (across the vertex)\n")
        ("regional", value<bool>()->default_value(true),
            "\nUsed to indicate that a given latitude-longitude grid"
            " should not be periodic in the longitudinal direction.\n")
        ("out_header", value<bool>()->default_value(true),
            "\nIf present, output a header at the beginning of the output file"
            " indicating the columns of the file.\n")

        ("in_fmt", value<string>()->default_value(""),
            "\nA comma-separated list of names of the auxiliary columns"
            " within the candidates output file."
            " (namely, the list must not include the time columns).\n")
        ("min_time", value<string>()->default_value("10"),
            "\nThe minimum length of a path either in terms of"
            " number of discrete times or as a duration, e.g. \"24h\"."
            " Note that the duration of a path is computed as"
            " the difference between the final time and initial time,"
            " so a \"24h\" duration correspond to 5 time steps in 6-hourly data"
            " (i.e. 0h,6,12,18,24UTC).\n")
        ("cal_type", value<string>()->default_value("standard"),
            "\nCalendar type\n")
        ("max_gap", value<string>()->default_value("3"),
            "\nMaximum time gap (in time steps or duration)"
            " The number of allowed missing points between spatially proximal candidate nodes"
            " while still considering them part of the same path.\n")
        ("track_threshold_cmd", value<string>()->default_value(""),
            "\nThreshold commands for path"
            " Filter paths based on the number of times where a particular threshold is satisfied."
            " Threshold commands are separated by a semicolon (\"<cmd1>;<cmd2>;...\")."
            " Each threshold command takes the form \"col,op,value,count\"."
            " These arguments are as follows."
            " col is the name of the column to use for thresholding, as specified in --in_fmt."
            " op is the operator that must be satisfied for threshold (options include >,>=,<,<=,=,!=,|>=,|<=)."
            " value is the value on the right-hand-side of the comparison."
            " count is either the minimum number of time slices where the threshold must be satisfied"
            " or the instruction \"all\", \"first\", or \"last\"."
            " Here \"all\" is used to indicate the threshold must be satisfied at all points along the path,"
            " \"first\" is used to indicate the threshold must be satisfied only at the first point along the path, and"
            " \"last\" is used to indicate the threshold must be satisfied only at the last point along the path.\n")
        ("prioritize", value<string>()->default_value(""),
            "\nVariable to use when prioritizing paths\n")
        ("min_path_length", value<int>()->default_value(0),
            "\nMinimum path length\n")
        ("range", value<double>()->default_value(8.0),
            "\nThe maximum distance between candidates"
            " along a path (in great-circle degrees).\n")
        ("min_endpoint_distance", value<double>()->default_value(0.0),
            "\nThe minimum great-circle distance between the first candidate"
            " on a path and the last candidate (in degrees).\n")
        ("min_path_distance", value<double>()->default_value(0.0),
            "\nThe minimum accumulated great-circle distance between nodes in a path (in degrees).\n")
        ("allow_repeated_times", value<bool>()->default_value(false),
            "\nAllow repeated times\n")

        ("sea_level_pressure", value<string>()->default_value(""),
            "\nname of variable with sea level pressure\n")
        ("geopotential_at_surface", value<string>()->default_value(""),
            "\nname of variable with geopotential at the surface\n")
        ("geopotential", value<string>()->default_value(""),
            "\nname of variable with geopotential (or geopotential height)"
            " for thickness calc\n")
        ("500mb_height", value<string>()->default_value(""),
            "\nname of variable with 500mb height for thickness calc\n")
        ("300mb_height", value<string>()->default_value(""),
            "\nname of variable with 300mb height for thickness calc\n")
        ("surface_wind_u", value<string>()->default_value(""),
            "\nname of variable with surface wind x-component\n")
        ("surface_wind_v", value<string>()->default_value(""),
            "\nname of variable with surface wind y-component\n")

        ("first_step", value<long>()->default_value(0),
            "\nfirst time step to process\n")
        ("last_step", value<long>()->default_value(-1),
            "\nlast time step to process\n")
        ("time_filter", value<long>()->default_value(1),
            "\nstride to process time steps at\n")
        ("n_threads", value<int>()->default_value(-1),
            "\nSets the thread pool size on each MPI rank."
            " When the default value of -1 is used TECA will coordinate the thread"
            " pools across ranks such each thread is bound to a unique physical core.\n")

        ("verbose", value<int>()->default_value(0),
            "\nUse 1 to enable verbose mode, otherwise 0.\n")
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
    p_teca_multi_cf_reader mcf_reader = teca_multi_cf_reader::New();
    p_teca_normalize_coordinates sim_coords = teca_normalize_coordinates::New();
    p_teca_cartesian_mesh_source regrid_src_1 = teca_cartesian_mesh_source::New();
    p_teca_cartesian_mesh_source regrid_src_2 = teca_cartesian_mesh_source::New();
    p_teca_cartesian_mesh_regrid regrid_1 = teca_cartesian_mesh_regrid::New();
    p_teca_cartesian_mesh_regrid regrid_2 = teca_cartesian_mesh_regrid::New();
    p_teca_rename_variables rename_1 = teca_rename_variables::New();
    p_teca_rename_variables rename_2 = teca_rename_variables::New();
    p_teca_mesh_join join = teca_mesh_join::New();
    p_teca_derived_quantity thickness = teca_derived_quantity::New();
    p_teca_l2_norm surf_wind = teca_l2_norm::New();
    p_teca_detect_nodes candidates = teca_detect_nodes::New();
    p_teca_table_reduce map_reduce = teca_table_reduce::New();
    p_teca_table_sort sort = teca_table_sort::New();
    p_teca_table_writer candidate_writer = teca_table_writer::New();
    p_teca_stitch_nodes tracks = teca_stitch_nodes::New();
    p_teca_table_calendar calendar = teca_table_calendar::New();
    p_teca_table_writer track_writer = teca_table_writer::New();

    thickness->set_dependent_variables({"Z500", "Z300"});
    thickness->set_derived_variable("thickness");
    surf_wind->set_component_0_variable("VAR_10U");
    surf_wind->set_component_1_variable("VAR_10V");
    surf_wind->set_l2_norm_variable("surface_wind_speed");

    cf_reader->get_properties_description("cf_reader", advanced_opt_defs);
    mcf_reader->get_properties_description("mcf_reader", advanced_opt_defs);
    regrid_src_1->get_properties_description("regrid_source_1", advanced_opt_defs);
    regrid_src_2->get_properties_description("regrid_source_2", advanced_opt_defs);
    regrid_1->get_properties_description("regrid_1", advanced_opt_defs);
    regrid_2->get_properties_description("regrid_2", advanced_opt_defs);
    rename_1->get_properties_description("rename_1", advanced_opt_defs);
    rename_2->get_properties_description("rename_2", advanced_opt_defs);
    join->get_properties_description("join", advanced_opt_defs);
    thickness->get_properties_description("thickness", advanced_opt_defs);
    surf_wind->get_properties_description("surface_wind_speed", advanced_opt_defs);
    candidates->get_properties_description("candidates", advanced_opt_defs);
    map_reduce->get_properties_description("map_reduce", advanced_opt_defs);
    sort->get_properties_description("sort", advanced_opt_defs);
    candidate_writer->get_properties_description("candidate_writer", advanced_opt_defs);
    tracks->get_properties_description("tracks", advanced_opt_defs);
    calendar->get_properties_description("calendar", advanced_opt_defs);
    track_writer->get_properties_description("track_writer", advanced_opt_defs);

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
    regrid_src_1->set_properties("regrid_source_1", opt_vals);
    regrid_src_2->set_properties("regrid_source_2", opt_vals);
    regrid_1->set_properties("regrid_1", opt_vals);
    regrid_2->set_properties("regrid_2", opt_vals);
    rename_1->set_properties("rename_1", opt_vals);
    rename_2->set_properties("rename_2", opt_vals);
    join->set_properties("join", opt_vals);
    thickness->set_properties("thickness", opt_vals);
    surf_wind->set_properties("surface_wind_speed", opt_vals);
    candidates->set_properties("candidates", opt_vals);
    map_reduce->set_properties("map_reduce", opt_vals);
    sort->set_properties("sort", opt_vals);
    candidate_writer->set_properties("candidate_writer", opt_vals);
    tracks->set_properties("tracks", opt_vals);
    calendar->set_properties("calendar", opt_vals);
    track_writer->set_properties("track_writer", opt_vals);

    // now pass in the basic options, these are processed
    // last so that they will take precedence

    // configure the reader
    bool have_file = opt_vals.count("input_file");
    bool have_regex = opt_vals.count("input_regex");
    // some minimal check for missing options
    if ((have_file && have_regex) || !(have_file || have_regex))
    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_FATAL_ERROR("Exactly one of --input_file or --input_regex can be specified. "
                "Use --input_file to activate the multi_cf_reader (HighResMIP datasets) "
                "and --input_regex to activate the cf_reader (CAM like datasets)")
        }
        return -1;
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

    if (!opt_vals["z_axis_variable"].defaulted())
    {
        cf_reader->set_z_axis_variable(opt_vals["z_axis_variable"].as<string>());
        mcf_reader->set_z_axis_variable(opt_vals["z_axis_variable"].as<string>());
    }

    if (opt_vals.count("input_file"))
    {
        mcf_reader->set_input_file(opt_vals["input_file"].as<string>());
        mcf_reader->set_validate_time_axis(0);
        sim_coords->set_input_connection(mcf_reader->get_output_port());
    }
    else if (opt_vals.count("input_regex"))
    {
        cf_reader->set_files_regex(opt_vals["input_regex"].as<string>());
        sim_coords->set_input_connection(cf_reader->get_output_port());
    }
    p_teca_algorithm head = sim_coords;

    if (!opt_vals["in_connect"].defaulted())
    {
        candidates->set_in_connect(opt_vals["in_connect"].as<string>());
        tracks->set_in_connect(opt_vals["in_connect"].as<string>());
    }

    if (!opt_vals["search_by_min"].defaulted())
    {
       candidates->set_search_by_min(opt_vals["search_by_min"].as<string>());
    }

    if (!opt_vals["search_by_max"].defaulted())
    {
       candidates->set_search_by_max(opt_vals["search_by_max"].as<string>());
    }

    if (opt_vals["search_by_min"].as<string>() == "" && opt_vals["search_by_max"].as<string>() == "")
    {
       if (opt_vals["sea_level_pressure"].as<string>() == "")
          TECA_FATAL_ERROR("Missing name of variable with sea level pressure")
       else
          candidates->set_search_by_min(opt_vals["sea_level_pressure"].as<string>());
    }

    //thickness calculation
    if (opt_vals["geopotential"].as<string>() == "")
    {
       if (opt_vals["300mb_height"].as<string>() == "")
       {
          TECA_FATAL_ERROR("Missing name of variable with 300mb height"
                " or with geopotential for thickness calc")
       }
       else
       {
          thickness->set_dependent_variable(1, opt_vals["300mb_height"].as<string>());
       }
       if (opt_vals["500mb_height"].as<string>() == "")
       {
          TECA_FATAL_ERROR("Missing name of variable with 500mb height"
                " or with geopotential for thickness calc")
       }
       else
       {
          thickness->set_dependent_variable(0, opt_vals["500mb_height"].as<string>());
       }
    }
    else
    {
       teca_metadata md = sim_coords->update_metadata();
       teca_metadata coords;
       md.get("coordinates", coords);
       const_p_teca_variant_array x = coords.get("x");
       const_p_teca_variant_array y = coords.get("y");
       double bounds[6] = {0.0};
       md.get("bounds", bounds, 6);

       //slice variables at pressure level of 300mb
       regrid_src_1->set_bounds({bounds[0], bounds[1], bounds[2], bounds[3], 300, 300, 0, 0});
       regrid_src_1->set_whole_extents({0lu, x->size() - 1lu, 0lu, y->size() - 1lu, 0lu, 0lu, 0lu, 0lu});
       regrid_src_1->set_t_axis_variable(md);
       regrid_src_1->set_t_axis(md);

       regrid_1->set_interpolation_mode_linear();
       regrid_1->set_input_connection(0, regrid_src_1->get_output_port());
       regrid_1->set_input_connection(1, sim_coords->get_output_port());

       rename_1->set_original_variable_names({opt_vals["geopotential"].as<string>()});
       rename_1->set_new_variable_names({"Z300"});
       rename_1->set_input_connection(regrid_1->get_output_port());

       //slice variables at pressure level of 500mb
       regrid_src_2->set_bounds({bounds[0], bounds[1], bounds[2], bounds[3], 500, 500, 0, 0});
       regrid_src_2->set_whole_extents({0lu, x->size() - 1lu, 0lu, y->size() - 1lu, 0lu, 0lu, 0lu, 0lu});
       regrid_src_2->set_t_axis_variable(md);
       regrid_src_2->set_t_axis(md);

       regrid_2->set_interpolation_mode_linear();
       regrid_2->set_input_connection(0, regrid_src_2->get_output_port());
       regrid_2->set_input_connection(1, sim_coords->get_output_port());

       rename_2->set_original_variable_names({opt_vals["geopotential"].as<string>()});
       rename_2->set_new_variable_names({"Z500"});
       rename_2->set_input_connection(regrid_2->get_output_port());

       //join the two new meshes
       join->set_number_of_input_connections(2);
       join->set_input_connection(0, rename_1->get_output_port());
       join->set_input_connection(1, rename_2->get_output_port());
       head = join;
    }
    size_t n_var = thickness->get_number_of_dependent_variables();
    if (n_var != 2)
    {
       TECA_FATAL_ERROR("thickness calculation requires 2 "
             "variables. given " << n_var)
       return -1;
    }
    thickness->set_execute_callback(point_wise_difference(thickness->get_dependent_variable(0),
                                                          thickness->get_dependent_variable(1),
                                                          thickness->get_derived_variable(0)));

    if (!opt_vals["closed_contour_cmd"].defaulted())
    {
       candidates->set_closed_contour_cmd(opt_vals["closed_contour_cmd"].as<string>());
    }
    else
    {
       if (opt_vals["sea_level_pressure"].as<string>() == "")
          TECA_FATAL_ERROR("Missing name of variable with sea level pressure")
       else
       {
          std::string text = opt_vals["sea_level_pressure"].as<string>()+",200.0,5.5,0;"+thickness->get_derived_variable(0)+",-58.8,6.5,1.0";
          candidates->set_closed_contour_cmd(text);
       }
    }

    if (!opt_vals["no_closed_contour_cmd"].defaulted())
    {
       candidates->set_no_closed_contour_cmd(opt_vals["no_closed_contour_cmd"].as<string>());
    }

    if (!opt_vals["candidate_threshold_cmd"].defaulted())
    {
       candidates->set_candidate_threshold_cmd(opt_vals["candidate_threshold_cmd"].as<string>());
    }

    //surface wind speed calculation
    if (opt_vals["surface_wind_u"].as<string>() == "")
       TECA_FATAL_ERROR("Missing name of variable with surface wind x-component")
    else
       surf_wind->set_component_0_variable(opt_vals["surface_wind_u"].as<string>());

    if (opt_vals["surface_wind_v"].as<string>() == "")
       TECA_FATAL_ERROR("Missing name of variable with surface wind y-component")
    else
       surf_wind->set_component_1_variable(opt_vals["surface_wind_v"].as<string>());

    if (!opt_vals["output_cmd"].defaulted())
    {
       candidates->set_output_cmd(opt_vals["output_cmd"].as<string>());
    }
    else
    {
       if (opt_vals["sea_level_pressure"].as<string>() == "")
          TECA_FATAL_ERROR("Missing name of variable with sea level pressure")

       if (opt_vals["geopotential_at_surface"].as<string>() == "")
          TECA_FATAL_ERROR("Missing name of variable with geopotential at the surface")

       std::string text = opt_vals["sea_level_pressure"].as<string>()+",min,0;"+surf_wind->get_l2_norm_variable()+",max,2;"+opt_vals["geopotential_at_surface"].as<string>()+",min,0";
       candidates->set_output_cmd(text);
    }

    if (!opt_vals["in_fmt"].defaulted())
    {
       tracks->set_in_fmt(opt_vals["in_fmt"].as<string>());
    }
    else
    {
       if (opt_vals["sea_level_pressure"].as<string>() == "")
          TECA_FATAL_ERROR("Missing name of variable with sea level pressure")

       if (opt_vals["geopotential_at_surface"].as<string>() == "")
          TECA_FATAL_ERROR("Missing name of variable with geopotential at the surface")

       std::string text = "i,j,lat,lon,"+opt_vals["sea_level_pressure"].as<string>()+","+surf_wind->get_l2_norm_variable()+","+opt_vals["geopotential_at_surface"].as<string>();
       tracks->set_in_fmt(text);
    }

    if (!opt_vals["search_by_threshold"].defaulted())
    {
       candidates->set_search_by_threshold(opt_vals["search_by_threshold"].as<string>());
    }

    if (!opt_vals["min_lon"].defaulted())
    {
       candidates->set_min_lon(opt_vals["min_lon"].as<double>());
    }

    if (!opt_vals["max_lon"].defaulted())
    {
       candidates->set_max_lon(opt_vals["max_lon"].as<double>());
    }

    if (!opt_vals["min_lat"].defaulted())
    {
       candidates->set_min_lat(opt_vals["min_lat"].as<double>());
    }

    if (!opt_vals["max_lat"].defaulted())
    {
       candidates->set_max_lat(opt_vals["max_lat"].as<double>());
    }

    if (!opt_vals["merge_dist"].defaulted())
    {
       candidates->set_merge_dist(opt_vals["merge_dist"].as<double>());
    }

    if (!opt_vals["diag_connect"].defaulted())
    {
       candidates->set_diag_connect(opt_vals["diag_connect"].as<bool>());
    }

    if (!opt_vals["regional"].defaulted())
    {
       candidates->set_regional(opt_vals["regional"].as<bool>());
    }

    if (!opt_vals["out_header"].defaulted())
    {
       candidates->set_out_header(opt_vals["out_header"].as<bool>());
    }
    candidates->set_verbose(opt_vals["verbose"].as<int>());
    candidates->initialize();

    if (!opt_vals["first_step"].defaulted())
        map_reduce->set_start_index(opt_vals["first_step"].as<long>());

    if (!opt_vals["last_step"].defaulted())
        map_reduce->set_end_index(opt_vals["last_step"].as<long>());

    if (!opt_vals["time_filter"].defaulted())
        map_reduce->set_stride(opt_vals["time_filter"].as<long>());

    if (!opt_vals["n_threads"].defaulted())
        map_reduce->set_thread_pool_size(opt_vals["n_threads"].as<int>());
    else
        map_reduce->set_thread_pool_size(-1);

    candidate_writer->set_file_name(opt_vals["candidate_file"].as<string>());
    candidate_writer->set_output_format_auto();

    sort->set_index_column("step");

    if (!opt_vals["min_time"].defaulted())
    {
       tracks->set_min_time(opt_vals["min_time"].as<string>());
    }

    if (!opt_vals["cal_type"].defaulted())
    {
       tracks->set_cal_type(opt_vals["cal_type"].as<string>());
    }

    if (!opt_vals["max_gap"].defaulted())
    {
       tracks->set_max_gap(opt_vals["max_gap"].as<string>());
    }

    if (!opt_vals["track_threshold_cmd"].defaulted())
    {
       tracks->set_track_threshold_cmd(opt_vals["track_threshold_cmd"].as<string>());
    }
    else
    {
       if (opt_vals["geopotential_at_surface"].as<string>() == "")
          TECA_FATAL_ERROR("Missing name of variable with geopotential at the surface")
       else
       {
          std::string text = surf_wind->get_l2_norm_variable()+",>=,10.0,10;lat,<=,50.0,10;lat,>=,-50.0,10;"+opt_vals["geopotential_at_surface"].as<string>()+",<=,15.0,10";
          tracks->set_track_threshold_cmd(text);
       }
    }

    if (!opt_vals["prioritize"].defaulted())
    {
       tracks->set_prioritize(opt_vals["prioritize"].as<string>());
    }

    if (!opt_vals["min_path_length"].defaulted())
    {
       tracks->set_min_path_length(opt_vals["min_path_length"].as<int>());
    }

    if (!opt_vals["range"].defaulted())
    {
       tracks->set_range(opt_vals["range"].as<double>());
    }

    if (!opt_vals["min_endpoint_distance"].defaulted())
    {
       tracks->set_min_endpoint_distance(opt_vals["min_endpoint_distance"].as<double>());
    }

    if (!opt_vals["min_path_distance"].defaulted())
    {
       tracks->set_min_path_distance(opt_vals["min_path_distance"].as<double>());
    }

    if (!opt_vals["allow_repeated_times"].defaulted())
    {
       tracks->set_allow_repeated_times(opt_vals["allow_repeated_times"].as<bool>());
    }
    tracks->initialize();

    track_writer->set_file_name(opt_vals["track_file"].as<string>());
    track_writer->set_output_format_auto();

    // connect all the stages
    thickness->set_input_connection(head->get_output_port());
    surf_wind->set_input_connection(thickness->get_output_port());
    candidates->set_input_connection(surf_wind->get_output_port());
    map_reduce->set_input_connection(candidates->get_output_port());
    candidate_writer->set_input_connection(map_reduce->get_output_port());
    sort->set_input_connection(candidate_writer->get_output_port());
    calendar->set_input_connection(sort->get_output_port());
    tracks->set_input_connection(calendar->get_output_port());
    track_writer->set_input_connection(tracks->get_output_port());

    // run the pipeline
    track_writer->update();

    return 0;
}
