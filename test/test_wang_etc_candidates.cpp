#include "teca_config.h"
#include "teca_cf_reader.h"
#include "teca_normalize_coordinates.h"
#include "teca_vorticity.h"
#include "teca_laplacian.h"
#include "teca_wang_etc_candidates.h"
#include "teca_cartesian_mesh_regrid.h"
#include "teca_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_dataset_diff.h"
#include "teca_file_util.h"
#include "teca_table_reader.h"
#include "teca_table_reduce.h"
#include "teca_table_sort.h"
#include "teca_table_calendar.h"
#include "teca_table_writer.h"
#include "teca_test_util.h"
#include "teca_time_step_executive.h"
#include "teca_vtk_cartesian_mesh_writer.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"

#include <vector>
#include <string>
#include <iostream>

using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();

    teca_system_interface::set_stack_trace_on_error();

    // parse command line
    if (argc != 19)
    {
        if (rank == 0)
            cerr << endl << "Usage error:" << endl
                << "test_wang_etc_candidates [files regex] [vorcicity var] "
                   "[pressure var] [elevation file] [elevation var] [min pressure delta] "
                   "[max pressure radius] [min vorticity] [max elevation] [search window] "
                   "[lon 0] [lon 1] [lat 0] [lat 1] [output file] [first step] [last step] "
                   "[n threads]"

                << endl << endl;
        return -1;
    }

    std::string files_regex = argv[1];

    std::string lon_var = "longitude";
    std::string lat_var = "latitude";

    std::string vorticity_var = argv[2];
    std::string pressure_var = argv[3];
    std::string elevation_file = argv[4];
    std::string elevation_var = argv[5];

    double min_pressure_delta = strtod(argv[6], nullptr);
    double max_pressure_radius = strtod(argv[7], nullptr);
    double min_vorticity = strtod(argv[8], nullptr);
    double max_elevation = strtod(argv[9], nullptr);
    double search_win = strtod(argv[10], nullptr);

    double lon_0 = strtod(argv[11], nullptr);
    double lon_1 = strtod(argv[12], nullptr);
    double lat_0 = strtod(argv[13], nullptr);
    double lat_1 = strtod(argv[14], nullptr);

    std::string baseline = argv[15];

    int first_step = atoi(argv[16]);
    int last_step = atoi(argv[16]);
    int n_threads = atoi(argv[18]);

    int have_baseline = 0;
    if (rank == 0)
    {
        if (teca_file_util::file_exists(baseline.c_str()))
            have_baseline = 1;
    }
    teca_test_util::bcast(have_baseline);

    // create the pipeline
    // simulation data reader
    p_teca_cf_reader sim_reader = teca_cf_reader::New();
    sim_reader->set_files_regex(files_regex);
    sim_reader->set_x_axis_variable(lon_var);
    sim_reader->set_y_axis_variable(lat_var);

    p_teca_normalize_coordinates sim_coords = teca_normalize_coordinates::New();
    sim_coords->set_input_connection(sim_reader->get_output_port());

    // elevation reader
    p_teca_cf_reader elev_reader = teca_cf_reader::New();
    elev_reader->set_file_name(elevation_file);
    elev_reader->set_x_axis_variable(lon_var);
    elev_reader->set_y_axis_variable(lat_var);
    elev_reader->set_t_axis_variable("");

    p_teca_normalize_coordinates elev_coords = teca_normalize_coordinates::New();
    elev_coords->set_input_connection(elev_reader->get_output_port());

    // regrid moves elevation feild onto simulation mesh
    p_teca_cartesian_mesh_regrid regrid = teca_cartesian_mesh_regrid::New();
    regrid->set_input_connection(0, sim_coords->get_output_port());
    regrid->set_input_connection(1, elev_coords->get_output_port());
    regrid->add_source_array(elevation_var);

    // laplacian as proxy for vorticity
    p_teca_laplacian laplace = teca_laplacian::New();
    laplace->set_input_connection(regrid->get_output_port());
    laplace->set_scalar_variable(pressure_var);

    // candidate detection
    p_teca_wang_etc_candidates cand = teca_wang_etc_candidates::New();
    cand->set_input_connection(laplace->get_output_port());
    cand->set_pressure_variable(pressure_var);
    cand->set_vorticity_variable(pressure_var+"_laplacian");
    cand->set_elevation_variable(elevation_var);
    cand->set_min_pressure_delta(min_pressure_delta);
    cand->set_max_pressure_radius(max_pressure_radius);
    cand->set_min_vorticity(min_vorticity);
    cand->set_max_elevation(max_elevation);
    cand->set_search_window(search_win);
    cand->set_search_lat_low(lat_0);
    cand->set_search_lat_high(lat_1);
    cand->set_search_lon_low(lon_0);
    cand->set_search_lon_high(lon_1);

    // map-reduce
    p_teca_table_reduce map_reduce = teca_table_reduce::New();
    map_reduce->set_input_connection(cand->get_output_port());
    map_reduce->set_first_step(first_step);
    map_reduce->set_last_step(last_step);
    map_reduce->set_verbose(1);
    map_reduce->set_thread_pool_size(n_threads);

    // sort results in time
    p_teca_table_sort sort = teca_table_sort::New();
    sort->set_input_connection(map_reduce->get_output_port());
    sort->set_index_column("storm_id");

    // compute dates
    p_teca_table_calendar cal = teca_table_calendar::New();
    cal->set_input_connection(sort->get_output_port());

    // regression test
    if (have_baseline)
    {
        // run the test
        p_teca_table_reader table_reader = teca_table_reader::New();
        table_reader->set_file_name(baseline);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, table_reader->get_output_port());
        diff->set_input_connection(1, cal->get_output_port());
        diff->update();
    }
    else
    {
        // make a baseline
        if (rank == 0)
            cerr << "generating baseline image " << baseline << endl;
        p_teca_table_writer table_writer = teca_table_writer::New();
        table_writer->set_input_connection(cal->get_output_port());
        table_writer->set_file_name(baseline.c_str());
        table_writer->update();
    }

    return 0;
}
