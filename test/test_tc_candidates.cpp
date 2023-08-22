#include "teca_config.h"
#include "teca_cf_reader.h"
#include "teca_normalize_coordinates.h"
#include "teca_l2_norm.h"
#include "teca_vorticity.h"
#include "teca_derived_quantity.h"
#include "teca_derived_quantity_numerics.h"
#include "teca_tc_candidates.h"
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
#include "teca_cartesian_mesh_writer.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"
#include "teca_system_util.h"
#include "teca_mpi.h"
#include "teca_thread_util.h"

#include <vector>
#include <string>
#include <iostream>

using namespace teca_derived_quantity_numerics;


// --------------------------------------------------------------------------
int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();

    teca_system_interface::set_stack_trace_on_error();
    teca_system_interface::set_stack_trace_on_mpi_error();

    // parse command line
    std::string regex;
    std::string baseline;
    int have_baseline = 0;
    long start_index = 0;
    long end_index = -1;
    int n_threads = -1;
    int n_omp_threads = 1;
    std::string ux_850mb;
    std::string uy_850mb;
    std::string ux_surf;
    std::string uy_surf;
    std::string P_surf;
    std::string T_500mb;
    std::string T_200mb;
    std::string z_1000mb;
    std::string z_200mb;
    double low_lat = 0;
    double high_lat = -1;
    int max_it = 50;

    if (argc != 19)
    {
        std::cerr << std::endl << "Usage error:" << std::endl
            << "test_tc_candidates [input regex] [output] [first step] [last step] [n threads] "
               "[n_omp_threads] [850 mb wind x] [850 mb wind y] [surface wind x] [surface wind y] "
               "[surface pressure] [500 mb temp] [200 mb temp] [1000 mb z] [200 mb z] [low lat] "
               "[high lat] [max it]"
            << std::endl << std::endl;
        return -1;
    }

    // parse command line
    regex = argv[1];
    baseline = argv[2];
    if (teca_file_util::file_exists(baseline.c_str()))
        have_baseline = 1;
    start_index = atoi(argv[3]);
    end_index = atoi(argv[4]);
    n_threads = atoi(argv[5]);
    n_omp_threads = atoi(argv[6]);
    ux_850mb = argv[7];
    uy_850mb = argv[8];
    ux_surf = argv[9];
    uy_surf = argv[10];
    P_surf = argv[11];
    T_500mb = argv[12];
    T_200mb = argv[13];
    z_1000mb = argv[14];
    z_200mb = argv[15];
    low_lat = atof(argv[16]);
    high_lat = atof(argv[17]);
    max_it = atoi(argv[18]);

    // create the pipeline objects
    p_teca_cf_reader cf_reader = teca_cf_reader::New();
    cf_reader->set_files_regex(regex);

    p_teca_normalize_coordinates coords = teca_normalize_coordinates::New();
    coords->set_input_connection(cf_reader->get_output_port());

    // surface wind speed
    p_teca_l2_norm surf_wind = teca_l2_norm::New();
    surf_wind->set_input_connection(coords->get_output_port());
    surf_wind->set_component_0_variable(ux_surf);
    surf_wind->set_component_1_variable(uy_surf);
    surf_wind->set_l2_norm_variable("surface_wind");

    // vorticity at 850mb
    p_teca_vorticity vort_850mb = teca_vorticity::New();
    vort_850mb->set_input_connection(surf_wind->get_output_port());
    vort_850mb->set_component_0_variable(ux_850mb);
    vort_850mb->set_component_1_variable(uy_850mb);
    vort_850mb->set_vorticity_variable("850mb_vorticity");

    // core temperature
    p_teca_derived_quantity core_temp = teca_derived_quantity::New();
    core_temp->set_input_connection(vort_850mb->get_output_port());
    core_temp->set_dependent_variables({T_500mb, T_200mb});
    core_temp->set_derived_variables({"core_temperature"}, {{}});
    core_temp->set_execute_callback(
        point_wise_average(T_500mb, T_200mb, "core_temperature"));

    // thickness
    p_teca_derived_quantity thickness = teca_derived_quantity::New();
    thickness->set_input_connection(core_temp->get_output_port());
    thickness->set_dependent_variables({z_1000mb, z_200mb});
    thickness->set_derived_variables({"thickness"},{{}});
    thickness->set_execute_callback(
        point_wise_difference(z_1000mb, z_200mb, "thickness"));

    // candidate detection
    p_teca_tc_candidates cand = teca_tc_candidates::New();
    cand->set_input_connection(thickness->get_output_port());
    cand->set_surface_wind_speed_variable("surface_wind");
    cand->set_vorticity_850mb_variable("850mb_vorticity");
    cand->set_sea_level_pressure_variable(P_surf);
    cand->set_core_temperature_variable("core_temperature");
    cand->set_thickness_variable("thickness");
    cand->set_max_core_radius(2.0);
    cand->set_min_vorticity_850mb(1.6e-4);
    cand->set_vorticity_850mb_window(7.74446);
    cand->set_max_pressure_delta(400.0);
    cand->set_max_pressure_radius(5.0);
    cand->set_max_core_temperature_delta(0.8);
    cand->set_max_core_temperature_radius(5.0);
    cand->set_max_thickness_delta(50.0);
    cand->set_max_thickness_radius(4.0);
    cand->set_search_lat_low(low_lat);
    cand->set_search_lat_high(high_lat);
    //cand->set_search_lon_low();
    //cand->set_search_lon_high();
    cand->set_omp_num_threads(n_omp_threads);
    cand->set_minimizer_iterations(max_it);

    // map-reduce
    p_teca_table_reduce map_reduce = teca_table_reduce::New();
    map_reduce->set_input_connection(cand->get_output_port());
    map_reduce->set_start_index(start_index);
    map_reduce->set_end_index(end_index);
    map_reduce->set_verbose(1);
    map_reduce->set_thread_pool_size(n_threads);

    // sort results in time
    p_teca_table_sort sort = teca_table_sort::New();
    sort->set_input_connection(map_reduce->get_output_port());
    sort->set_index_column("surface_wind");

    // compute dates
    p_teca_table_calendar cal = teca_table_calendar::New();
    cal->set_input_connection(sort->get_output_port());

    // regression test
    bool do_test = true;
    teca_system_util::get_environment_variable("TECA_DO_TEST", do_test);
    if (do_test && have_baseline)
    {
        // run the test
        if (rank == 0)
            std::cerr << "running the test ... " << std::endl;

        p_teca_table_reader table_reader = teca_table_reader::New();
        table_reader->set_file_name(baseline);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, table_reader->get_output_port());
        diff->set_input_connection(1, cal->get_output_port());
        diff->set_verbose(1);

        // storm id is non-deterministic when OpenMP threading is used
        diff->set_skip_array("storm_id");

        diff->update();
    }
    else
    {
        // make a baseline
        if (rank == 0)
            std::cerr << "generating baseline image " << baseline << std::endl;

        p_teca_table_writer table_writer = teca_table_writer::New();
        table_writer->set_input_connection(cal->get_output_port());
        table_writer->set_file_name(baseline.c_str());

        table_writer->update();
    }

    return 0;
}
