#include "teca_config.h"
#include "teca_cf_reader.h"
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
#include "teca_test_util.h"
#include "teca_time_step_executive.h"
#include "teca_vtk_cartesian_mesh_writer.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"

#include <vector>
#include <string>
#include <iostream>

using namespace std;
using namespace teca_derived_quantity_numerics;


// --------------------------------------------------------------------------
int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();

    teca_system_interface::set_stack_trace_on_error();

    // parse command line
    string regex;
    string baseline;
    int have_baseline = 0;
    long first_step = 0;
    long last_step = -1;
    unsigned int n_threads = 1;
    string ux_850mb;
    string uy_850mb;
    string ux_surf;
    string uy_surf;
    string P_surf;
    string T_500mb;
    string T_200mb;
    string z_1000mb;
    string z_200mb;
    double low_lat = 0;
    double high_lat = -1;

    if (rank == 0)
    {
        if (argc != 17)
        {
            cerr << endl << "Usage error:" << endl
                << "test_tc_candidates [input regex] [output] [first step] [last step] [n threads] "
                   "[850 mb wind x] [850 mb wind y] [surface wind x] [surface wind y] [surface pressure] "
                   "[500 mb temp] [200 mb temp] [1000 mb z] [200 mb z] [low lat] [high lat]"
                << endl << endl;
            return -1;
        }

        // parse command line
        regex = argv[1];
        baseline = argv[2];
        if (teca_file_util::file_exists(baseline.c_str()))
            have_baseline = 1;
        first_step = atoi(argv[3]);
        last_step = atoi(argv[4]);
        n_threads = atoi(argv[5]);
        ux_850mb = argv[6];
        uy_850mb = argv[7];
        ux_surf = argv[8];
        uy_surf = argv[9];
        P_surf = argv[10];
        T_500mb = argv[11];
        T_200mb = argv[12];
        z_1000mb = argv[13];
        z_200mb = argv[14];
        low_lat = atof(argv[15]);
        high_lat = atof(argv[16]);
    }

    teca_test_util::bcast(regex);
    teca_test_util::bcast(baseline);
    teca_test_util::bcast(have_baseline);
    teca_test_util::bcast(first_step);
    teca_test_util::bcast(last_step);
    teca_test_util::bcast(n_threads);
    teca_test_util::bcast(ux_850mb);
    teca_test_util::bcast(uy_850mb);
    teca_test_util::bcast(ux_surf);
    teca_test_util::bcast(uy_surf);
    teca_test_util::bcast(P_surf);
    teca_test_util::bcast(T_500mb);
    teca_test_util::bcast(T_200mb);
    teca_test_util::bcast(z_1000mb);
    teca_test_util::bcast(z_200mb);
    teca_test_util::bcast(low_lat);
    teca_test_util::bcast(high_lat);

    // create the pipeline objects
    p_teca_cf_reader cf_reader = teca_cf_reader::New();
    cf_reader->set_files_regex(regex);

    // surface wind speed
    p_teca_l2_norm surf_wind = teca_l2_norm::New();
    surf_wind->set_input_connection(cf_reader->get_output_port());
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
    core_temp->set_derived_variable("core_temperature");
    core_temp->set_execute_callback(
        point_wise_average(T_500mb, T_200mb, "core_temperature"));

    // thickness
    p_teca_derived_quantity thickness = teca_derived_quantity::New();
    thickness->set_input_connection(core_temp->get_output_port());
    thickness->set_dependent_variables({z_1000mb, z_200mb});
    thickness->set_derived_variable("thickness");
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
