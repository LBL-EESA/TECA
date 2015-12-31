#include "teca_cf_reader.h"
#include "teca_mask.h"
#include "teca_l2_norm.h"
#include "teca_connected_components.h"
#include "teca_vtk_cartesian_mesh_writer.h"
#include "teca_time_step_executive.h"

#include <iostream>
#include <string>

using namespace std;

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cerr << "test_connected_components [bool 3d] [dataset regex]" << endl;
        return -1;
    }

    int use_3d = atoi(argv[1]);
    string dataset_regex = argv[2];

    p_teca_cf_reader cfr = teca_cf_reader::New();
    if (use_3d)
    {
        cfr->set_files_regex(dataset_regex);
        cfr->set_x_axis_variable("lon");
        cfr->set_y_axis_variable("lat");
        cfr->set_z_axis_variable("plev");
        cfr->set_t_axis_variable("time");
    }
    else
    {
        cfr->set_files_regex(dataset_regex);
        cfr->set_x_axis_variable("lon");
        cfr->set_y_axis_variable("lat");
        cfr->set_t_axis_variable("time");
    }

    p_teca_mask mask = teca_mask::New();
    mask->set_low_threshold_value(1e4);
    mask->set_mask_value(0);
    mask->append_mask_variable("U");
    mask->append_mask_variable("V");
    mask->set_input_connection(cfr->get_output_port());

    p_teca_l2_norm l2n = teca_l2_norm::New();
    if (use_3d)
    {
        l2n->set_component_0_variable("U");
        l2n->set_component_1_variable("V");
    }
    else
    {
        l2n->set_component_0_variable("U850");
        l2n->set_component_1_variable("V850");
    }
    l2n->set_l2_norm_variable("wind_speed");
    l2n->set_input_connection(mask->get_output_port());

    p_teca_connected_components cc = teca_connected_components::New();
    cc->set_threshold_variable("wind_speed");
    if (use_3d)
    {
        cc->set_low_threshold_value(30);
        cc->set_label_variable("30_mps_ccomps");
    }
    else
    {
        cc->set_low_threshold_value(15);
        cc->set_label_variable("15_mps_ccomps");
    }
    cc->set_input_connection(l2n->get_output_port());

    p_teca_time_step_executive exe = teca_time_step_executive::New();
    exe->set_first_step(0);
    exe->set_last_step(0);

    p_teca_vtk_cartesian_mesh_writer wri = teca_vtk_cartesian_mesh_writer::New();
    wri->set_input_connection(cc->get_output_port());
    wri->set_executive(exe);
    if (use_3d)
        wri->set_file_name("3d_ccomps_%t%.vtk");
    else
        wri->set_file_name("2d_ccomps_%t%.vtk");

    wri->update();

    return 0;
}
