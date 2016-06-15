#include "teca_cf_reader.h"
#include "teca_mask.h"
#include "teca_l2_norm.h"
#include "teca_connected_components.h"
#include "teca_vtk_cartesian_mesh_writer.h"
#include "teca_time_step_executive.h"
#include "teca_system_interface.h"

#include <iostream>
#include <string>

using namespace std;

int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    if (argc != 12)
    {
        cerr << "test_connected_components [dataset regex] "
            "[x var] [y var] [z var] [t var] [u var] [v var] "
            "[first step] [last step] [threshold] [out file]"
            << endl;
        return -1;
    }

    string dataset_regex = argv[1];
    string x_var = argv[2];
    string y_var = argv[3];
    string z_var = argv[4][0] == '.' ? "" : argv[4];
    string t_var = argv[5][0] == '.' ? "" : argv[5];
    string u_var = argv[6];
    string v_var = argv[7];
    int first_step = atoi(argv[8]);
    int last_step = atoi(argv[9]);
    double threshold = atof(argv[10]);
    string out_file = argv[11];

    p_teca_cf_reader cfr = teca_cf_reader::New();
    cfr->set_files_regex(dataset_regex);
    cfr->set_x_axis_variable(x_var);
    cfr->set_y_axis_variable(y_var);
    cfr->set_z_axis_variable(z_var);
    cfr->set_t_axis_variable(t_var);

    p_teca_mask mask = teca_mask::New();
    mask->set_low_threshold_value(1e4);
    mask->set_mask_value(0);
    mask->append_mask_variable(u_var);
    mask->append_mask_variable(v_var);
    mask->set_input_connection(cfr->get_output_port());

    p_teca_l2_norm l2n = teca_l2_norm::New();
    l2n->set_component_0_variable(u_var);
    l2n->set_component_1_variable(v_var);
    l2n->set_l2_norm_variable("wind_speed");
    l2n->set_input_connection(mask->get_output_port());

    p_teca_connected_components cc = teca_connected_components::New();
    cc->set_threshold_variable("wind_speed");
    cc->set_low_threshold_value(threshold);
    cc->set_label_variable("con_comps");
    cc->set_input_connection(l2n->get_output_port());

    p_teca_time_step_executive exe = teca_time_step_executive::New();
    exe->set_first_step(first_step);
    exe->set_last_step(last_step);

    p_teca_vtk_cartesian_mesh_writer wri = teca_vtk_cartesian_mesh_writer::New();
    wri->set_input_connection(cc->get_output_port());
    wri->set_executive(exe);
    wri->set_file_name(out_file);

    wri->update();

    return 0;
}
