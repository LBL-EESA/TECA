#include "teca_cf_reader.h"
#include "teca_normalize_coordinates.h"
#include "teca_mask.h"
#include "teca_l2_norm.h"
#include "teca_binary_segmentation.h"
#include "teca_connected_components.h"
#include "teca_2d_component_area.h"
#include "teca_dataset_capture.h"
#include "teca_metadata.h"
#include "teca_table.h"
#include "teca_dataset_source.h"
#include "teca_dataset_diff.h"
#include "teca_table_reader.h"
#include "teca_table_sort.h"
#include "teca_table_writer.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_index_executive.h"
#include "teca_file_util.h"
#include "teca_system_interface.h"
#include "teca_system_util.h"
#include "teca_profiler.h"

#include <iostream>
#include <string>

using namespace std;

int main(int argc, char **argv)
{
    teca_profiler::initialize();
    teca_system_interface::set_stack_trace_on_error();

    if (argc != 13)
    {
        cerr << "test_connected_components [dataset regex] "
            "[x var] [y var] [z var] [t var] [u var] [v var] "
            "[first step] [last step] [threshold] [out file] "
            "[baseline]"
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
    int end_index = atoi(argv[9]);
    double threshold = atof(argv[10]);
    string out_file = argv[11];
    string  baseline = argv[12];
    int have_baseline = 0;
    if (teca_file_util::file_exists(baseline.c_str()))
        have_baseline = 1;

    p_teca_cf_reader cfr = teca_cf_reader::New();
    cfr->set_files_regex(dataset_regex);
    cfr->set_x_axis_variable(x_var);
    cfr->set_y_axis_variable(y_var);
    cfr->set_z_axis_variable(z_var);
    cfr->set_t_axis_variable(t_var);
    cfr->set_periodic_in_x(1);

    p_teca_normalize_coordinates coords = teca_normalize_coordinates::New();
    coords->set_input_connection(cfr->get_output_port());

    p_teca_mask mask = teca_mask::New();
    mask->set_input_connection(coords->get_output_port());
    mask->set_low_threshold_value(1e4);
    mask->set_mask_value(0);
    mask->append_mask_variable(u_var);
    mask->append_mask_variable(v_var);

    p_teca_l2_norm l2n = teca_l2_norm::New();
    l2n->set_input_connection(mask->get_output_port());
    l2n->set_component_0_variable(u_var);
    l2n->set_component_1_variable(v_var);
    l2n->set_l2_norm_variable("wind_speed");

    p_teca_binary_segmentation bs = teca_binary_segmentation::New();
    bs->set_input_connection(l2n->get_output_port());
    bs->set_threshold_variable("wind_speed");
    bs->set_low_threshold_value(threshold);
    bs->set_segmentation_variable("wind_speed_seg");

    p_teca_connected_components cc = teca_connected_components::New();
    cc->set_input_connection(bs->get_output_port());
    cc->set_segmentation_variable("wind_speed_seg");
    cc->set_component_variable("con_comps");
    cc->set_verbose(1);

    p_teca_2d_component_area ca = teca_2d_component_area::New();
    ca->set_input_connection(cc->get_output_port());
    ca->set_component_variable("con_comps");

    p_teca_dataset_capture cao = teca_dataset_capture::New();
    cao->set_input_connection(ca->get_output_port());

    p_teca_index_executive exe = teca_index_executive::New();
    exe->set_start_index(first_step);
    exe->set_end_index(end_index);
    exe->set_verbose(1);

    p_teca_cartesian_mesh_writer wri = teca_cartesian_mesh_writer::New();
    wri->set_input_connection(cao->get_output_port());
    wri->set_executive(exe);
    wri->set_file_name(out_file);

    wri->update();

    // put the component area in a table
    const_p_teca_dataset ds = cao->get_dataset();
    teca_metadata mdo = ds->get_metadata();

    p_teca_variant_array comp_label = mdo.get("component_ids");
    p_teca_variant_array comp_area = mdo.get("component_area");

    p_teca_table test_data = teca_table::New();
    test_data->append_column("comp_label", comp_label);
    test_data->append_column("comp_area", comp_area);

    // feed the table into the regression test
    p_teca_dataset_source dss = teca_dataset_source::New();
    dss->set_dataset(test_data);

    // sort by area since the GPU and CPU versions generate different label ids
    p_teca_table_sort sort = teca_table_sort::New();
    sort->set_input_connection(dss->get_output_port());
    sort->set_index_column("comp_area");
    sort->set_ascending_order(1);

    // regression test
    bool do_test = true;
    teca_system_util::get_environment_variable("TECA_DO_TEST", do_test);
    if (do_test && have_baseline)
    {
        // run the test
        p_teca_table_reader table_reader = teca_table_reader::New();
        table_reader->set_file_name(baseline);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, table_reader->get_output_port());
        diff->set_input_connection(1, sort->get_output_port());
        diff->set_skip_arrays({"comp_label"});

        diff->update();
    }
    else
    {
        // make a baseline
        cerr << "generating baseline image " << baseline << endl;
        p_teca_table_writer table_writer = teca_table_writer::New();
        table_writer->set_input_connection(sort->get_output_port());
        table_writer->set_file_name(baseline.c_str());

        table_writer->update();
    }

    teca_profiler::finalize();
    return 0;
}
