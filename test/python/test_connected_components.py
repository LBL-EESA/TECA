from teca import *
import sys

set_stack_trace_on_error()

if not len(sys.argv) == 9:
    sys.stderr.write('test_conn_comp.py [dataset regex] [z_var] ' \
        '[u_var] [v_var] [threshold] [first step] [last step] [out file]\n')
    sys.exit(-1)

data_regex = sys.argv[1]
x_var = 'lon'
y_var = 'lat'
z_var = sys.argv[2]
t_var = 'time'
u_var = sys.argv[3]
v_var = sys.argv[4]
threshold = float(sys.argv[5])
first_step = int(sys.argv[6])
end_index = int(sys.argv[7])
out_file = sys.argv[8]

cfr = teca_cf_reader.New()
cfr.set_files_regex(data_regex)
cfr.set_x_axis_variable(x_var)
cfr.set_y_axis_variable(y_var)
cfr.set_z_axis_variable(z_var)
cfr.set_t_axis_variable(t_var)

coords = teca_normalize_coordinates.New()
coords.set_input_connection(cfr.get_output_port())

mask = teca_mask.New()
mask.set_input_connection(coords.get_output_port())
mask.set_low_threshold_value(1e4)
mask.set_mask_value(0)
mask.append_mask_variable(u_var)
mask.append_mask_variable(v_var)

l2n = teca_l2_norm.New()
l2n.set_input_connection(mask.get_output_port())
l2n.set_component_0_variable(u_var)
l2n.set_component_1_variable(v_var)
l2n.set_l2_norm_variable('wind_speed')

bs = teca_binary_segmentation.New()
bs.set_input_connection(l2n.get_output_port())
bs.set_threshold_variable('wind_speed')
bs.set_low_threshold_value(threshold)
bs.set_segmentation_variable('wind_speed_seg')

cc = teca_connected_components.New()
cc.set_input_connection(bs.get_output_port())
cc.set_segmentation_variable('wind_speed_seg')
cc.set_component_variable('con_comps')

exe = teca_index_executive.New()
exe.set_start_index(first_step)
exe.set_end_index(end_index)

wri = teca_cartesian_mesh_writer.New()
wri.set_input_connection(cc.get_output_port())
wri.set_executive(exe)
wri.set_file_name(out_file)

wri.update()
