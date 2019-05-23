from teca import *
import sys

set_stack_trace_on_error()

argc = len(sys.argv)
if not argc >= 7:
    sys.stderr.write('test_cf_reader.py [dataset regex] ' \
        '[first step] [last step] [out file name] ' \
        '[comp 0] [comp 1]\n')
    sys.exit(-1)

data_regex = sys.argv[1]
first_step = int(sys.argv[2])
end_index = int(sys.argv[3])
out_file = sys.argv[4]
comp_0 = sys.argv[5]
comp_1 = sys.argv[6]
comp_2 = ""
if argc >= 8:
    comp_2 = sys.argv[7]

cfr = teca_cf_reader.New()
cfr.set_files_regex(data_regex)
cfr.set_x_axis_variable('lon')
cfr.set_y_axis_variable('lat')
cfr.set_t_axis_variable('time')

coords = teca_normalize_coordinates.New()
coords.set_input_connection(cfr.get_output_port())

l2 = teca_l2_norm.New()
l2.set_component_0_variable(comp_0)
l2.set_component_1_variable(comp_1)
l2.set_component_2_variable(comp_2)
l2.set_l2_norm_variable("norm")
l2.set_input_connection(coords.get_output_port())

vort = teca_vorticity.New()
vort.set_component_0_variable(comp_0)
vort.set_component_1_variable(comp_1)
vort.set_vorticity_variable("vort")
vort.set_input_connection(l2.get_output_port())

exe = teca_index_executive.New()
exe.set_start_index(first_step)
exe.set_end_index(end_index)

wri = teca_cartesian_mesh_writer.New()
wri.set_input_connection(vort.get_output_port())
wri.set_executive(exe)
wri.set_file_name(out_file)

wri.update()
