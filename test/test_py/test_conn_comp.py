from teca_py_core import *
from teca_py_io import *
from teca_py_alg import *
import sys

if not len(sys.argv) == 3:
    sys.stderr.write('test_conn_comp.py [dataset regex] [out file]\n')
    sys.exit(-1)

data_regex = sys.argv[1]
out_file = sys.argv[2]

cfr = teca_cf_reader.New()
cfr.set_files_regex(data_regex)
cfr.set_x_axis_variable('lon')
cfr.set_y_axis_variable('lat')
cfr.set_z_axis_variable('plev')
cfr.set_t_axis_variable('time')

mask = teca_mask.New()
mask.set_low_threshold_value(1e4)
mask.set_mask_value(0)
mask.append_mask_variable('U')
mask.append_mask_variable('V')
mask.set_input_connection(cfr.get_output_port())

l2n = teca_l2_norm.New()
l2n.set_component_0_variable('U')
l2n.set_component_1_variable('V')
l2n.set_l2_norm_variable('wind_speed')
l2n.set_input_connection(mask.get_output_port())

cc = teca_connected_components.New()
cc.set_threshold_variable('wind_speed')
cc.set_low_threshold_value(30)
cc.set_label_variable('30_mps_ccomps')
cc.set_input_connection(l2n.get_output_port())

exe = teca_time_step_executive.New()
exe.set_first_step(0)
exe.set_last_step(-1)

wri = teca_vtk_cartesian_mesh_writer.New()
wri.set_input_connection(cc.get_output_port())
wri.set_executive(exe)
wri.set_file_name(out_file)

wri.update()
