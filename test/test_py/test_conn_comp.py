from teca_py_core import *
from teca_py_io import *
from teca_py_alg import *

d3 = True

cfr = teca_cf_reader.New()
if d3:
    cfr.set_files_regex('/home/bloring/work/teca/jet/cam5_1_amip_run2\.cam2\.h2\.1980-12-\.*')
    cfr.set_x_axis_variable('lon')
    cfr.set_y_axis_variable('lat')
    cfr.set_z_axis_variable('plev')
    cfr.set_t_axis_variable('time')
else:
    cfr.set_files_regex('/home/bloring/work/teca/data/cam5_1_amip_run2\.cam2\.h2\.*')
    cfr.set_x_axis_variable('lon')
    cfr.set_y_axis_variable('lat')
    cfr.set_t_axis_variable('time')

mask = teca_mask.New()
mask.set_low_threshold_value(1e4)
mask.set_mask_value(0)
mask.append_mask_variable('U')
mask.append_mask_variable('V')
mask.set_input_connection(cfr.get_output_port())

l2n = teca_l2_norm.New()
if d3:
    l2n.set_component_0_variable('U')
    l2n.set_component_1_variable('V')
else:
    l2n.set_component_0_variable('U850')
    l2n.set_component_1_variable('V850')
l2n.set_l2_norm_variable('wind_speed')
l2n.set_input_connection(mask.get_output_port())

cc = teca_connected_components.New()
cc.set_threshold_variable('wind_speed')
if d3:
    cc.set_low_threshold_value(30)
    cc.set_label_variable('30_mps_ccomps')
else:
    cc.set_low_threshold_value(15)
    cc.set_label_variable('15_mps_ccomps')
cc.set_input_connection(l2n.get_output_port())

exe = teca_time_step_executive.New()
exe.set_first_step(0)
exe.set_last_step(0)

wri = teca_vtk_cartesian_mesh_writer.New()
wri.set_input_connection(cc.get_output_port())
wri.set_executive(exe)
if d3:
    wri.set_file_name('3d_ccomps_%t%.vtk')
else:
    wri.set_file_name('2d_ccomps_%t%.vtk')

wri.update()
