from teca_py_core import *
from teca_py_io import *

cfr = teca_cf_reader.New()
cfr.set_files_regex('/home/bloring/work/teca/data/cam5_1_amip_run2\.cam2\.h2\.*')
cfr.set_x_axis_variable('lon')
cfr.set_y_axis_variable('lat')
cfr.set_z_axis_variable('')
cfr.set_t_axis_variable('time')

exe = teca_time_step_executive.New()
exe.set_first_step(0)
exe.set_last_step(-1)
exe.set_arrays(['U850', 'V850'])

wri = teca_vtk_cartesian_mesh_writer.New()
wri.set_input_connection(cfr.get_output_port())
wri.set_executive(exe)
wri.set_file_name('amip_run2_%t%.vtk')

wri.update()
