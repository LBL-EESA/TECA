from teca_py_core import *
from teca_py_io import *
from teca_py_alg import *
import sys

if not len(sys.argv) == 3:
    sys.stderr.write('test_cf_reader.py [dataset regex] [out file name]\n')
    sys.exit(-1)

data_regex = sys.argv[1]
out_file = sys.argv[2]

cfr = teca_cf_reader.New()
cfr.set_files_regex(data_regex)
cfr.set_x_axis_variable('lon')
cfr.set_y_axis_variable('lat')
cfr.set_t_axis_variable('time')

exe = teca_time_step_executive.New()
exe.set_first_step(0)
exe.set_last_step(-1)
exe.set_arrays(['U850', 'V850'])

wri = teca_vtk_cartesian_mesh_writer.New()
wri.set_input_connection(cfr.get_output_port())
wri.set_executive(exe)
wri.set_file_name(out_file)

wri.update()
