from teca import *
import numpy
if get_teca_has_cupy():
    import cupy
import sys

set_stack_trace_on_error()

if not len(sys.argv) == 7:
    sys.stderr.write('test_programmable_algorithm.py [dataset regex] ' \
        '[u_var] [v_var] [first step] [last step] [out file name]\n')
    sys.exit(-1)

data_regex = sys.argv[1]
u_var = sys.argv[2]
v_var = sys.argv[3]
first_step = int(sys.argv[4])
end_index = int(sys.argv[5])
out_file = sys.argv[6]

class wind_speed:
    @staticmethod
    def execute(port, data_in, req):
        global u_var, v_var
        dev = -1
        np = numpy
        if get_teca_has_cuda() and get_teca_has_cupy():
            dev = req['device_id']
            if dev >= 0:
                cupy.cuda.Device(dev).use()
                np = cupy
        in_mesh = as_teca_cartesian_mesh(data_in[0])
        out_mesh = teca_cartesian_mesh.New()
        out_mesh.shallow_copy(in_mesh)
        arrays = out_mesh.get_point_arrays()
        if dev < 0:
            u = arrays[u_var].get_host_accessible()
            v = arrays[v_var].get_host_accessible()
        else:
            u = arrays[u_var].get_cuda_accessible()
            v = arrays[v_var].get_cuda_accessible()
        w = np.sqrt(u*u + v*v)
        arrays['wind_speed'] = w
        return out_mesh

cfr = teca_cf_reader.New()
cfr.set_files_regex(data_regex)
cfr.set_x_axis_variable('lon')
cfr.set_y_axis_variable('lat')
cfr.set_t_axis_variable('time')

coords = teca_normalize_coordinates.New()
coords.set_input_connection(cfr.get_output_port())

alg = teca_derived_quantity.New()
alg.append_dependent_variable(u_var)
alg.append_dependent_variable(v_var)
alg.set_derived_variable('wind_speed')
alg.set_execute_callback(wind_speed.execute)
alg.set_input_connection(coords.get_output_port())

exe = teca_index_executive.New()
exe.set_start_index(first_step)
exe.set_end_index(end_index)

wri = teca_cartesian_mesh_writer.New()
wri.set_input_connection(alg.get_output_port())
wri.set_executive(exe)
wri.set_file_name(out_file)

wri.update()
