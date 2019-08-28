from teca import *
import numpy as np
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
    def report(port, md_in):
        md = md_in[0]
        md.append('variables', 'wind_speed')
        return md

    @staticmethod
    def request(port, md_in, req_in):
        global u_var, v_var
        req = teca_metadata(req_in)
        req['arrays'] = [u_var, v_var]
        return [req]

    @staticmethod
    def execute(port, data_in, req):
        global u_var, v_var
        in_mesh = as_teca_cartesian_mesh(data_in[0])
        out_mesh = teca_cartesian_mesh.New()
        out_mesh.shallow_copy(in_mesh)
        arrays = out_mesh.get_point_arrays()
        u = arrays[u_var]
        v = arrays[v_var]
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

alg = teca_programmable_algorithm.New()
alg.set_name('wind_speed')
alg.set_number_of_input_connections(1)
alg.set_number_of_output_ports(1)
alg.set_report_callback(wind_speed.report)
alg.set_request_callback(wind_speed.request)
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
