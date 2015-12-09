from teca_py_core import *
from teca_py_io import *
from teca_py_alg import *
import numpy as np
import sys

class wind_speed:
    @staticmethod
    def report(port, md_in):
        md = md_in[0]
        md.append('variables', 'wind_speed_850')
        return md

    @staticmethod
    def request(port, md_in, req_in):
        req = teca_metadata(req_in)
        req['arrays'] = ['U850', 'V850']
        return [req]

    @staticmethod
    def execute(port, data_in, req):
        in_mesh = as_teca_cartesian_mesh(data_in[0])
        out_mesh = teca_cartesian_mesh.New()
        out_mesh.shallow_copy(in_mesh)
        arrays = out_mesh.get_point_arrays()
        u = arrays['U850']
        v = arrays['V850']
        w = np.sqrt(u*u + v*v)
        arrays['wind_speed_850'] = w
        return out_mesh

cfr = teca_cf_reader.New()
cfr.set_files_regex('/home/bloring/work/teca/data/cam5_1_amip_run2\.cam2\.h2\.*')
cfr.set_x_axis_variable('lon')
cfr.set_y_axis_variable('lat')
cfr.set_t_axis_variable('time')

alg = teca_programmable_algorithm.New()
alg.set_number_of_input_connections(1)
alg.set_number_of_output_ports(1)
alg.set_report_callback(wind_speed.report)
alg.set_request_callback(wind_speed.request)
alg.set_execute_callback(wind_speed.execute)
alg.set_input_connection(cfr.get_output_port())

exe = teca_time_step_executive.New()
exe.set_first_step(0)
exe.set_last_step(0)

wri = teca_vtk_cartesian_mesh_writer.New()
wri.set_input_connection(alg.get_output_port())
wri.set_executive(exe)
wri.set_file_name('amip_run2_%t%.vtk')

wri.update()
