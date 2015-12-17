from teca_py_data import *
from teca_py_io import *
from teca_py_alg import *
import numpy as np

# a simple TECA algorithm that computes wind speed
class wind_speed:
    @staticmethod
    def report(o_port, rep_in):
        # add the names of the variables we could generate
        rep_in[0].append('variables', 'wind_speed_850')
        return rep_in[0]

    @staticmethod
    def request(o_port, rep_in, req_in):
        # add the name of arrays that we need to compute
        req_in['arrays'] = ['U850', 'V850']
        return [req_in]

    @staticmethod
    def execute(o_port, data_in, req_in):
        # pass the incoming data through
        in_mesh = as_teca_cartesian_mesh(data_in[0])
        out_mesh = teca_cartesian_mesh.New()
        out_mesh.shallow_copy(in_mesh)
        # pull the arrays we need out of the incoming dataset
        arrays = out_mesh.get_point_arrays()
        u = arrays['U850']
        v = arrays['V850']
        # compute the derived quantity
        w = np.sqrt(u*u + v*v)
        # add it to the output
        arrays['wind_speed_850'] = w
        # return the dataset
        return out_mesh

# build the pipleine starting with a NetCDF CF-2 reader
cfr = teca_cf_reader.New()
cfr.set_files_regex('cam5_1_amip_run2\.cam2\.h2\.*')
cfr.set_x_axis_variable('lon')
cfr.set_y_axis_variable('lat')
cfr.set_t_axis_variable('time')

# add our wind speed computation
alg = teca_programmable_algorithm.New()
alg.set_report_callback(wind_speed.report)
alg.set_request_callback(wind_speed.request)
alg.set_execute_callback(wind_speed.execute)
alg.set_input_connection(cfr.get_output_port())

# add the writer
wri = teca_vtk_cartesian_mesh_writer.New()
wri.set_input_connection(alg.get_output_port())
wri.set_file_name('amip_run2_%t%.vtk')

# configure the executive. this will generate a request for each time step.
exe = teca_time_step_executive.New()
exe.set_first_step(0)
exe.set_last_step(-1)
wri.set_executive(exe)

# execute the pipeline
wri.update()