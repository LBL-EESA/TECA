from teca import *
import numpy as np
import sys

def get_request_callback(rank, var_names):
    def request(port, md_in, req_in):
        sys.stderr.write('descriptive_stats::request MPI %d\n'%(rank))
        req = teca_metadata(req_in)
        req['arrays'] = var_names
        return [req]
    return request

def get_execute_callback(rank, var_names):
    def execute(port, data_in, req):
        sys.stderr.write('descriptive_stats::execute MPI %d\n'%(rank))

        mesh = as_teca_cartesian_mesh(data_in[0])

        table = teca_table.New()
        table.declare_columns(['step','time'], ['ul','d'])
        table << mesh.get_time_step() << mesh.get_time()

        for var_name in var_names:

            table.declare_columns(['min '+var_name, 'avg '+var_name, \
                'max '+var_name, 'std '+var_name, 'low_q '+var_name, \
                'med '+var_name, 'up_q '+var_name], ['d']*7)

            var = mesh.get_point_arrays().get(var_name).as_array()

            table << float(np.min(var)) << float(np.average(var)) \
                << float(np.max(var)) << float(np.std(var)) \
                << map(float, np.percentile(var, [25.,50.,75.]))

        return table
    return execute
