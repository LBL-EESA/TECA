from mpi4py import *
from teca import *
import numpy as np
import sys

class descriptive_stats(teca_python_algorithm):

    def __init__(self):
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.var_names = []

    def set_variable_names(self, vn):
        self.var_names = vn

    def request(self, port, md_in, req_in):
        sys.stderr.write('descriptive_stats::request MPI %d\n'%(self.rank))
        req = teca_metadata(req_in)
        req['arrays'] = self.var_names
        return [req]

    def execute(self, port, data_in, req):
        sys.stderr.write('descriptive_stats::execute MPI %d\n'%(self.rank))

        mesh = as_teca_cartesian_mesh(data_in[0])

        table = teca_table.New()
        table.declare_columns(['step','time'], ['ul','d'])
        table << mesh.get_time_step() << mesh.get_time()

        for var_name in self.var_names:

            table.declare_columns(['min '+var_name, 'avg '+var_name, \
                'max '+var_name, 'std '+var_name, 'low_q '+var_name, \
                'med '+var_name, 'up_q '+var_name], ['d']*7)

            var = mesh.get_point_arrays().get(var_name).as_array()

            table << np.min(var) << np.average(var) \
                << np.max(var) << np.std(var) \
                << np.percentile(var, [25.,50.,75.])

        return table
