try:
    from mpi4py import *
    rank = MPI.COMM_WORLD.Get_rank()
    n_ranks = MPI.COMM_WORLD.Get_size()
except:
    rank = 0
    n_ranks = 1
from teca import *
import numpy as np
import sys

if len(sys.argv) < 7:
    sys.stderr.write('test_map_reduce.py [dataset regex] ' \
        '[out file name] [first step] [last step] [n threads]' \
        '[array 1] .. [ array n]\n')
    sys.exit(-1)

data_regex = sys.argv[1]
out_file = sys.argv[2]
first_step = int(sys.argv[3])
last_step = int(sys.argv[4])
n_threads = int(sys.argv[5])
var_names = sys.argv[6:]

class descriptive_stats:
    @staticmethod
    def request(port, md_in, req_in):
        global var_names
        req = teca_metadata(req_in)
        req['arrays'] = var_names
        return [req]

    @staticmethod
    def execute(port, data_in, req):
        global var_names, rank
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

if (rank == 0):
    sys.stderr.write('Testing on %d MPI processes\n'%(n_ranks))

cfr = teca_cf_reader.New()
cfr.set_files_regex(data_regex)

stats = teca_programmable_algorithm.New()
stats.set_request_callback(descriptive_stats.request)
stats.set_execute_callback(descriptive_stats.execute)
stats.set_input_connection(cfr.get_output_port())

mr = teca_table_reduce.New()
mr.set_input_connection(stats.get_output_port())
mr.set_first_step(first_step)
mr.set_last_step(last_step)
mr.set_thread_pool_size(n_threads)

tw = teca_table_writer.New()
tw.set_input_connection(mr.get_output_port())
tw.set_file_name(out_file)

tw.update()
