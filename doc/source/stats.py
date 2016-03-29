from mpi4py import *
rank = MPI.COMM_WORLD.Get_rank()
n_ranks = MPI.COMM_WORLD.Get_size()
from teca import *
import sys
import stats_callbacks

if len(sys.argv) < 7:
    sys.stderr.write('global_stats.py [dataset regex] ' \
        '[out file name] [first step] [last step] [n threads]' \
        '[array 1] .. [ array n]\n')
    sys.exit(-1)

data_regex = sys.argv[1]
out_file = sys.argv[2]
first_step = int(sys.argv[3])
last_step = int(sys.argv[4])
n_threads = int(sys.argv[5])
var_names = sys.argv[6:]

if (rank == 0):
    sys.stderr.write('Testing on %d MPI processes\n'%(n_ranks))

cfr = teca_cf_reader.New()
cfr.set_files_regex(data_regex)

alg = teca_programmable_algorithm.New()
alg.set_input_connection(cfr.get_output_port())
alg.set_request_callback(stats_callbacks.get_request_callback(rank, var_names))
alg.set_execute_callback(stats_callbacks.get_execute_callback(rank, var_names))

mr = teca_table_reduce.New()
mr.set_input_connection(alg.get_output_port())
mr.set_first_step(first_step)
mr.set_last_step(last_step)
mr.set_thread_pool_size(n_threads)

tw = teca_table_writer.New()
tw.set_input_connection(mr.get_output_port())
tw.set_file_name(out_file)

tw.update()