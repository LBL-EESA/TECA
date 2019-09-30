from mpi4py import *
from teca import *
import sys
from stats_callbacks import descriptive_stats

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

if MPI.COMM_WORLD.Get_rank() == 0:
    sys.stderr.write('Testing on %d MPI processes\n'%(MPI.COMM_WORLD.Get_size()))

cfr = teca_cf_reader.New()
cfr.set_files_regex(data_regex)

alg = descriptive_stats.New()
alg.set_input_connection(cfr.get_output_port())
alg.set_variable_names(var_names)

mr = teca_table_reduce.New()
mr.set_input_connection(alg.get_output_port())
mr.set_thread_pool_size(n_threads)
mr.set_start_index(first_step)
mr.set_end_index(last_step)

tw = teca_table_writer.New()
tw.set_input_connection(mr.get_output_port())
tw.set_file_name(out_file)

tw.update()
