from mpi4py import *
from teca import *
import sys

# import the user defined class extending TECA
from descriptive_stats import *
from plot_columns import *

# Process the command line.
if len(sys.argv) < 7:
    sys.stderr.write('global_stats.py [dataset regex] ' \
        '[out file] [first step] [last step] [n threads]' \
        '[array 1] .. [ array n]\n\n' \
        'dataset regex - a regular expression identifying the files to process\n' \
        'out file      - the name path of the file to store the results in\n' \
        'first step    - the first time step to process. Use 0 for all\n' \
        'last step     - the last time step to process. Use -1 for all\n' \
        'n threads     - the number of CPU threads to use. Use -1 for all\n' \
        'array 1 ... n - a list of variables to comput statistics for\n\n')
    sys.exit(-1)

data_regex = sys.argv[1]
out_file = sys.argv[2]
first_step = int(sys.argv[3])
last_step = int(sys.argv[4])
n_threads = int(sys.argv[5])
var_names = sys.argv[6:]

if MPI.COMM_WORLD.Get_rank() == 0:
    sys.stderr.write('stats.py running on %d MPI processes\n'
                     % (MPI.COMM_WORLD.Get_size()))

# Connect a NetCDF CF2 reader. This stage will serve up the climate simulation
# when the pipeline runs.
cfr = teca_cf_reader.New()
cfr.set_files_regex(data_regex)

# Connect a user defined pipeline stage computing global statistics on the list
# of variables provided on the command line.
alg = descriptive_stats.New()
alg.set_input_connection(cfr.get_output_port())
alg.set_variable_names(var_names)

# Connect the map reduce execution engine. This stage contains an internal load
# balancer that parallelizes the calculation across MPI ranks, CPU cores, and
# GPUs.
mr = teca_table_reduce.New()
mr.set_input_connection(alg.get_output_port())
mr.set_thread_pool_size(n_threads)
mr.set_start_index(first_step)
mr.set_end_index(last_step)

## sort the results in time. this is neccssary when running in parallel the
## results will be accumulated in a non-deterministic order when C++ threading
## is used.
#ts = teca_table_sort.New()
#ts.set_input_connection(mr.get_output_port())
#
## Connect a user defined pipeline stage plotting one of the arrays
#pc = plot_columns.New()
#pc.set_input_connection(ts.get_output_port())

# Connect a writer to store the resulting table (one row per time step) on disk.
tw = teca_table_writer.New()
tw.set_input_connection(mr.get_output_port())
tw.set_file_name(out_file)

# Run the pipeline.
tw.update()
