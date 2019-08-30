import sys
import os
from teca import *
from mpi4py import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_ranks = comm.Get_size()

# a reduction that writes its inputs to files
def get_reduce(file_template):
    def reduce(left, right):
        # construct the nested pipeline
        # serve up both inputs
        source = teca_dataset_source.New()
        ltab = as_teca_table(left)
        if ltab is not None:
            md = ltab.get_metadata()
            sys.stderr.write('%s writing table %d\n'%( \
                teca_parallel_id(), md[md['index_request_key']]))
            source.set_dataset(ltab)
        rtab = as_teca_table(right)
        if rtab is not None:
            md = rtab.get_metadata()
            sys.stderr.write('%s writing table %d\n'%( \
                teca_parallel_id(), md[md['index_request_key']]))
            source.set_dataset(rtab)
        # write data
        writer = teca_table_writer.New()
        writer.set_communicator(MPI.COMM_SELF)
        writer.set_input_connection(source.get_output_port())
        writer.set_executive(teca_index_executive.New())
        writer.set_file_name(file_template)
        # run the nested pipeline
        writer.update()
        # return nothing
        return None
    return reduce

if len(sys.argv) != 5:
    if rank == 0:
        sys.stderr.write('ERROR:\ntest_nested_pipeline.py ' \
            '[num rows] [num cols] [num tables] [num_threads]\n\n')
    sys.exit(-1)

num_rows = int(sys.argv[1])
num_cols = int(sys.argv[2])
num_tabs = int(sys.argv[3])
num_threads = int(sys.argv[4])

# generate a collection of tables to serve up
source = teca_dataset_source.New()
k = 0
while k < num_tabs:
    #sys.stderr.write('%s generating table %d\n'%(teca_parallel_id(),k))
    tab_id = (k + 1) * 1000
    tab = teca_table.New()
    tab.declare_columns(list(map(lambda x : chr(ord('A')+x), range(0,num_cols))), ['i']*num_cols)
    j = 0
    while j < num_rows:
        i = 0
        while i < num_cols:
            val = tab_id + j*num_cols + i
            tab << val
            i += 1
        j += 1
    source.append_dataset(tab)
    k += 1

mapper = teca_programmable_reduce.New()
mapper.set_name('table_writer')
mapper.set_input_connection(source.get_output_port())
mapper.set_verbose(1)
mapper.set_thread_pool_size(num_threads)
mapper.set_reduce_callback(get_reduce('test_nested_pipeline_%t%.csv'))

mapper.update()
