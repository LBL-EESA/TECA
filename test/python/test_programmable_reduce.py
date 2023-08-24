try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    n_ranks = MPI.COMM_WORLD.Get_size()
except:
    sys.stderr.write('import mpi4py failed. running in serial...\n')
    rank = 0
    n_ranks = 1
from teca import *
import numpy
if get_teca_has_cupy():
    import cupy
import sys
import os

set_stack_trace_on_error()
set_stack_trace_on_mpi_error()

if len(sys.argv) < 6:
    sys.stderr.write('test_programmable_reduce.py [dataset regex] ' \
        '[out file name] [first step] [last step] ' \
        '[array 1] .. [ array n]\n')
    sys.exit(-1)

data_regex = sys.argv[1]
out_file = sys.argv[2]
start_index = int(sys.argv[3])
end_index = int(sys.argv[4])
var_names = sys.argv[5:]

# give each MPI rank a GPU
n_threads = 1
os.environ['TECA_RANKS_PER_DEVICE'] = '-1'

def get_request_callback(var_names):
    def request(port, md_in, req_in):
        req = teca_metadata(req_in)
        req['arrays'] = var_names
        return [req]
    return request

def get_execute_callback(rank, var_names):
    def execute(port, data_in, req):

        # select CPU or GPU
        dev = -1
        np = numpy
        alloc = variant_array_allocator_malloc
        if get_teca_has_cuda() and get_teca_has_cupy():
            dev = req['device_id']
            if dev >= 0:
                alloc = variant_array_allocator_cuda
                cupy.cuda.Device(dev).use()
                np = cupy

        # report
        dev_str = 'CPU' if dev < 0 else 'GPU %d'%(dev)
        sys.stderr.write('[%d] execute %s\n'%(rank, dev_str))

        # get the input
        mesh = as_teca_mesh(data_in[0])

        # get the output
        table = teca_table.New()
        table.copy_metadata(mesh)
        table.set_default_allocator(alloc)

        table.declare_columns(['step','time'], ['ul','d'])
        table << mesh.get_time_step() << mesh.get_time()

        for var_name in var_names:

            # get data on the CPU or the GPU
            va = mesh.get_point_arrays().get(var_name)
            if dev < 0:
                hva = va.get_host_accessible()
            else:
                hva = va.get_cuda_accessible()

            # do the calculations.
            mn = np.min(hva)
            mx = np.max(hva)
            av = np.average(hva)
            dv = np.std(hva)
            qt = np.percentile(hva, [25.,50.,75.])
            lq = qt[0]
            md = qt[1]
            uq = qt[2]

            # insert the results into the table
            table.declare_columns(['min '+var_name, 'avg '+var_name, \
                'max '+var_name, 'std '+var_name, 'low_q '+var_name, \
                'med '+var_name, 'up_q '+var_name], ['d']*7)

            table << mn << av << mx << dv << lq << md << uq

        return table
    return execute

def get_reduce_callback(rank):
    def reduce(dev, data_in_0, data_in_1):
        # select CPU or GPU
        np = numpy
        alloc = variant_array_allocator_malloc
        if dev >= 0 and get_teca_has_cuda() and get_teca_has_cupy():
            alloc = variant_array_allocator_cuda
            cupy.cuda.Device(dev).use()
            np = cupy

        # report
        dev_str = 'CPU' if dev < 0 else 'GPU %d'%(dev)
        sys.stderr.write('[%d] reduce %s\n'%(rank, dev_str))

        # get the input
        table_0 = as_teca_table(data_in_0)
        table_1 = as_teca_table(data_in_1)

        # reduce
        data_out = None
        if table_0 is not None and table_1 is not None:
            data_out = as_teca_table(table_0.new_copy(alloc))
            data_out.concatenate_rows(table_1)

        elif table_0 is not None:
            data_out = table_0.new_copy(alloc)

        elif table_1 is not None:
            data_out = table_1.new_copy(alloc)

        return data_out
    return reduce

def get_finalize_callback(rank):
    def finalize(dev, data):
        sys.stderr.write('[%d] finalize\n'%(rank))
        return data
    return finalize


if (rank == 0):
    sys.stderr.write('Testing on %d MPI processes %d threads\n'%(n_ranks, n_threads))

cfr = teca_cf_reader.New()
cfr.set_files_regex(data_regex)

coords = teca_normalize_coordinates.New()
coords.set_input_connection(cfr.get_output_port())

stats = teca_programmable_algorithm.New()
stats.set_name('descriptive_stats')
stats.set_request_callback(get_request_callback(var_names))
stats.set_execute_callback(get_execute_callback(rank, var_names))
stats.set_input_connection(coords.get_output_port())

mr = teca_programmable_reduce.New()
mr.set_name('table_reduce')
mr.set_input_connection(stats.get_output_port())
mr.set_start_index(start_index)
mr.set_end_index(end_index)
mr.set_thread_pool_size(n_threads)
mr.set_reduce_callback(get_reduce_callback(rank))
mr.set_finalize_callback(get_finalize_callback(rank))

sort = teca_table_sort.New()
sort.set_input_connection(mr.get_output_port())
sort.set_index_column('time')

cal = teca_table_calendar.New()
cal.set_input_connection(sort.get_output_port())

do_test = system_util.get_environment_variable_bool('TECA_DO_TEST', True)
if do_test and os.path.exists(out_file):
    if rank == 0:
        sys.stderr.write('running the test ... \n')
    tr = teca_table_reader.New()
    tr.set_file_name(out_file)

    diff = teca_dataset_diff.New()
    diff.set_input_connection(0, tr.get_output_port())
    diff.set_input_connection(1, cal.get_output_port())
    diff.update()
else:
    if rank == 0:
        sys.stderr.write('writing baseline %s ... \n'%(out_file))
    tw = teca_table_writer.New()
    tw.set_input_connection(cal.get_output_port())
    tw.set_file_name(out_file)

    tw.update()
