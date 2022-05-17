from mpi4py import MPI
from teca import *
import os
import sys


if not len(sys.argv) == 3:
    sys.stderr.write('test_apply_binary_mask.py [input_file] [baseline_file] \n')
    sys.exit(-1)
input_file = sys.argv[1]
baseline_file = sys.argv[2]


# create the reader
mesh_data_reader = teca_cf_reader.New()
mesh_data_reader.set_files_regex(input_file)

# create the gradient calculation
grad = teca_gradient.New()
grad.set_scalar_field('p')
grad.set_gradient_field_x('grad_p_x')
grad.set_gradient_field_y('grad_p_y')
grad.set_input_connection(mesh_data_reader.get_output_port())

# create the executive
exec = teca_index_executive.New()
point_arrays = ["grad_p_x","grad_p_y"]
exec.set_arrays(point_arrays)

# check whether we should be generating the test
do_test = 1
try:
    do_test = int(os.environ["TECA_DO_TEST"])
except:
    pass

# do the test if flagged
if do_test and os.path.exists(baseline_file):
    baseline_reader = teca_cf_reader.New()
    baseline_reader.set_files_regex(baseline_file)

    # check the difference with the baseline dataset
    diff = teca_dataset_diff.New()
    diff.set_input_connection(0, baseline_reader.get_output_port())
    diff.set_input_connection(1, grad.get_output_port())
    diff.set_executive(exec)
    diff.set_verbose(1)

    diff.update()
# otherwise generate the baseline file
else:
    writer = teca_cf_writer.New()
    writer.set_input_connection(grad.get_output_port())
    writer.set_thread_pool_size(1)
    #writer.set_executive(exec)
    writer.set_point_arrays(point_arrays)
    writer.set_file_name(baseline_file)
    writer.update()
    sys.exit(-1)

sys.exit(0)
