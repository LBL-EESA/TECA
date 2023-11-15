import sys, os
from time import sleep
import platform
import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    pass
from teca import *

set_stack_trace_on_error()

if len(sys.argv) != 6:
    sys.stderr.write('test_information_array_io.py [n points] [n steps] ' \
        '[steps per file] [baseline file] [baseline step]\n')
    sys.exit(-1)

n_threads = 1
nx = int(sys.argv[1])
n_steps = int(sys.argv[2])
steps_per_file = int(sys.argv[3])
baseline = sys.argv[4]
check_step = int(sys.argv[5])

out_file = 'py_test_cf_writer_collective-%t%.nc'
files_regex = 'py_test_cf_writer_collective.*\\.nc$'
vrb = 1


class generate_data(teca_python_algorithm):
    """
    This class generates point centered data according to the function:

        z = sin^2(x*y + t)

    Additionally the variable 'counts' holds the number of cells
    equal to or above the threshold in the first element and the number
    of cells below the threshold in the second.The variable information
    variable 'threshold' stores the threshold.
    """

    def __init__(self):
        self.verbose = 0
        self.threshold = 0.5

    def set_threshold(self, val):
        self.threshold = val

    def set_verbose(self, val):
        self.verbose = val

    def print_status(self, msg):
        if self.verbose:
            rank = MPI.COMM_WORLD.Get_rank()
            sys.stderr.write('[%d] generate_data::%s\n'%(rank, msg))

    def get_point_array_names(self):
        return ['z']

    def get_info_array_names(self):
        return ['counts', 'z_threshold']

    def report(self, port, md_in):
        self.print_status('report(override)')

        # report arrays we generate
        md_out = teca_metadata(md_in[0])
        try:
            arrays = md_out['arrays']
        except:
            arrays = []
        md_out['arrays'] = arrays + ['z']

        # get the extent of the dataset
        wext = md_out['whole_extent']
        ncells = (wext[1] - wext[0] + 1)* \
                 (wext[3] - wext[2] + 1)*(wext[5] - wext[4] + 1)

        # create the metadata for the writer
        z_atts = teca_array_attributes(
            teca_double_array_code.get(),
            teca_array_attributes.point_centering,
            int(ncells), (1,1,0,1), 'meters', 'height',
            'height is defined by the function z=sin^2(x*y + t)',
            None)

        zt_atts = teca_array_attributes(
            teca_double_array_code.get(),
            teca_array_attributes.no_centering,
            1, (0,0,0,0), 'meters', 'threshold height',
            'value of height used to segment the z data',
            None)

        count_atts = teca_array_attributes(
            teca_int_array_code.get(),
            teca_array_attributes.no_centering,
            2, (0,0,0,0), 'cells', 'number of cells',
            'number of cells above and below the threshold value',
            None)

        # put it in the array attributes
        try:
            atts = md_out['attributes']
        except:
            atts = teca_metadata()
        atts['z'] = z_atts.to_metadata()
        atts['z_threshold'] = zt_atts.to_metadata()
        atts['counts'] = count_atts.to_metadata()
        md_out['attributes'] = atts

        return md_out

    def execute(self, port, data_in, req_in):

        mesh_in = as_const_teca_cartesian_mesh(data_in[0])

        self.print_status('execute time=%g step=%d'%(
            mesh_in.get_time(), mesh_in.get_time_step()))

        # compute sin^2(x*y + t)
        ext = mesh_in.get_extent()
        shape = (ext[3] - ext[2] + 1, ext[1] - ext[0] + 1)

        X = mesh_in.get_x_coordinates().as_array()
        Y = mesh_in.get_y_coordinates().as_array()
        t = mesh_in.get_time()

        rad_per_deg = np.pi/180.0
        X *= rad_per_deg
        Y *= rad_per_deg

        z = np.empty(shape, dtype=np.float64)

        j = 0
        for y in Y:
            sxyt = np.sin(Y[j]*X[:] + t)
            sxyt *= sxyt
            z[j,:] = sxyt
            j += 1

        # compute number of cells above and below the threshold
        ii = np.where(z.ravel() >= self.threshold)[0]
        n_above = len(ii)
        n_below = shape[0]*shape[1] - n_above
        count = np.array([n_above, n_below], dtype=np.int32)

        # document the threshold value
        z_threshold = np.array([self.threshold], dtype=np.float64)

        # create the output and add in the arrays
        mesh_out = teca_cartesian_mesh.New()
        mesh_out.shallow_copy(mesh_in)
        mesh_out.get_point_arrays().append('z', teca_variant_array.New(z))
        mesh_out.get_information_arrays().append('z_threshold', teca_variant_array.New(z_threshold))
        mesh_out.get_information_arrays().append('counts', teca_variant_array.New(count))

        return mesh_out


class print_info_arrays(teca_python_algorithm):
    """
    This class prints the information arrays that we wrote
    """

    def __init__(self):
        self.verbose = 0

    def set_verbose(self, val):
        self.verbose = val

    def print_status(self, msg):
        if self.verbose:
            rank = MPI.COMM_WORLD.Get_rank()
            sys.stderr.write('[%d] print_info_arrays::%s\n'%(rank, msg))

    def execute(self, port, data_in, req_in):

        mesh_in = as_const_teca_cartesian_mesh(data_in[0])
        mesh_out = teca_cartesian_mesh.New()
        mesh_out.shallow_copy(mesh_in)

        t = mesh_out.get_time()
        s = mesh_out.get_time_step()
        self.print_status('execute t=%g, s=%d, counts=(%s), z_threshold=%s'%(
            t, s, str(mesh_out.get_information_arrays().get('counts')),
            str(mesh_out.get_information_arrays().get('z_threshold'))))

        return mesh_out


##############
# Pipeline 1
##############
# construct a small mesh
src = teca_cartesian_mesh_source.New()
src.set_whole_extents([0, nx -1, 0, nx - 1, 0, 0, 0, n_steps-1])
src.set_bounds([-90.0, 90.0, -90.0, 90.0, 0.0, 0.0, 0.0, 2.*np.pi])
src.set_calendar('standard', 'days since 2019-09-24')

# generate point and information data to be written and then read
# the point data is z = sin^2(x*y + t) thus correctness can be easily
# verified in ParaView or ncview etc.
gd = generate_data.New()
gd.set_input_connection(src.get_output_port())
gd.set_verbose(vrb)

# write the data
wex = teca_index_executive.New()
wex.set_verbose(vrb)

cfw = teca_cf_writer.New()
cfw.set_input_connection(gd.get_output_port())
cfw.set_verbose(1)
cfw.set_flush_files(1)
cfw.set_file_name(out_file)
cfw.set_information_arrays(gd.get_info_array_names())
cfw.set_point_arrays(gd.get_point_array_names())
cfw.set_layout('number_of_steps')
cfw.set_steps_per_file(steps_per_file)
cfw.set_thread_pool_size(n_threads)
cfw.set_executive(wex)
cfw.update()

MPI.COMM_WORLD.Barrier()
if platform.system() == 'Darwin':
    sleep(10)

##############
# Pipeline 2
##############
rank = MPI.COMM_WORLD.Get_rank()
if rank == 0:
    # read the data back in
    cfr = teca_cf_reader.New()
    cfr.set_communicator(MPI.COMM_SELF)
    cfr.set_files_regex(files_regex)

    # print it out
    par = print_info_arrays.New()
    par.set_input_connection(cfr.get_output_port())
    par.set_verbose(vrb)

    rex = teca_index_executive.New()
    rex.set_arrays(['z', 'z_threshold', 'counts'])
    rex.set_verbose(vrb)
    rex.set_start_index(check_step)
    rex.set_end_index(check_step)

    fn = file_util.replace_timestep(baseline, check_step)
    do_test = system_util.get_environment_variable_bool('TECA_DO_TEST', True)
    if do_test and os.path.exists(fn):
        sys.stderr.write('running the test...\n')

        cmr = teca_cartesian_mesh_reader.New()
        cmr.set_file_name(fn)

        diff = teca_dataset_diff.New()
        diff.set_communicator(MPI.COMM_SELF)
        diff.set_input_connection(0, cmr.get_output_port())
        diff.set_input_connection(1, par.get_output_port())
        diff.set_executive(rex)
        #diff.set_relative_tolerance(1.e-4)
        diff.update()
    else:
        sys.stderr.write('writing the baseline...\n')

        cmw = teca_cartesian_mesh_writer.New()
        cmw.set_communicator(MPI.COMM_SELF)
        cmw.set_file_name(baseline)
        cmw.set_input_connection(par.get_output_port())
        cmw.set_file_name(baseline)
        cmw.set_executive(rex)
        cmw.update()
