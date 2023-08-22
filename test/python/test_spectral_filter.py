import sys,os
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    rank = 0

# give each MPI rank a GPU
if not 'TECA_RANKS_PER_DEVICE' in os.environ:
    os.environ['TECA_RANKS_PER_DEVICE'] = '-1'

from teca import *

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

have_cuda = get_teca_has_cuda() and get_teca_has_cupy()
if have_cuda:
    import cupy
else:
    import numpy


pi = 3.14159235658979

def get_number_of_seconds(unit):
    if 'seconds' in unit:
        return 1.
    elif 'minutes' in unit:
        return 60.
    elif 'hours' in  unit:
        return 60.*60.
    elif 'days' in unit:
        return 60.*60.*24.

    raise ValueError('Unsupported time axis units %s' % (unit))



class generate_data(teca_python_algorithm):
    """
    This class generates 4D point centered data according to the function:

        f(x,y,z,t) = sin( 2*pi*f1*t ) + sin( 2*pi*f2*t ) + ... sin ( 2*pi*fn*t )

    The values of f1, f2, ... fn are in units of Hz
    """
    def __init__(self):
        self.verbose = 0
        self.frequencies = []
        self.amplitudes = []
        self.bias = 0.

    def set_bias(self, bias):
        """ set the bias of the sin waves """
        self.bias = bias

    def set_amplitudes(self, amps):
        """ set the list of amplitudes of the sin waves """
        self.amplitudes = amps

    def set_frequencies(self, freqs):
        """ set the list of frequencies of the sin waves"""
        self.frequencies = freqs

    def print_status(self, msg):
        if self.verbose:
            rank = MPI.COMM_WORLD.Get_rank()
            sys.stderr.write('[%d] generate_data::%s\n'%(rank, msg))

    def get_point_array_names(self):
        return ['f_t']

    def report(self, port, md_in):
        self.print_status('report(override)')

        # report arrays we generate
        md_out = teca_metadata(md_in[0])
        try:
            arrays = md_out['arrays']
        except:
            arrays = []
        md_out['arrays'] = arrays + ['f_t']

        # get the extent of the dataset
        wext = md_out['whole_extent']
        ncells = (wext[1] - wext[0] + 1)* \
                 (wext[3] - wext[2] + 1)*(wext[5] - wext[4] + 1)

        # create the metadata for the writer
        faa = teca_array_attributes(
            teca_double_array_code.get(),
            teca_array_attributes.point_centering,
            int(ncells), (1,1,1,1), 'none', 'f(t)',
            'function of time',
            None)

        # put it in the array attributes
        try:
            atts = md_out['attributes']
        except:
            atts = teca_metadata()

        atts['f_t'] = faa.to_metadata()

        md_out['attributes'] = atts
        return md_out

    def execute(self, port, data_in, req_in):

        # get the device to execute on
        dev = -1
        npmod = numpy
        if have_cuda:
            dev = req_in['device_id']
            if dev >= 0:
                npmod = cupy
                cupy.cuda.Device(dev).use()

        # get the input
        mesh_in = as_const_teca_cartesian_mesh(data_in[0])

        # get the time units in seconds
        t_units = mesh_in.get_time_units()
        seconds_per = get_number_of_seconds(t_units)

        # get mesh dims and coordinate arrays
        ext = mesh_in.get_extent()
        nx = ext[3] - ext[2] + 1
        ny = ext[1] - ext[0] + 1
        nz = ext[5] - ext[4] + 1

        # get the time step size
        t_ext = mesh_in.get_temporal_extent()
        nt = t_ext[1] - t_ext[0] + 1

        t_bds = mesh_in.get_temporal_bounds()
        dt = seconds_per * ( t_bds[1] - t_bds[0] ) / ( nt - 1 )
        t0 = t_bds[0] * seconds_per

        # generate the output
        f_t = npmod.zeros((nt,nz,ny,nx), dtype=npmod.float64) + self.bias

        i = 0
        while i < nt:
            t = t0 + i*dt
            nf = len(self.frequencies)
            j = 0
            while j < nf:
                f = self.frequencies[j]
                A = self.amplitudes[j]
                f_t[i,:,:,:] += A * npmod.sin( 2*pi*f*t )
                j += 1
            i += 1

        # create the output and add in the arrays
        mesh_out = teca_cartesian_mesh.New()
        mesh_out.shallow_copy(mesh_in)
        mesh_out.get_point_arrays().append('f_t', teca_variant_array.New(f_t))

        return mesh_out





class plot_data(teca_python_algorithm):

    def __init__(self):
        self.filter_type = ''
        self.critical_freq = 0.
        self.filter_order = 0

    def set_filter_type(self, ftype, fcrit, forder):
        self.filter_type = ftype
        self.critical_freq = fcrit
        self.filter_order = int(forder)

    def execute(self, port, data_in, req_in):

        # get the device to execute on
        dev = -1
        npmod = numpy
        if have_cuda:
            dev = req_in['device_id']
            if dev >= 0:
                npmod = cupy
                cupy.cuda.Device(dev).use()

        # get the input
        mesh_in = as_const_teca_cartesian_mesh(data_in[0])

        # allocate the output and copy
        mesh_out = teca_cartesian_mesh.New()
        mesh_out.shallow_copy(mesh_in)

        #plot the original and transformed signal
        t_ext = mesh_out.get_temporal_extent()
        t_bds = mesh_out.get_temporal_bounds()

        nx,ny,nz,nt = mesh_out.get_array_shape('f_t')

        f_t_in = mesh_out.get_point_arrays().get('f_t').get_host_accessible()
        f_t_in.shape = (nt,nz,ny,nx)

        f_t_out = mesh_out.get_point_arrays().get('f_t_%s'%(ftype)).get_host_accessible()
        f_t_out.shape = (nt,nz,ny,nx)

        t = np.linspace(t_bds[0], t_bds[1], nt)

        plt.figure(figsize=(15,10))

        plt.subplot(2, 1, 1)
        plt.plot(t, f_t_in[:,0,0,0], 'r-')
        plt.ylabel('degrees F')
        plt.xlabel('%s'%(units))
        plt.title('Temperature', fontweight='bold')
        plt.grid(True)
        plt.xlim((t_bds[0], t_bds[1]))

        plt.subplot(2, 1, 2)
        plt.plot(t, f_t_out[:,0,0,0], 'g-', linewidth=2)
        plt.ylabel('degrees F')
        plt.xlabel('%s)'%(units))
        plt.title('%s(Temperature) order=%d f_c=%.2e Hz'%(self.filter_type, self.filter_order, self.critical_freq), fontweight='bold')
        plt.grid(True)
        plt.xlim((t_bds[0], t_bds[1]))

        plt.subplots_adjust(hspace=0.55, left=0.1, right=0.95)

        plt.savefig('test_spectral_filter_%s_%d-%d.png'%(ftype, t_ext[0], t_ext[1]), dpi=150)

        return mesh_out






# process the command line
if not len(sys.argv) == 18:
    sys.stderr.write('usage: a.out [nx] [ny] [nz] [nt] [t1] '
                     '[A0] [A1] [B] [T0] [T1] [TC] [units] '
                     '[filter type] [test type] [out file] [verbose]\n\n'
                     'f_t = A0 * sin( 2*pi*f0*t ) + A1 * sin( 2*pi*f1*t ) + B\n\n'
                     'f0 = 1 / T0\nf1 = 1 / T1\n'
                     'TC defines the filter\'s critical frequency 1 / TC\n'
                     'units is one of days, hours, minutes, or seconds\n'
                     'filter type is one of low_pass or high_pass\n'
                     'test type is one of analytic or emperic\n')
    sys.exit(-1)

nx = int(sys.argv[1])
ny = int(sys.argv[2])
nz = int(sys.argv[3])
nt = int(sys.argv[4])
t1 = float(sys.argv[5])
A0 = float(sys.argv[6])
A1 = float(sys.argv[7])
B = float(sys.argv[8])
T0 = float(sys.argv[9])
T1 = float(sys.argv[10])
TC = float(sys.argv[11])
units = sys.argv[12]
ftype = sys.argv[13]
order = float(sys.argv[14])
ttype = sys.argv[15]
out_file = sys.argv[16]
vrb = int(sys.argv[17])

seconds_per = get_number_of_seconds(units)
f0 = 1. / ( T0 * seconds_per )
f1 = 1. / ( T1 * seconds_per )
fc = 1. / ( TC * seconds_per )

# construct a small mesh
src = teca_cartesian_mesh_source.New()
src.set_x_axis_variable('x')
src.set_y_axis_variable('y')
src.set_z_axis_variable('z')
src.set_t_axis_variable('t')
src.set_whole_extents([0, nx - 1, 0, ny - 1, 0, nz - 1, 0, nt - 1])
src.set_bounds([0., 360., -90., 90., 0., 0., 0., t1])
src.set_calendar('standard', '%s since 2022-01-01 00:00:00'%(units))

# generate test data
gd = generate_data.New()
gd.set_input_connection(src.get_output_port())
gd.set_verbose(vrb)
gd.set_frequencies([f0, f1])
gd.set_amplitudes([A0, A1])
gd.set_bias(B)

# filter
filt = teca_spectral_filter.New()
filt.set_input_connection(gd.get_output_port())
filt.set_filter_parameters(ftype, order, TC, units)
filt.set_verbose(vrb)
filt.set_point_arrays(['f_t'])

# plot
plot = plot_data.New()
plot.set_input_connection(filt.get_output_port())


do_test = 1
if do_test:
    # run the test
    if rank == 0:
        sys.stdout.write('running test %s...\n' % (ttype))
    # run the test with a known solution. this test does not cover windowing, MPI,
    # or interaction with the writer.
    if ttype == 'analytic':

        exe = teca_spatial_executive.New()
        exe.set_temporal_partition_size(nt)

        dsc_out = teca_dataset_capture.New()
        dsc_out.set_input_connection(plot.get_output_port())
        dsc_out.set_executive(exe)
        dsc_out.update()

        mesh_out = as_teca_cartesian_mesh(dsc_out.get_dataset())

        f_t_out = mesh_out.get_point_arrays().get('f_t_%s'%(ftype)).get_host_accessible()
        f_t_out.shape = (nt,nz,ny,nx)

        t = np.linspace( 0., t1, nt)
        if ftype == 'low_pass':
            Y = A0 * np.sin( 2.*pi*f0*t*seconds_per ) + B
        else:
            Y = A1 * np.sin( 2.*pi*f1*t*seconds_per )

        mse = np.sum( (Y - f_t_out[:,0,0,0])**2 ) / nt
        sys.stderr.write('MSE = %g\n' % (mse))

        if mse > 0.04:
            sys.stderr.write('ERROR: Test failed MSE > 0.04\n')
            sys.exit(-1)
    else:
        # compare against the recorded solution. tests windowing and MPI parallel.
        baseline_reader = teca_cf_reader.New()
        baseline_reader.set_files_regex(out_file + '.*\.nc$')
        baseline_reader.set_x_axis_variable('x')
        baseline_reader.set_y_axis_variable('y')
        baseline_reader.set_z_axis_variable('z')
        baseline_reader.set_t_axis_variable('t')

        exe = teca_spatial_executive.New()
        exe.set_temporal_partition_size(nt//2)
        exe.set_verbose(vrb)
        exe.set_arrays(['f_t_%s'%(ftype)])

        diff = teca_dataset_diff.New()
        diff.set_input_connection(0, baseline_reader.get_output_port())
        diff.set_input_connection(1, plot.get_output_port())
        diff.set_executive(exe)
        diff.set_verbose(vrb)
        diff.update()


else:
    # make a baseline
    if rank == 0:
        sys.stdout.write('generating baseline %s...\n'%(out_file))

    # write the data
    wex = teca_index_executive.New()
    wex.set_verbose(vrb)

    cfw = teca_cf_writer.New()
    cfw.set_input_connection(plot.get_output_port())
    cfw.set_partitioner_to_spatial()
    cfw.set_temporal_partition_size(nt//2)
    cfw.set_verbose(1)
    cfw.set_file_name(out_file + '_%t%.nc')
    cfw.set_layout_to_yearly()
    #cfw.set_layout_to_number_of_steps()
    #cfw.set_steps_per_file(nt)
    cfw.set_point_arrays(['f_t_%s'%(ftype)])
    cfw.set_thread_pool_size(1)
    cfw.set_executive(wex)
    cfw.update()
