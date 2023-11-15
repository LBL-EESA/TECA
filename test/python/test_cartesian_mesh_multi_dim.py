import sys, os
from time import sleep
import platform
import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    pass
from teca import *


class generate_data(teca_python_algorithm):
    """
    This class generates 3D point centered data according to the function:

        f(x,y,z,t) = sin(x + t) * sin(y + t) * sin(z + t)

    and 2D data according to the function:

        g(x,y,t) = cos(x + t) * cos(y + t)
    """
    def __init__(self):
        self.verbose = 0

    def set_threshold(self, val):
        self.threshold = val

    def set_verbose(self, val):
        self.verbose = val

    def print_status(self, msg):
        if self.verbose:
            rank = MPI.COMM_WORLD.Get_rank()
            sys.stderr.write('[%d] generate_data::%s\n'%(rank, msg))

    def get_point_array_names(self):
        return ['f','g']

    def report(self, port, md_in):
        self.print_status('report(override)')

        # report arrays we generate
        md_out = teca_metadata(md_in[0])
        try:
            arrays = md_out['arrays']
        except:
            arrays = []
        md_out['arrays'] = arrays + ['f','g']

        # get the extent of the dataset
        wext = md_out['whole_extent']
        ncells = (wext[1] - wext[0] + 1)* \
                 (wext[3] - wext[2] + 1)*(wext[5] - wext[4] + 1)

        # create the metadata for the writer
        faa = teca_array_attributes(
            teca_double_array_code.get(),
            teca_array_attributes.point_centering,
            int(ncells), (1,1,1,1), 'none', 'f(x,y,z,t)',
            'sin(x + t) * sin(y + t) * sin(z + t)',
            None)

        fatts = faa.to_metadata()

        gaa = teca_array_attributes(
            teca_double_array_code.get(),
            teca_array_attributes.point_centering,
            int(ncells), (1,1,0,1), 'none', 'g(x,y,t)',
            'sin(x + t) * cos(x + t)',
            None)

        gatts = gaa.to_metadata()

        # put it in the array attributes
        try:
            atts = md_out['attributes']
        except:
            atts = teca_metadata()

        atts['f'] = fatts
        atts['g'] = gatts

        md_out['attributes'] = atts
        return md_out

    def execute(self, port, data_in, req_in):

        mesh_in = as_const_teca_cartesian_mesh(data_in[0])

        self.print_status('execute time=%g step=%d'%(
            mesh_in.get_time(), mesh_in.get_time_step()))

        # get mesh dims and coordinate arrays
        ext = mesh_in.get_extent()
        nx = ext[3] - ext[2] + 1
        ny = ext[1] - ext[0] + 1
        nz = ext[5] - ext[4] + 1

        x = mesh_in.get_x_coordinates().get_host_accessible()
        y = mesh_in.get_y_coordinates().get_host_accessible()
        z = mesh_in.get_z_coordinates().get_host_accessible()
        t = mesh_in.get_time()

        # generate the 3D variable. f = sin^2(x*y*z + t)
        f = np.empty((nz,ny,nx), dtype=np.float64)
        k = 0
        while k < nz:
            j = 0
            while j < ny:
                f[k,j,:] = np.sin(x[:] + t)*np.sin(y[j] + t)*np.sin(z[k] + t)
                j += 1
            k += 1

        # generate the 2D variable. g = sin(x + t) * exp(-y)
        g = np.empty((ny,nx), dtype=np.float64)
        j = 0
        while j < ny:
            g[j,:] = np.cos(y[j] + t)*np.cos(x[:] + t)
            j += 1

        # create the output and add in the arrays
        mesh_out = teca_cartesian_mesh.New()
        mesh_out.shallow_copy(mesh_in)
        mesh_out.get_point_arrays().append('f', teca_variant_array.New(f))
        mesh_out.get_point_arrays().append('g', teca_variant_array.New(g))

        return mesh_out



# process the command line
if not len(sys.argv) == 6:
    sys.stderr.write('usage: a.out [nx] [nz] [nt] [out file] [verbose]\n')
    sys.exit(-1)

nx = int(sys.argv[1])
nz = int(sys.argv[2])
n_steps = int(sys.argv[3])
out_file = sys.argv[4]
vrb = int(sys.argv[5])
pi2 = 2*np.pi

# construct a small mesh
src = teca_cartesian_mesh_source.New()
src.set_x_axis_variable('x')
src.set_y_axis_variable('y')
src.set_z_axis_variable('z')
src.set_t_axis_variable('t')
src.set_whole_extents([0, nx -1, 0, nx - 1, 0, nz - 1, 0, n_steps-1])
src.set_bounds([-pi2, pi2, -pi2, pi2, 0., 10., 0., pi2])
src.set_calendar('standard', 'days since 2022-01-18')

# generate 2d and 3d data
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
cfw.set_point_arrays(gd.get_point_array_names())
cfw.set_layout('number_of_steps')
cfw.set_steps_per_file(n_steps)
cfw.set_thread_pool_size(1)
cfw.set_executive(wex)
cfw.update()
