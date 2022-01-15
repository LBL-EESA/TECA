import sys
from mpi4py import MPI
from teca import *
import numpy

if get_teca_has_cupy():
    import cupy

class descriptive_stats(teca_python_algorithm):
    """ This class illustrates extending TECA to do a simple calculation. The
    class Computes the global min, max, average, and quartiles of a list of
    user provided variables. The results are stored in a table. When used with
    the teca_table_reduce execution engine and MPI the calculations are
    parallelized over the available CPU cores. If TECA was compiled with CUDA
    support the calculations will also make use of the available GPUs. A table
    with one per time step is generated. """

    def __init__(self):
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.have_cuda = get_teca_has_cuda() and get_teca_has_cupy()
        self.var_names = []

    def set_variable_names(self, vn):
        """ set a list of variables to process """
        self.var_names = vn

    def request(self, port, md_in, req_in):
        """ TECA pipeline request phase. """
        if self.get_verbose():
            sys.stderr.write('descriptive_stats::request MPI %d\n'%(self.rank))
        # request the variables to process.
        req = teca_metadata(req_in)
        req['arrays'] = self.var_names
        return [req]

    def execute(self, port, data_in, req):
        """ TECA pipeline execute phase. """

        # get the mesh object to process. This could be a Cartesian mesh with a
        # lat lon grid, or a Arakawa C grid, or some other mesh type. By
        # accessing the data through the common base class teca_mesh, we can
        # process the different mesh types using the same code. One only needs
        # to access through the derived type when one needs the geometric
        # information such as mesh coordinates.
        mesh = as_teca_cartesian_mesh(data_in[0])

        # the default will be to use Numpy on the CPU. The variable 'np' is used
        # to select either the Numpy library or the Cupy library for subsequent
        # calculations
        dev = -1
        np = numpy

        # use CUDA via Cupy if it is available and the execution engine assigned
        # this step to a GPU. device_id will be -1 if we are assigned a CPU core
        # other wise device_id will hold the assigned CUDA device.
        if self.have_cuda:
            dev = req['device_id']
            if dev >= 0:
                cupy.cuda.Device(dev).use()
                np = cupy

        # report
        if self.get_verbose():
            dev_name = 'CPU' if dev < 0 else 'GPU %d'%(dev)
            sys.stderr.write('descriptive_stats::execute MPI %d \n'
                             % (self.rank, dev_name))

        # create a table for the results of the calculation. there will be one
        # row of data for each simulation time step processed. each row will
        # contain the time step, the time, and a column for each statistic
        # (min, max, average, quartiles). This execute producers one row of the
        # table.
        table = teca_table.New()

        table.set_calendar(mesh.get_calendar())
        table.set_time_units(mesh.get_time_units())

        table.declare_columns(['step','time'], ['ul','d'])
        table << mesh.get_time_step() << mesh.get_time()

        # process each of the user provided variables
        for var_name in self.var_names:

            # get the variable data as a Numpy/Cupy compatible array. These
            # calls make use of the Numpy array interface protocol, or in the
            # case of CUDA, the Numba CUDA array interface protocol, for
            # zero-copy transfer of data from TECA to Numpy/Cupy.
            va = mesh.get_point_arrays().get(var_name)
            if dev < 0:
                hva = va.get_cpu_accessible()
            else:
                hva = va.get_cuda_accessible()

            # do the calculations. np is set to either the Numpy module or the
            # Cupy module depending on which device we have been assigned. Note
            # that cupy percentile operation retruns a CUDA backed array and
            # needs to be moved to the host
            mn = np.min(hva)
            mx = np.max(hva)
            av = np.average(hva)
            dv = np.std(hva)
            qt = np.percentile(hva, [25.,50.,75.])
            lq = qt[0]
            md = qt[1]
            uq = qt[2]

            # create new table columns for the statistics on the current
            # variable by specifying a name and type for each column
            table.declare_columns(['min '+var_name, 'avg '+var_name, \
                'max '+var_name, 'std '+var_name, 'low_q '+var_name, \
                'med '+var_name, 'up_q '+var_name], ['d']*7)

            # store the results in the table. the stream insertion operator '<<'
            # works column by column and advances to the next row
            # automatically. here the new columns we just created are filled.
            # when all is said and done the table has a single row.
            table << mn << mx << av << dv << lq << md << uq

        return table
