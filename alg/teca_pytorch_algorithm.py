import os
import sys
from socket import gethostname
import numpy as np

class teca_pytorch_algorithm(teca_python_algorithm):
    """
    A TECA algorithm that provides access to torch. To use this class, derive
    a new class from it and from your class:

    1. call set input_/output_variable. this tells the pytorch_algorithm
       which array to process and how to name the result.

    2. call set_model. this installs your torch model. Use load_state_dict
       to load state dict from the file system in parallel.

    3. override preprocess. The input numpy array is passed in.  return the
       array to send to torch after applying any preprocessing or transforms.

    4. override postprocess. the tensor returned from torch is passed. return a
       numpy array with the correct mesh dimensions

    5. Optionally override the usual teca_python_algorithm methods as needed.

    """
    def __init__(self):

        self.input_variable = None
        self.output_variable = None
        self.output_variable_atts = None

        self.model = None
        self.model_path = None
        self.device = 'cpu'
        self.n_threads = -1
        self.n_threads_max = 4
        self.verbose = 0
        self.initialized = False

    def set_verbose(self, val):
        """
        Set the verbosity of the run, higher values will result in more
        terminal output
        """
        self.verbose = val

    def set_input_variable(self, name):
        """
        set the name of the variable to be processed
        """
        self.input_variable = name

    def set_output_variable(self, name, atts):
        """
        set the variable name to store the results under and
        its attributes. Attributes are optional and may be None
        but are required for the CF writer to write the result
        to disk.
        """
        self.output_variable = name
        self.output_variable_atts = atts

    def set_thread_pool_size(self, val):
        """
        Set the number of threads in each rank's thread pool. Setting
        to a value of -1 will result in the thread pool being sized
        such that each thread is uniquely and exclusively bound to a
        specific core accounting for thread pools in other ranks
        running on the same node
        """
        self.n_threads = val

    def set_max_thread_pool_size(self, val):
        """
        Set aniupper bound on the thread pool size. This is applied
        during automatic thread pool sizing.
        """
        self.n_threads_max = val

    def set_target_device(self, val):
        """
        Set the target device. May be one of 'cpu' or 'cuda'.
        """
        if val == 'cpu' or val == 'cuda':
            self.device = val
        else:
            raise RuntimeError('Invalid target device %s' % (val))

    def set_model(self, model):
        """
        set PyTorch model
        """
        self.model = model

    def initialize(self):
        """
        determine the mapping to hardware for the current MPI layout.
        if device is cpu then this configures OpenMP such that its
        thread pools have 1 thread per physical core.
        this also imports torch. this must be called prior to using any
        torch api's etc.
        """
        event = teca_time_py_event('teca_pytorch_algorithm::initialize')

        if self.initialized:
            return

        rank = 0
        n_ranks = 1
        comm = self.get_communicator()
        if get_teca_has_mpi():
            rank = comm.Get_rank()
            n_ranks = comm.Get_size()

        # tell OpenMP to report on what it does
        if self.verbose > 2:
            os.putenv('OMP_DISPLAY_ENV', 'true')

        # check for user specified OpenMP environment configuration
        omp_num_threads = os.getenv('OMP_NUM_THREADS')
        omp_places = os.getenv('OMP_PLACES')
        omp_proc_bind = os.getenv('OMP_PROC_BIND')
        if omp_num_threads is not None or omp_places is not None \
            or omp_proc_bind is not None:

            # at least one of the OpenMP environment control variables
            # was set. we will now bail out and use those settings
            if rank == 0:
                sys.stderr.write('[0] STATUS: OpenMP environment override '
                                 'detected. OMP_NUM_THREADS=%s '
                                 'OMP_PROC_BIND=%s OMP_PLACES=%s\n' % (
                                 str(omp_num_threads), str(omp_proc_bind),
                                 str(omp_places)))
                sys.stderr.flush()

            n_threads = 0

        else:
            # we will set the OpenMP control envirnment variables
            # detemrmine the number of physical cores are available
            # on this node, accounting for all MPI ranks scheduled to
            # run here.
            try:
                # let the user request a specific number of threads
                n_threads = self.n_threads

                n_threads, affinity, device_ids = \
                    thread_util.thread_parameters(comm, n_threads, 1, -1, -1,
                                                  0 if self.verbose < 2 else 1)

                # let the user request a bound on the number of threads
                if self.n_threads_max > 0:
                    n_threads = min(n_threads, self.n_threads_max)

                # construct the places list explicitly
                places = '{%d}'%(affinity[0])
                i = 1
                while i < n_threads:
                    places += ',{%d}'%(affinity[i])
                    i += 1

                os.putenv('OMP_NUM_THREADS', '%d'%(n_threads))
                os.putenv('OMP_PROC_BIND', 'true')
                os.putenv('OMP_PLACES', places)

                if self.verbose:
                    sys.stderr.write('[%d] STATUS: %s : %d : OMP_NUM_THREADS=%d'
                                     ' OMP_PROC_BIND=true OMP_PLACES=%s\n' % (
                                     rank, gethostname(), rank, n_threads,
                                     places))
                    sys.stderr.flush()

            except(RuntimeError):
                # we failed to detect the number of physical cores per MPI rank
                os.putenv('OMP_NUM_THREADS', '1')
                n_threads = 1

                sys.stderr.write('[0] STATUS: Failed to determine the '
                                 'number of physical cores available per '
                                 'MPI rank. OMP_NUM_THREADS=1\n')
                sys.stderr.flush()

        global torch
        import torch

        if n_threads:
            # also tell torch explicitly
            torch.set_num_threads(n_threads)
            torch.set_num_interop_threads(n_threads)

        if 'cuda' in self.device:
            # check that CUDA is present
            if torch.cuda.is_available():
                # get the number of devices and assign them to ranks round
                # robin
                n_dev = torch.cuda.device_count()
                dev_id = rank % n_dev

                if self.device == 'cuda':
                    # select the GPU that this rank will use.
                    self.device = 'cuda:%d' % (dev_id)

                if self.verbose:
                    dev_name = torch.cuda.get_device_name(self.device)

                    sys.stderr.write('[%d] STATUS: %s : %d : %d/%d : %s\n' % (
                                     rank, gethostname(), rank, dev_id, n_dev,
                                     dev_name))
                    sys.stderr.flush()
            else:
                # fall back to OpenMP
                if rank == 0:
                   sys.stderr.write('[%d] WARNING: CUDA was requested but is not'
                                    ' available. OpenMP will be used.\n')
                   sys.stderr.flush()

                self.device = 'cpu'

        self.initialized = True

    def check_initialized(self):
        """
        verify that the user called initialize
        """
        if not self.initialized:
            raise RuntimeError('Not initialized! call '
                               'teca_pytroch_algorithm::initialize before '
                               'use to configure OpenMP and import torch')

    def load_state_dict(self, filename):
        """
        Load only the pytorch state_dict parameters file.
        """
        event = teca_time_py_event('teca_pytorch_algorithm::load_state_dict')

        self.check_initialized()

        comm = self.get_communicator()
        rank = comm.Get_rank()

        sd = None
        if rank == 0:
            sd = torch.load(filename, map_location='cpu')

        sd = comm.bcast(sd, root=0)

        return sd

    def load_model(self, filename, model):
        """
        Load the state dict named by 'filename' and install them into the
        passed model instance 'model'. This also moves the model on the current
        target device, and puts the model into inference mode.
        """
        event = teca_time_py_event('teca_pytorch_algorithm::load_model')

        self.check_initialized()

        # load the model weights from disk
        model_state = self.load_state_dict(filename)

        # install weights, send to target device, run in inference mode
        model.load_state_dict(model_state)
        model.to(self.device)
        model.eval()

        self.model = model

    def preprocess(self, in_array):
        """
        Override this to preprocess the passed in array before it is passed to
        torch. The passed array has the shape of the input/output mesh. the
        default implementation does nothing.
        """
        return in_array

    def postprocess(self, out_tensor):
        """
        Override this to postprocess the tensor data returned from torch.
        return the result as a numpy array. the return should be sized
        compatibly with the output mesh. The default implementation converts
        the tensor to a ndarray.
        """
        return out_tensor.numpy()

    def report(self, port, rep_in):
        """ TECA report override """
        event = teca_time_py_event('teca_pytorch_algorithm::report')

        self.check_initialized()

        # check for required parameters.
        if self.model is None:
            raise RuntimeError('A torch model has not been specified')

        if self.input_variable is None:
            raise RuntimeError('input_variable has not been specified')

        if self.output_variable is None:
            raise RuntimeError('output_variable has not been specified')

        # add the variable we proeduce to the report
        rep = teca_metadata(rep_in[0])

        if rep.has('variables'):
            rep.append('variables', self.output_variable)
        else:
            rep.set('variables', self.output_variable)

        attributes = rep["attributes"]
        attributes[self.output_variable] = self.output_variable_atts.to_metadata()
        rep["attributes"] = attributes

        return rep

    def request(self, port, md_in, req_in):
        """ TECA request override """
        event = teca_time_py_event('teca_pytorch_algorithm::request')

        self.check_initialized()

        req = teca_metadata(req_in)

        arrays = []
        if req.has('arrays'):
            arrays = req['arrays']
            if type(arrays) != list:
                arrays = [arrays]

        # remove the arrays we produce
        try:
            arrays.remove(self.output_variable)
        except(Exception):
            pass

        # add the arrays we need
        arrays.append(self.input_variable)

        req['arrays'] = arrays

        return [req]

    def execute(self, port, data_in, req):
        """ TECA execute override """
        event = teca_time_py_event('teca_pytorch_algorithm::execute')

        self.check_initialized()

        # get the input array and reshape it to a 2D layout that's compatible
        # with numpy and torch
        in_mesh = as_teca_cartesian_mesh(data_in[0])

        if in_mesh is None:
            raise RuntimeError('empty input, or not a mesh')

        arrays = in_mesh.get_point_arrays()
        in_va = arrays[self.input_variable].get_host_accessible()

        ext = in_mesh.get_extent()
        in_va.shape = (ext[3] - ext[2] + 1,
                       ext[1] - ext[0] + 1)

        # let the derived class do model specific preprocessing
        in_array = self.preprocess(in_va)

        # send to torch for processing
        in_tensor = torch.from_numpy(in_array).to(self.device)

        with torch.no_grad():
            out_tensor = self.model(in_tensor)

        if out_tensor is None:
            raise RuntimeError("Model failed to get predictions")

        # let the derived class do model specific posprocessing
        out_array = self.postprocess(out_tensor)

        # build the output
        out_mesh = teca_cartesian_mesh.New()
        out_mesh.shallow_copy(in_mesh)

        out_va = teca_variant_array.New(out_array)
        out_mesh.get_point_arrays().set(self.output_variable, out_va)

        return out_mesh
