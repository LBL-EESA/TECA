import os
import sys
import socket
import numpy as np
import torch
import torch.nn.functional as F

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

        self.model = None
        self.model_path = None
        self.device = 'cpu'
        self.threads_per_core = 1
        self.verbose = 0

    def __str__(self):
        ms_str = 'input_variable=%s, output_variable=%s\n' % (
                  self.input_variable, self.output_variable)
        ms_str += 'model=%s\n' % (str(self.model))
        ms_str += 'model_path=%s\n' % (self.model_path)
        ms_str += 'device=%s\n' % (str(self.device))
        ms_str += 'thread pool size=%d\n' % (self.get_thread_pool_size())
        ms_str += 'hostname=%s, pid=%s' % (
                  socket.gethostname(), os.getpid())

        return ms_str

    def load_state_dict(self, filename):
        """
        Load only the pytorch state_dict parameters file only
        once and broadcast it to all ranks
        """
        event = teca_time_py_event('teca_pytorch_algorithm::load_state_dict')

        comm = self.get_communicator()
        rank = comm.Get_rank()

        sd = None
        if rank == 0:
            sd = torch.load(filename,
                            map_location=lambda storage, loc: storage)

        sd = comm.bcast(sd, root=0)

        return sd

    def set_verbose(self, val):
        """
        Set the verbosity of the run, higher values will result in more
        terminal output
        """
        self.verbose = val

    def set_input_variable(self, name):
        """
        set the variable name that will be inputed to the model
        """
        self.input_variable = name

        if self.output_variable is None:
            self.set_output_variable(self.input_variable + '_pred')

    def set_output_variable(self, name):
        """
        set the variable name that will be the output to the model
        """
        self.output_variable = name

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

    def set_threads_per_core(self, threads_per_core):
        """
        Set the number of threds per core. Typically 1 gives the best
        performance for CPU bound applicaitons, however 2 can sometimes
        be useful. More than 2 will likeley harm performance.
        """
        if threads_per_core > 4:
            raise ValueError('Using more than 2 threads per phyiscal core '
                             'will degrade performance on most architectures.')
        self.threads_per_core = threads_per_core

    def set_thread_pool_size(self, n_requested):
        """
        Sets the number of threads used for intra-op parallelism on CPU
        """
        event = teca_time_py_event('teca_pytorch_algorithm::set_thread_pool_size')

        rank = 0
        n_ranks = 1
        comm = self.get_communicator()
        if get_teca_has_mpi():
            rank = comm.Get_rank()
            n_ranks = comm.Get_size()

        n_threads = n_requested

        if n_requested > 0:
            # pass directly to torch
            torch.set_num_threads(n_requested)
        else:
            # detemrmine the number of physical cores are available
            # on this node, accounting for all MPI ranks scheduled to
            # run here.
            try:
                n_threads, affinity = \
                    thread_util.thread_parameters(comm, -1, 0, 0)

                # make use of hyper-threads
                n_threads *= self.threads_per_core

                # pass to torch
                torch.set_num_threads(n_threads)

            except(RuntimeError):
                # we failed to detect the number of physical cores per MPI rank
                # if this is an MPI job then fall back to 2 threads per rank
                # and if not let torch use all (what happens when you do
                # nothing)
                n_threads = -1
                comm = self.get_communicator()
                if get_teca_has_mpi() and comm.Get_size() > 1:
                    torch.set_num_threads(2)
                    n_threads = 2
                if rank == 0:
                    sys.stderr.write('STATUS: Failed to determine the number '
                                     'of physical cores available per MPI '
                                     'rank.\n')
                    sys.stderr.flush()

        # print a report describing the load balancing decisions
        if self.verbose:
            if get_teca_has_mpi():
                thread_map = comm.gather(n_threads, root=0)
            else:
                thread_map = [n_threads]
            if rank == 0:
                sys.stderr.write('STATUS: pytorch_algorithm thread '
                                 'parameters :\n')
                for i in range(n_ranks):
                    sys.stderr.write('  %d : %d\n' % (i, thread_map[i]))
                sys.stderr.flush()

    def get_thread_pool_size(self):
        """
        Gets the number of threads available for intra-op parallelism on CPU
        """
        return torch.get_num_threads()

    def set_torch_device(self, device="cpu"):
        """
        Set device to either 'cuda' or 'cpu'
        """
        if device[:4] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError('Failed to set device to CUDA. '
                               'CUDA is not available')

        self.device = device

    def set_model(self, model):
        """
        set Pytorch pretrained model
        """
        event = teca_time_py_event('teca_pytorch_algorithm::set_model')

        self.model = model
        self.model.eval()

    def report(self, port, rep_in):
        """ TECA report override """
        event = teca_time_py_event('teca_pytorch_algorithm::report')

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

        return rep

    def request(self, port, md_in, req_in):
        """ TECA request override """
        event = teca_time_py_event('teca_pytorch_algorithm::request')

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

        # get the input array and reshape it to a 2D layout that's compatible
        # with numpy and torch
        in_mesh = as_teca_cartesian_mesh(data_in[0])

        if in_mesh is None:
            raise RuntimeError('empty input, or not a mesh')

        arrays = in_mesh.get_point_arrays()
        in_va = arrays[self.input_variable]

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
