import os
import sys
import socket
import teca_py
import numpy as np
import torch
import torch.nn.functional as F


class teca_model_segmentation(teca_py.teca_python_algorithm):
    """
    A generic TECA algorithm that provides torch based deep
    learning feature detecting algorithms with core TECA
    capabilities for easy integration.
    """
    def __init__(self):
        self.variable_name = None
        self.pred_name = None
        self.var_array = None
        self.pred_array = None
        self.inference_function = None
        self.model = None
        self.model_path = None
        self.device = 'cpu'
        self.threads_per_core = 1

    def __str__(self):
        ms_str = 'variable_name=%s, pred_name=%s\n' % (
                  self.variable_name, self.pred_name)
        ms_str += 'model=%s\n' % (str(self.model))
        ms_str += 'model_path=%s\n' % (self.model_path)
        ms_str += 'device=%s\n' % (str(self.device))
        ms_str += 'thread pool size=%d\n' % (self.get_thread_pool_size())
        ms_str += 'hostname=%s, pid=%s' % (
                  socket.gethostname(), os.getpid())

        return ms_str

    def load_state_dict(self, state_dict_file):
        """
        Load only the pytorch state_dict parameters file only
        once and broadcast it to all ranks
        """
        comm = self.get_communicator()
        rank = comm.Get_rank()

        sd = None
        if rank == 0:
            sd = torch.load(state_dict_file,
                            map_location=lambda storage,
                            loc: storage)

        sd = comm.bcast(sd, root=0)

        return sd

    def set_variable_name(self, name):
        """
        set the variable name that will be inputed to the model
        """
        self.variable_name = str(name)
        if self.pred_name is None:
            self.set_pred_name(self.variable_name + '_pred')

    def set_pred_name(self, name):
        """
        set the variable name that will be the output to the model
        """
        self.pred_name = name

    def set_inference_function(self, fn):
        """
        set the final inference function. ex. torch.sigmoid()
        """
        self.inference_function = fn

    def input_preprocess(self, input_data):
        return input_data

    def output_postprocess(self, output_data):
        return output_data

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
        if n_requested > 0:
            # pass to torch
            torch.set_num_threads(n_requested)

        else:

            # detemrmine the number of physical cores are available
            # on this node, accounting for all MPI ranks scheduled to
            # run here.
            rank = 0
            n_ranks = 1
            comm = self.get_communicator()
            if teca_py.get_teca_has_mpi():
                rank = comm.Get_rank()
                n_ranks = comm.Get_size()

            try:
                n_threads, affinity = \
                     teca_py.thread_util.thread_parameters(comm, -1, 0, 0)

                # make use of hyper-threads
                n_threads *= self.threads_per_core

                # pass to torch
                torch.set_num_threads(n_threads)

            except(RuntimeError):
                # we failed to detect the number of physical cores per MPI rank
                # if this is an MPI job then fall back to 2 threads per rank and
                # if not let torch use all (what happens when you do nothing)
                n_threads = -1
                comm = self.get_communicator()
                if teca_py.get_teca_has_mpi() and comm.Get_size() > 1:
                    torch.set_num_threads(2)
                    n_threads = 2
                else:
                    pass
                if rank == 0:
                   sys.stderr.write('STATUS: Failed to determine the number of'
                                    ' physical cores available per MPI rank.'
                                    ' using %d threads per MPI rank.\n'%(
                                        n_threads))


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
        self.model = model
        self.model.eval()

    def report(self, port, rep_in):
        rep = teca_py.teca_metadata(rep_in[0])

        if rep.has('variables'):
            rep.append('variables', self.pred_name)
        else:
            rep.set('variables', self.pred_name)

        return rep

    def request(self, port, md_in, req_in):
        if not self.variable_name:
            raise RuntimeError("No variable to request specifed")

        req = teca_py.teca_metadata(req_in)

        arrays = []
        if req.has('arrays'):
            arrays = req['arrays']
            if type(arrays) != list:
                arrays = [arrays]

        # remove the arrays we produce
        try:
            arrays.remove(self.pred_name)
        except:
            pass

        # add the arrays we need
        arrays.append(self.variable_name)

        req['arrays'] = arrays

        return [req]

    def execute(self, port, data_in, req):
        """
        expects an array of an input variable to run through
        the torch model and get the segmentation results as an
        output.
        """
        in_mesh = teca_py.as_teca_cartesian_mesh(data_in[0])

        if in_mesh is None:
            raise RuntimeError('empty input, or not a mesh')

        if self.model is None:
            raise RuntimeError('A pretrained model has not been specified')

        if self.variable_name is None:
            raise RuntimeError('variable_name has not been specified')

        if self.var_array is None:
            raise RuntimeError('data variable array has not been set')

        if self.inference_function is None:
            raise RuntimeError('The inference function has not been set')

        self.var_array = self.input_preprocess(self.var_array)

        self.var_array = torch.from_numpy(self.var_array).to(self.device)

        with torch.no_grad():
            self.pred_array = \
                self.inference_function(self.model(self.var_array))

        if self.pred_array is None:
            raise RuntimeError("Model failed to get predictions")

        self.pred_array = self.output_postprocess(self.pred_array)

        out_mesh = teca_py.teca_cartesian_mesh.New()
        out_mesh.shallow_copy(in_mesh)

        self.pred_array = teca_py.teca_variant_array.New(self.pred_array)
        out_mesh.get_point_arrays().set(self.pred_name, self.pred_array)

        return out_mesh
