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
        self.torch_inference_fn = None
        self.transform_fn = None
        self.transport_fn_args = None
        self.model = None
        self.model_path = None
        self.device = 'cpu'

    def __str__(self):
        ms_str = 'variable_name=%s, pred_name=%s\n' % (
                  self.variable_name, self.pred_name)
        ms_str += 'model=%s\n' % (str(self.model))
        ms_str += 'model_path=%s\n' % (self.model_path)
        ms_str += 'device=%s\n' % (str(self.device))
        ms_str += 'torch_num_threads=%d\n' % (self.get_num_threads())
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

    def set_torch_inference_fn(self, torch_fn):
        """
        set the final torch inference function. ex. torch.sigmoid()
        """
        self.torch_inference_fn = torch_fn

    def input_preprocess(self, input_data):
        return input_data

    def output_postprocess(self, output_data):
        return output_data

    def __set_transform_fn(self, fn, *args):
        """
        if the data need to be transformed in a way then a function
        could be provided to be applied on the requested data before
        running it to the model.
        """
        if not hasattr(fn, '__call__'):
            raise TypeError(
                "ERROR: The provided data transform function"
                "is not a function"
                )

        if not args:
            raise ValueError(
                "ERROR: The provided data transform function "
                "must at least have 1 argument -- the data array object to "
                "apply the transformation on."
                )

        self.transform_fn = fn
        self.transport_fn_args = args

    def set_num_threads(self, n):
        """
        torch: Sets the number of threads used for intra-op parallelism on CPU
        """
        # n=-1: use default
        if n != -1:
            torch.set_num_threads(n)

    def get_num_threads(self):
        """
        torch: Gets the number of threads available for intra-op parallelism on CPU
        """
        return torch.get_num_threads()

    def set_torch_device(self, device="cpu"):
        """
        Set device to either 'cuda' or 'cpu'
        """
        if device[:4] == "cuda" and not torch.cuda.is_available():
            raise AttributeError(
                "ERROR: Couldn\'t set device to cuda, cuda is "
                "not available"
                )

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
            raise ValueError("No variable to request specifed")

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
            raise ValueError("ERROR: empty input, or not a mesh")

        if self.model is None:
            raise ValueError(
                "ERROR: pretrained model has not been specified"
                )

        if self.variable_name is None:
            raise ValueError(
                "ERROR: data variable name has not been specified"
                )

        if self.var_array is None:
            raise ValueError(
                "ERROR: data variable array has not been set"
                )

        if self.torch_inference_fn is None:
            raise ValueError(
                "ERROR: final torch inference layer"
                "has not been set"
                )

        self.var_array = self.input_preprocess(self.var_array)

        self.var_array = torch.from_numpy(self.var_array).to(self.device)

        with torch.no_grad():
            self.pred_array = self.torch_inference_fn(
                self.model(self.var_array)
            )

        if self.pred_array is None:
            raise Exception("ERROR: Model failed to get predictions")

        self.pred_array = self.output_postprocess(self.pred_array)

        out_mesh = teca_py.teca_cartesian_mesh.New()
        out_mesh.shallow_copy(in_mesh)

        self.pred_array = teca_py.teca_variant_array.New(self.pred_array)
        out_mesh.get_point_arrays().set(self.pred_name, self.pred_array)

        return out_mesh
