import numpy as np

class teca_deeplab_ar_detect(teca_pytorch_algorithm):
    """
    This algorithm detects Atmospheric Rivers using deep learning techniques
    derived from the DeepLabv3+ architecture. Given an input field of
    integrated vapor transport (IVT) magnitude, it calculates the probability
    of an AR event and stores it in a new scalar field named 'ar_probability'.
    """
    def __init__(self):
        super().__init__()

        self.set_input_variable("IVT")

        arp_atts = teca_array_attributes(
            teca_float_array_code.get(), teca_array_attributes.point_centering,
            0, teca_array_attributes.xyt_active(), 'unitless', 'posterior AR flag',
            'the posterior probability of the presence of an atmospheric river',
            None)

        self.set_output_variable("ar_probability", arp_atts)

    def set_ivt_variable(self, var):
        """
        set the name of the variable containing the integrated vapor
        transport(IVT) magnitude field.
        """
        self.set_input_variable(var)

    def load_model(self, filename):
        """
        Load model from file system. In MPI parallel runs rank 0
        loads the model file and broadcasts it to the other ranks.
        """
        event = teca_time_py_event('teca_deeplab_ar_detect::load_model')

        # this creates OpenMP thread pools and imports torch
        # it must be called *before* we import torch
        self.initialize()

        # import our torch codes only now that torch has been initialized
        global teca_deeplab_ar_detect_internals
        from teca_deeplab_ar_detect_internals \
            import teca_deeplab_ar_detect_internals

        # create an instance of the model
        model = teca_deeplab_ar_detect_internals.DeepLabv3_plus(
            n_classes=1, _print=False)

        # load model weights from state on disk
        super().load_model(filename, model)

    def get_padding_sizes(self, div, dim):
        """
        given a divisor(div) and an input mesh dimension(dim)
        returns a tuple of values holding the number of values to
        add onto the low and high sides of the mesh to make the mesh
        dimension evely divisible by the divisor
        """
        # ghost cells in the y direction
        target_shape = div * np.ceil(dim / div)
        target_shape_diff = target_shape - dim

        pad_low = int(np.ceil(target_shape_diff / 2.0))
        pad_high = int(np.floor(target_shape_diff / 2.0))

        return pad_low, pad_high

    def preprocess(self, in_array):
        """
        resize the array to be a multiple of 64 in the y direction and 128 in
        the x direction amd convert to 3 channel (i.e. RGB image like)
        """
        event = teca_time_py_event('teca_deeplab_ar_detect::preprocess')

        nx_in = in_array.shape[1]
        ny_in = in_array.shape[0]

        # get the padding sizes to make the mesh evenly divisible by 64 in the
        # x direction and 128 in the y direction
        ng_x0, ng_x1 = self.get_padding_sizes(64.0, nx_in)
        ng_y0, ng_y1 = self.get_padding_sizes(128.0, ny_in)

        nx_out = ng_x0 + ng_x1 + nx_in
        ny_out = ng_y0 + ng_y1 + ny_in

        # allocate a new larger array
        out_array = np.zeros((1, 3, ny_out, nx_out), dtype=np.float32)

        # copy the input array into the center
        out_array[:, :, ng_y0 : ng_y0 + ny_in,
                  ng_x0 : ng_x0 + nx_in] = in_array

        # cache the padding info in order to extract the result
        self.ng_x0 = ng_x0
        self.ng_y0 = ng_y0
        self.nx_in = nx_in
        self.ny_in = ny_in

        return out_array

    def postprocess(self, out_tensor):
        """
        convert the tensor to a numpy array and extract the output data from
        the padded tensor. padding was added during preprocess.
        """
        event = teca_time_py_event('teca_deeplab_ar_detect::postprocess')

        # normalize the output
        tmp = torch.sigmoid(out_tensor)

        # move to the CPU if running on a GPU
        if self.device != 'cpu':
            tmp = tmp.to('cpu')

        # convert from torch tensor to numpy ndarray
        out_array = tmp.numpy()

        # extract the valid portion of the result
        out_array = out_array[:, :, self.ng_y0 : self.ng_y0 + self.ny_in,
                              self.ng_x0 : self.ng_x0 + self.nx_in]

        return out_array
