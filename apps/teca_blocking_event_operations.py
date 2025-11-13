# Operators for blocking event detection and tracking.
# Used in teca_blocking_detect_blobs.in and teca_blocking_stick_blobs.in
import numpy
import teca

if teca.get_teca_has_cupy():
    import cupy


class DetectGeopotentialHeightAnomalyOp:
    """
    Operator for detecting geopotential height anomalies in a dataset.

    This class compares the geopotential height field to a threshold field and generates
    a mask indicating where the anomaly condition is met (i.e., where the geopotential height
    is greater than or equal to the threshold).

    Attributes:
        geopotential_height_var (str): Name of the input array containing geopotential height values.
        geopotential_height_threshold_var (str): Name of the input array containing threshold values.
        mask_varname (str): Name of the output mask array to be created.
    """

    def __init__(self, geopotential_height_var, geopotential_height_threshold_var, mask_varname):
        """
        Initialize the operator with the specified variable names.

        Parameters
        ----------
        geopotential_height_var : str
            Name of the input array containing geopotential height values.
        geopotential_height_threshold_var : str
            Name of the input array containing threshold values.
        mask_varname : str
            Name of the output mask array to be created.
        """
        self.geopotential_height_var = geopotential_height_var
        self.geopotential_height_threshold_var = geopotential_height_threshold_var
        self.mask_varname = mask_varname

    def __call__(self, port, data_in, req):
        """
        Apply the anomaly detection operation to the input data.

        Handles both CPU and CUDA execution depending on the environment and request.
        Returns a mesh with the mask array added, where each element is 1 if the anomaly
        condition is met, 0 otherwise.

        Parameters
        ----------
        port : int
            The input port number (not used in this implementation).
        data_in : list
            List containing the input mesh data.
        req : dict
            Request dictionary that may contain 'device_id' for CUDA execution.

        Returns
        -------
        teca_cartesian_mesh
            The output mesh with the added mask array indicating anomalies.
        """
        dev = -1
        np = numpy
        if teca.get_teca_has_cuda() and teca.get_teca_has_cupy():
            dev = req.get('device_id', -1)
            if dev >= 0:
                cupy.cuda.Device(dev).use()
                np = cupy

        in_mesh = teca.as_teca_cartesian_mesh(data_in[0])
        out_mesh = teca.teca_cartesian_mesh.New()
        out_mesh.shallow_copy(in_mesh)
        arrays = out_mesh.get_point_arrays()

        geopotential_height_array = arrays[self.geopotential_height_var]
        geopotential_height_threshold_array = arrays[self.geopotential_height_threshold_var]

        if dev < 0:
            geopotential_height = geopotential_height_array.get_host_accessible()
            geopotential_height_threshold = geopotential_height_threshold_array.get_host_accessible()
        else:
            geopotential_height = geopotential_height_array.get_cuda_accessible()
            geopotential_height_threshold = geopotential_height_threshold_array.get_cuda_accessible()

        segmentation = (geopotential_height >= geopotential_height_threshold).astype(np.byte)
        arrays[self.mask_varname] = segmentation
        return out_mesh


class IntersectRegionsOp:
    """
    Operator for computing the intersection of two sets of labeled regions.

    Region intersections are encoded by concatenating both region IDs
    (region2 in the high 16 bits, region1 in the low 16 bits).

    Attributes:
        region_varname1 (str): Name of the first region variable.
        region_varname2 (str): Name of the second region variable.
        background_value (int): Value of the background in both input and output masks.
        intersection_varname (str): Name of the output mask array to be created.
    """

    def __init__(self, region_varname1, region_varname2, background_value, intersection_varname):
        """
        Initialize the operator with the specified variable names and background value.

        Parameters
        ----------
        region_varname1 : str
            Name of the first region variable.
        region_varname2 : str
            Name of the second region variable.
        background_value : int
            Value of the background in both input and output masks.
        intersection_varname : str
            Name of the output mask array to be created.
        """
        self.region_varname1 = region_varname1
        self.region_varname2 = region_varname2
        self.background_value = background_value
        self.intersection_varname = intersection_varname

    def __call__(self, port, data_in, req):
        """
        Compute the intersection of two sets of labeled regions in a mesh.

        Parameters
        ----------
        port : int
            The input port number (not used in this implementation).
        data_in : list
            List containing the input mesh data.
        req : dict
            Request dictionary that may contain 'device_id' for CUDA execution.

        Returns
        -------
        teca_cartesian_mesh
            A new mesh with the intersection. Region intersections are encoded
            by bit-packing both region IDs into a single int32 (region2 in the
            high 16 bits, region1 in the low 16 bits).
        """
        dev = -1
        np = numpy
        if teca.get_teca_has_cuda() and teca.get_teca_has_cupy():
            dev = req.get('device_id', -1)
            if dev >= 0:
                cupy.cuda.Device(dev).use()
                np = cupy

        in_mesh = teca.as_teca_cartesian_mesh(data_in[0])
        out_mesh = teca.teca_cartesian_mesh.New()
        out_mesh.shallow_copy(in_mesh)
        arrays = out_mesh.get_point_arrays()

        region_labels_1_array = arrays[self.region_varname1]
        region_labels_2_array = arrays[self.region_varname2]

        if dev < 0:
            region_labels_1_array = region_labels_1_array.get_host_accessible()
            region_labels_2_array = region_labels_2_array.get_host_accessible()
        else:
            region_labels_1_array = region_labels_1_array.get_cuda_accessible()
            region_labels_2_array = region_labels_2_array.get_cuda_accessible()

        # Pack intersecting region IDs into single int32: region1 in low 16 bits,
        # region2 in high 16 bits, each supporting up to 65535 unique regions.
        # If either region is background, the output is background.
        # This encoding allows efficient storage and later unpacking of region
        # pairs.
        intersection = np.where(
            (region_labels_1_array != self.background_value) &
            (region_labels_2_array != self.background_value),
            np.int32(region_labels_2_array) << 16 | region_labels_1_array,
            self.background_value
        ).astype(np.int32)

        arrays[self.intersection_varname] = intersection
        return out_mesh


class BinarizeComponentIdsOp:
    """
    Operator for binarizing component IDs in a dataset.

    This class converts labeled regions to a binary mask where each element is 1
    if it is part of a blocking event candidate (non-background), otherwise 0.

    Attributes:
        component_varname (str): Name of the input array containing component IDs.
        binary_varname (str): Name of the output binary mask array to be created.
        background_value (int): Value representing background in the component array.
    """

    def __init__(self, component_varname, binary_varname, background_value=0):
        """
        Initialize the operator with the specified variable names and background value.

        Parameters
        ----------
        component_varname : str
            Name of the input array containing component IDs.
        binary_varname : str
            Name of the output binary mask array to be created.
        background_value : int, optional
            Value representing background in the component array (default: 0).
        """
        self.component_varname = component_varname
        self.binary_varname = binary_varname
        self.background_value = background_value

    def __call__(self, port, data_in, req):
        """
        Apply the binarization operation to the input data.

        Handles both CPU and CUDA execution depending on the environment and request.
        Returns a mesh with the binary mask array added, where each element is 1
        if it is part of a blocking event candidate, otherwise 0.

        Parameters
        ----------
        port : int
            The input port number (not used in this implementation).
        data_in : list
            List containing the input mesh data.
        req : dict
            Request dictionary that may contain 'device_id' for CUDA execution.

        Returns
        -------
        teca_cartesian_mesh
            The output mesh with the added binary mask array.
        """
        dev = -1
        np = numpy
        if teca.get_teca_has_cuda() and teca.get_teca_has_cupy():
            dev = req.get('device_id', -1)
            if dev >= 0:
                cupy.cuda.Device(dev).use()
                np = cupy

        in_mesh = teca.as_teca_cartesian_mesh(data_in[0])
        out_mesh = teca.teca_cartesian_mesh.New()
        out_mesh.shallow_copy(in_mesh)
        arrays = out_mesh.get_point_arrays()

        component_array = arrays[self.component_varname]

        if dev < 0:
            component_data = component_array.get_host_accessible()
        else:
            component_data = component_array.get_cuda_accessible()

        # Create binary mask: 1 if component is not background, 0 otherwise
        binary_mask = (component_data != self.background_value).astype(np.byte)
        arrays[self.binary_varname] = binary_mask
        return out_mesh