import numpy


class teca_python_vertical_reduction(teca_python_algorithm):
    """
    The base class used for writing new vertical reduction algorithms
    in Python. Contains plumbing that connects user provided overrides
    to an instance of teca_vertical_reduction. Users are expected to
    override one or more of report, request, and/or execute.
    """

    derived_variables_attributes = {}
    dependent_variables = []

    def set_dependent_variables(self, variables_list):
        if type(variables_list) is str:
            variables_list = [variables_list]

        self.dependent_variables = variables_list

    def get_derived_variables(self):
        return list(self.derived_variables_attributes)

    def add_derived_variable_and_attributes(self, var, attributes):
        self.derived_variables_attributes[var] = attributes

    def report(self, port, input_md):
        """ Define the TECA report phase for this algorithm"""
        out_md = teca_metadata(input_md[0])

        attributes = out_md["attributes"]

        # set the variables and their attributes
        for var in self.get_derived_variables():
            attributes[var] = self.derived_variables_attributes[var].\
                              to_metadata()
        out_md["attributes"] = attributes

        # get the input extents
        whole_extent = out_md["whole_extent"]

        # set the output extent, with vertical dim reduced
        whole_extent[4] = whole_extent[5] = 0
        out_md["whole_extent"] = whole_extent

        # fix bounds if it is present
        bounds = out_md["bounds"]
        bounds[4] = bounds[5] = 0.0
        out_md["bounds"] = bounds

        return out_md

    def request(self, port, input_md, request):
        """ Define the TECA request phase for this algorithm"""

        up_reqs = []

        # copy the incoming request to preserve the downstream
        # requirements and add the arrays we need
        req = teca_metadata(request)

        # transform extent, add back the vertical dimension
        md = teca_metadata(input_md[0])

        # get the whole extent and bounds
        bounds = numpy.zeros(6, dtype=numpy.int32)
        whole_extent = numpy.zeros(6, dtype=numpy.int32)
        bounds = md["bounds"]
        whole_extent = md["whole_extent"]

        try:
            bounds_up = request["bounds"]
            has_bounds = True
        except KeyError:
            has_bounds = False

        try:
            extent_up = request["extent"]
            has_extent = True
        except KeyError:
            has_extent = False

        # restore vertical bounds
        if has_bounds:
            bounds_up[4] = bounds[4]
            bounds_up[5] = bounds[5]
            req["bounds"] = bounds_up
        # restore vertical extent
        elif has_extent:
            extent_up[4] = whole_extent[4]
            extent_up[5] = whole_extent[5]
            req["extent"] = extent_up
        # no subset requested, request all the data
        else:
            req["extent"] = whole_extent

        # get the list of variable available. we need to see if
        # the valid value mask is available and if so request it
        variables = md["variables"]

        # add the dependent variables into the requested arrays
        arrays = req["arrays"]

        # convert the array list to a list if needed
        if type(arrays) is str:
            arrays = [arrays]

        for dep_var in self.dependent_variables:
            # request the array needed for the calculation
            arrays.append(dep_var)

            # request the valid value mask if they are available.
            mask_var = dep_var + "_valid"
            if (mask_var in variables):
                arrays.append(mask_var)

        # capture the arrays we produce
        for var in self.get_derived_variables():
            arrays.remove(var)

        # update the request
        req["arrays"] = arrays

        # send it up
        up_reqs.append(req)

        return up_reqs

    def execute(self, port, input_data, request):
        """ Define the TECA execute phase for this algorithm"""
        # get the input mesh
        in_mesh = as_const_teca_cartesian_mesh(input_data[0])

        # construct the output
        out_mesh = as_const_teca_cartesian_mesh(in_mesh.new_instance())

        # copy metadata
        out_mesh.copy_metadata(in_mesh)

        # fix the metadata
        out_md = teca_metadata(out_mesh.get_metadata())

        # fix whole extent
        try:
            whole_extent = out_md["whole_extent"]
            whole_extent[4] = whole_extent[5] = 0
            out_md["whole_extent"] = whole_extent
        except KeyError:
            pass

        # fix extent
        try:
            extent = out_md["extent"]
            extent[4] = extent[5] = 0
            out_md["extent"] = extent
        except KeyError:
            pass

        # fix bounds
        try:
            bounds = out_md["bounds"]
            bounds[4] = bounds[5] = 0
            out_md["bounds"] = bounds
        except KeyError:
            pass

        out_mesh.set_metadata(out_md)

        # fix the z axis
        cart_mesh = as_const_teca_cartesian_mesh(out_mesh)

        z_var = cart_mesh.get_z_coordinate_variable()

        in_z = cart_mesh.get_z_coordinates()

        out_z = in_z.new_instance(1)
        cart_mesh.set_z_coordinates(z_var, out_z)

        return out_mesh
